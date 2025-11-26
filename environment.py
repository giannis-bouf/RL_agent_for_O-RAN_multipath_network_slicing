import numpy as np
import gymnasium as gym
from arrival import SliceReqGenerator
import logging
import pickle
import copy

class NetworkEnv(gym.Env):

    def __init__(self, Fs_max=4, n_DUs=2, n_CUs=2, c_DU=8, c_CU=32, C_max=2,
                 P_cpu=3, P_start=16, theta_idle=0.4, Gamma=12,
                 H=12, Dt=1, request_rate=1, training_mode = True, dataset = 1):
        self.horizon_length = H     # Horizon equal to total number of time slots

        self.c_DU = c_DU            # DU computational capacity
        self.c_CU = c_CU            # CU computational capacity
        self.n_DUs = n_DUs          # Number of DUs
        self.n_CUs = n_CUs          # Number of CUs
        self.Fs_max = Fs_max        # Number of maximum SFC size
        self.C_max = C_max          # Number of maximum instances per VNF

        self.P_cpu = P_cpu              # Power consumption per CPU unit
        self.P_start = P_start          # Cold-start power overhead
        self.theta_idle = theta_idle    # Fraction of Power Consumption when CU in idle mode
        self.llambda = [np.random.uniform(0.3, 0.7) for _ in range (self.n_CUs)]    # Normalized workload per CU
        # Workload (Gaussian) distribution per CU
        self.rho_lambda = [np.clip(np.random.normal(self.llambda[idx], 0.1), 0.0, 1.0) for idx in range(self.n_CUs)]

        self.CPU_per_instance = [0] + [1]*(Fs_max-1)    # Number of CPU demand per instance per VNF
        # FH/BH latency per link (ιδια προς το παρον)
        self.FH_latency = [int(np.random.uniform(3, 8)) for _ in range(self.n_DUs)]
        self.BH_latency = np.random.uniform(12, 40, size=(self.n_DUs, self.n_CUs)).astype(int)
        self.Gamma = Gamma                              # "Severity of delay violations" constant

        # Weights in f(a_k)-action encoding function per block
        self.cu_w = (self.C_max + 1)**(self.n_DUs*(self.Fs_max-1))                    # "Allocation in CUs - X^{CU}" block
        self.p_w  = self.cu_w * ((self.C_max + 1)**(self.n_CUs*(self.Fs_max-1)))      # "Path" block
        self.sp_w = self.p_w * (((self.n_DUs + 1)*(self.n_CUs + 1)) ** self.C_max)    # "Split point" block

        # Reward function coefficients
        self.reward_coeffs = [1,2,3]
        
        if training_mode:
            self.dataset_file = f"dataset/historic_data.pkl"
            self.total_episodes = 1000
        else:
            self.dataset_file = f"dataset/test_data.pkl"
            self.total_episodes = 10

        with open(self.dataset_file, "rb") as f:
            scenarios = pickle.load(f)

        self.scenarios = scenarios
        self.scenarios_idx = 0

        self.current_scenario = [None] * self.horizon_length

        for sl in self.scenarios[self.scenarios_idx].arriving_reqs:
            if self.current_scenario[sl[1]] == None:
                self.current_scenario[sl[1]] = [sl.copy()]
            else:
                self.current_scenario[sl[1]].append(sl.copy())

        self.current_scenario_active_slices = self.scenarios[self.scenarios_idx].active
        self.scenario_reward = 0
        self.score_history = []

        # Available computing resources as function of time
        self.DU_av_capacity = np.full(shape=(self.horizon_length, self.n_DUs, 1), fill_value=self.c_DU, dtype=int)
        self.CU_av_capacity = np.full(shape=(self.horizon_length, self.n_CUs, 1), fill_value=self.c_CU, dtype=int)

        # Remaining uptime per DU/CU
        # self.DU_uptime = np.zeros(shape=(self.n_DUs, 1), dtype=np.float64)
        # self.CU_uptime = np.zeros(shape=(self.n_CUs, 1), dtype=np.float64) 

        # Node computational utilization as function of time
        self.DU_utilization = np.full(shape=(self.horizon_length, self.n_DUs, 1), fill_value=0, dtype=np.float64)
        self.CU_utilization = np.full(shape=(self.horizon_length, self.n_CUs, 1), fill_value=0, dtype=np.float64)

        # Node power consumption as function of time
        self.DU_phi = np.full(shape=(self.horizon_length, self.n_DUs, 1), fill_value=0, dtype=np.float64)
        self.CU_phi = np.full(shape=(self.horizon_length, self.n_CUs, 1), fill_value=0, dtype=np.float64)

        self.timestep = 0
        while self.current_scenario[self.timestep] == None:
            self.timestep += 1
        self.current_slice = self.current_scenario[self.timestep].pop()
        
        """
        State s_k = <AC_k=[AC_1(k), ..., AC_{|E|+|C|}(k)],
                     #### AT_k=[AT_1(k), ..., AT_{|E|+|C|}(k)], 
                     SI_k=[F_s, C_s(t_k), delta_s, ht_s], 
                     t_k>

        Observation:
            - AC_k.size = n_DUs + n_CUs
              0 <= AC_i <= c_DU, if i = DU
              0 <= AC_i <= c_CU, if i = CU
            ###- AT_k.size = n_DUs + n_CUs
            ###  0 <= AT_i <= H-1
            - SI_k
              3 <= F_s <= Fs_max, size: 1
              C_{0,s} = 1, 1 <= C_s(t_k) <= C_max, size: Fs_max (padding the remaining ones)
              25 <= delta_s <= 100, size: 1
              0 <= ht_s <= H/2, size: 1 
        """
        #state_space_dim = 2*(self.n_DUs + self.n_CUs) + self.Fs_max + 3
        state_space_dim = self.n_DUs + self.n_CUs + self.Fs_max + 3
        lower_bds = []
        upper_bds = []

        # AC_k
        lower_bds += [0] * (self.n_DUs + self.n_CUs)
        upper_bds += [self.c_DU]*self.n_DUs + [self.c_CU]*self.n_CUs 

        # AT_k
        # lower_bds += [0] * (self.n_DUs + self.n_CUs)
        # upper_bds += [self.horizon_length - 1] * (self.n_DUs + self.n_CUs)

        # SI_k
        lower_bds += [3] #F_s
        upper_bds += [self.Fs_max]

        lower_bds += [1]*3 + [0]*(self.Fs_max-3) #C_s(t_k)        # Εδω τι γινεται εφοσον εχουμε μεταβλητο Fs
        upper_bds += [1] + [self.C_max]*(self.Fs_max-1)

        lower_bds += [25] #delta_s
        upper_bds += [100]

        lower_bds += [0] #ht_s
        upper_bds += [int(self.horizon_length/2 - 1)]

        lower_bds = np.array(lower_bds)
        upper_bds = np.array(upper_bds)

        self.observation_space = gym.spaces.Box(
            low=lower_bds, high=upper_bds, shape=(state_space_dim,), dtype=np.float64
        )

        """
        Action a_k = <X^{DU}(k), X^{CU}(k), P(k)>

        - X^{DU}(k)
          n_DUs x (Fs_max-1) entries, 0 <= X^{DU}_{i,j} <= C_max
          size: (C_max + 1)^(n_DUs * (Fs_max-1))
        - X^{CU}(k)
          n_CUs x (Fs_max-1) entries, 0 <= X^{CU}_{i,j} <= C_max
          size: (C_max + 1)^(n_CUs * (Fs_max-1))
        - P(k)
          C_max rows
          each row: (n_DUs + 1)*(n_CUs + 1) combinations
          size: ((n_DUs + 1)*(n_CUs + 1))^C_max
        """
        #self.action_space_dim = ((self.C_max + 1) ** ((self.n_DUs + self.n_CUs) * (self.Fs_max-1))) \
        #                        * (((self.n_DUs + 1)*(self.n_CUs + 1)*self.Fs_max) ** self.C_max)
        self.action_space_dim = ((self.C_max + 1) ** ((self.n_DUs + self.n_CUs) * (self.Fs_max-1))) \
                                * (((self.n_DUs + 1)*(self.n_CUs + 1)) ** self.C_max)
        self.action_space = gym.spaces.Discrete(self.action_space_dim)

    def reset(self, seed=None, options=None):
        self.scenarios_idx += 1
        if self.scenarios_idx == len(self.scenarios):
            with open(self.dataset_file, "rb") as f:
                scenarios = pickle.load(f)

            self.scenarios = scenarios
            self.scenarios_idx = 0

        self.current_scenario = [None] * self.horizon_length
        self.scenario_reward = 0

        for sl in self.scenarios[self.scenarios_idx].arriving_reqs:
            if self.current_scenario[sl[1]] == None:
                self.current_scenario[sl[1]] = [sl.copy()]
            else:
                self.current_scenario[sl[1]].append(sl.copy())

        self.current_scenario_active_slices = self.scenarios[self.scenarios_idx].active

        # Available computing resources as function of time
        self.DU_av_capacity = np.full(shape=(self.horizon_length, self.n_DUs, 1), fill_value=self.c_DU, dtype=int)
        self.CU_av_capacity = np.full(shape=(self.horizon_length, self.n_CUs, 1), fill_value=self.c_CU, dtype=int)

        # Remaining uptime per DU/CU
        # self.DU_uptime = np.zeros(shape=(self.n_DUs, 1), dtype=np.float64)
        # self.CU_uptime = np.zeros(shape=(self.n_CUs, 1), dtype=np.float64) 

        # Node computational utilization as function of time
        self.DU_utilization = np.full(shape=(self.horizon_length, self.n_DUs, 1), fill_value=0, dtype=np.float64)
        self.CU_utilization = np.full(shape=(self.horizon_length, self.n_CUs, 1), fill_value=0, dtype=np.float64)

        # Node power consumption as function of time
        self.DU_phi = np.full(shape=(self.horizon_length, self.n_DUs, 1), fill_value=0, dtype=np.float64)
        self.CU_phi = np.full(shape=(self.horizon_length, self.n_CUs, 1), fill_value=0, dtype=np.float64)

        self.timestep = 0
        while self.current_scenario[self.timestep] == None:
            self.timestep += 1
        self.current_slice = self.current_scenario[self.timestep].pop()

        return self.get_state(), {}
    
    def get_state(self):
        t_s = self.current_slice[1]
        C_s_tk = np.array([self.current_slice[4][k][self.timestep - t_s] for k in range(self.current_slice[0])])

        # Padding with 0 if F_s < F_s_max
        if self.current_slice[0] < self.Fs_max:
            C_s_tk = np.concatenate([C_s_tk, np.zeros(self.Fs_max - self.current_slice[0])])

        state = np.concatenate([
            self.DU_av_capacity[self.timestep].flatten(),
            self.CU_av_capacity[self.timestep].flatten(),
            # self.DU_uptime[self.timestep].flatten(),
            # self.CU_uptime[self.timestep].flatten(),
            np.array([self.current_slice[0]]),              # F_s
            C_s_tk,                                         # C_s(t_k)
            np.array([self.current_slice[3]]),              # delta_s
            np.array([self.current_slice[2]]),              # ht_s
            np.array([self.timestep])                       # t_k
        ])
        
        return state
    
    def get_info(self):
        info = {
            'time': self.timestep,
            'active': self.current_scenario_active_slices
        }
        return info

    def transition(self, action):
        current_state =  self.get_state()
        reward = float('-inf')

        if action == 0:
            if (self.timestep < self.horizon_length - 1) and \
               (self.timestep < self.current_slice[1] + self.current_slice[2]):
                if self.current_scenario[self.timestep + 1] == None:
                    self.current_scenario[self.timestep + 1] = [self.current_slice.copy()]
                else:
                    self.current_scenario[self.timestep + 1].append(self.current_slice.copy())
        else:
            reward = 0

            # Finding encoding for each block - Factorization of action f(a_k)
            encoded_p  = action // self.p_w;     rem = action % self.p_w
            encoded_cu = rem // self.cu_w
            encoded_du = rem % self.cu_w

            # Decode each number in order to find the respective matrix
            P_k    = self.decode_path_matrix(encoded_p)
            X_DU_k = self.decode_allocation_DU(encoded_du)
            X_CU_k = self.decode_allocation_CU(encoded_cu)

            deployment_in_RC = any(P_k[p][1] > 0 for p in range(self.C_max))
            # min/max utilization across all nodes
            max_util = 0; min_util = np.inf

            # for every time step from current time until slice's termination
            for t in range(self.current_slice[1] + self.current_slice[2] - self.timestep):
                for du in range(self.n_DUs):
                    # If t = t_0 or node idle in t-1: cold start overhead may be needed
                    prev_phi = 0 if self.timestep + t == 0 else self.DU_phi[self.timestep + t - 1][du][0]
                    cold_start = (prev_phi == 0)

                    for k in range(self.current_slice[0]-1):
                        # Reduce the available resources according to the demand of each VNF on each node
                        self.DU_av_capacity[self.timestep + t][du][0] -= X_DU_k[du][k] * self.CPU_per_instance[k+1]
                        # Increase the power consumption of this node accordingly
                        self.DU_phi[self.timestep + t][du][0] += X_DU_k[du][k] * self.CPU_per_instance[k+1] * self.P_cpu
                    
                    # If cold start was considered and node is active now: apply cold start overhead
                    if self.DU_phi[self.timestep + t][du][0] > 0 and cold_start:
                        self.DU_phi[self.timestep + t][du][0] += self.P_start

                    # u^i_{node}(t) = 1 - (AC_i(t)/c_i)
                    self.DU_utilization[self.timestep + t][du][0] = 1 - (self.DU_av_capacity[self.timestep + t][du][0] / self.c_DU)
                    max_util = np.maximum(max_util, self.DU_utilization[self.timestep + t][du][0])
                    min_util = np.minimum(min_util, self.DU_utilization[self.timestep + t][du][0])
                    
                    # DU is active <=> VNF 1 is deployed there. We update the remaining up time accordingly
                    # if X_DU_k[du][0] > 0 and \
                    #   (self.current_slice[2]-(self.timestep - self.current_slice[1])) > self.DU_uptime[du][0]:
                    #    self.DU_uptime[du][0] = self.current_slice[2]-(self.timestep - self.current_slice[1])

                # If we will deploy VNFs in the Regional Cloud, we act accordingly for CUs
                if deployment_in_RC:
                    for cu in range(self.n_CUs):
                        prev_phi = 0 if self.timestep + t == 0 else self.CU_phi[self.timestep + t - 1][cu][0]
                        cold_start = (prev_phi == 0)
                        # During computing CU power consumption, we have to take into consideration their distributed workload
                        E_lambda = (self.theta_idle + self.llambda[cu] * (1-self.theta_idle)) * self.rho_lambda[cu]

                        for k in range(self.current_slice[0]-1):
                            self.CU_av_capacity[self.timestep + t][cu][0] -= X_CU_k[cu][k] * self.CPU_per_instance[k+1]

                            self.CU_phi[self.timestep + t][cu][0] += X_CU_k[cu][k] * self.CPU_per_instance[k+1] * self.P_cpu * E_lambda

                        if self.CU_phi[self.timestep + t][cu][0] > 0 and cold_start:
                            self.CU_phi[self.timestep + t][cu][0] += self.P_start
                        
                        self.CU_utilization[self.timestep + t][cu][0] = 1 - (self.CU_av_capacity[self.timestep + t][cu][0] / self.c_CU)
                        max_util = np.maximum(max_util, self.CU_utilization[self.timestep + t][cu][0])
                        min_util = np.minimum(min_util, self.CU_utilization[self.timestep + t][cu][0])
                        
                        # if X_CU_k[cu][self.current_slice[0]-1-1] > 0 and \
                        #    (self.current_slice[2]-(self.timestep - self.current_slice[1])) > self.CU_uptime[cu][0]:
                        #    self.CU_uptime[cu][0] = self.current_slice[2]-(self.timestep - self.current_slice[1])

            # Power consumption for each moment in current timestep
            for du in range(self.n_DUs):
                reward -= self.reward_coeffs[0] * self.DU_phi[self.timestep][du][0]
            for cu in range(self.n_CUs):
                reward -= self.reward_coeffs[0] * self.CU_phi[self.timestep][cu][0]

            # Maximum load imbalance
            reward -= self.reward_coeffs[1] * (max_util - min_util)

            # Compute the max E2E delay experienced across all paths using the path matrix P_k
            max_delay = 0
            for p in range(self.C_max):
                delay = 0
                if P_k[p][0] > 0:
                    delay += self.FH_latency[P_k[p][0]-1]
                    if P_k[p][1] > 0:
                        delay += self.BH_latency[P_k[p][0]-1][P_k[p][1]-1]
                max_delay = np.maximum(max_delay, delay)
            
            # If SLA violation occurs, apply proportionate penalty
            if max_delay > self.current_slice[3]:
                reward -= self.reward_coeffs[2] * (max_delay - self.current_slice[3]) * self.Gamma

        self.scenario_reward += reward
        terminated = False

        dt = 0
        while self.current_scenario[self.timestep] == None or self.current_scenario[self.timestep] == []:
            self.timestep += 1
            dt += 1
            if self.timestep == self.horizon_length:
                terminated = True
                self.score_history.append(self.scenario_reward)
                self.timestep = 0
                break

        if not terminated:
            self.current_slice = self.current_scenario[self.timestep].pop()
            # self.DU_uptime = np.maximum(self.DU_uptime - dt, 0)
            # self.CU_uptime = np.maximum(self.CU_uptime - dt, 0)

        next_state = self.get_state()

        return next_state, reward, terminated, False, self.get_info()

    def decode_split_matrix(self, encoded_sp):
        """
        Encoding used: sum_{i=1}^{C_{max}} SP(k){i} * (Fs_max)^{i-1}
        Returns list SP of length C_max
        SP[i] = digit i of encoded_sp in base Fs_max
        """
        base = self.Fs_max
        SP = [0] * self.C_max
        temp = encoded_sp
        for i in range(self.C_max):
            SP[i] = temp % base
            temp //= base

        return SP

    def decode_path_matrix(self, encoded_p):
        """
        Encoding used: sum_{i=1...C_max} (P(k){i,0}*(n_CUs+1) + P(k){i,1}) * ((n_DUs+1)(n_CUs+1))^(i-1)
        Decode encoded path matrix into P(k) with shape C_max*2
        P[i][0] = DU index in [0,n_DUs]
        P[i][1] = CU index in [0,n_CUs]
        """
        base = (self.n_DUs+1) * (self.n_CUs+1)
        P = [[0,0] for _ in range(self.C_max)]

        temp = encoded_p
        for i in range(self.C_max):
            var = temp % base
            temp //= base

            P[i][0] = var // (self.n_CUs + 1)
            P[i][1] = var %  (self.n_CUs + 1)

        return P
    
    def decode_allocation_matrix(self, encoded_x, N):
        """
        Encoding used: sum_{i=1...n__} sum_{j=1...Fs_max-1} X_s(k)_{i,j} * (C_max+1)^{(i-1)(Fs_max-1)+(j-1)}
        Decoder for allocation matrices X^{DU}(k) and X^{CU}(k) with shape n_DUs*(Fs_max-1) and n_CUs*(Fs_max-1), respectively
        X{i,j} in [0,C_max] which denotes how many replicas of VNF j+1 does node i hosts
        """
        X = [[0]*(self.Fs_max-1) for _ in range(N)]
        temp = encoded_x

        for i in range(N):
            for j in range(self.Fs_max-1):
                X[i][j] = temp % (self.C_max + 1)
                temp //= (self.C_max + 1)

        return X
    
    def decode_allocation_DU(self, encoded_du):
        return self.decode_allocation_matrix(encoded_du, self.n_DUs)
    
    def decode_allocation_CU(self, encoded_cu):
        return self.decode_allocation_matrix(encoded_cu, self.n_CUs)
    
    def is_action_valid(self, action):
        if action % 1000000 == 0:
            print("Action is: ", action)

        # Factorization of action f(a_k)
        encoded_p  = action // self.p_w;     rem = action % self.p_w
        encoded_cu = rem // self.cu_w
        encoded_du = rem % self.cu_w

        # Path matrix associated with action
        P = self.decode_path_matrix(encoded_p)
        active = np.array([False] * (self.n_DUs + self.n_CUs))
        for j in range(2):
            for i in range(self.C_max):
                if P[i][j] > 0:
                    active[j*self.n_DUs + P[i][j]-1] = True
        

        # Replica DU allocation matrix associated with action
        X_DU = self.decode_allocation_DU(encoded_du)
        for du in range(self.n_DUs):
            # If an inactive DU hosts at least one replica of any VNf -> Invalid action
            if not active[du]:
                for k in range(self.Fs_max-1):
                    if X_DU[du][k] > 0:
                        return False
            else:
                # If DU doesn't host VNF 1 or all sequenced VNFs -> Invalid action
                if X_DU[du][0] == 0:
                    return False
                split = False
                for k in range(self.current_slice[0]-1):
                    if X_DU[du][k] == 0:
                        split = True
                    if split and X_DU[du][k] == 1:
                        return False
                # If any replica with VNF index beyond the range of SFC is allocated -> Invalid action
                for k in range(self.current_slice[0]-1, self.Fs_max-1):
                    if X_DU[du][k] > 0:
                        return False
                # If the total computational demand is greater that the available capacity -> Invalid action
                required_capacity = sum(X_DU[du][k] * self.CPU_per_instance[k+1] for k in range(self.current_slice[0]-1))
                if self.DU_av_capacity[self.timestep][du][0] < required_capacity:
                    return False
        
        # The same constraints apply for the Regional Cloud
        # Replica CU allocation matrix associated with action
        X_CU = self.decode_allocation_CU(encoded_cu)
        for cu in range(self.n_CUs):
            if not active[self.n_DUs + cu]:
                for k in range(self.Fs_max-1):
                    if X_CU[cu][k] > 0:
                        return False
            else :
                # If DU doesn't host VNF Fs-1 -> Invalid action
                if X_CU[cu][self.current_slice[0]-2] == 0:
                    return False
                split = False
                for k in range(self.current_slice[0]-1):
                    if X_CU[cu][k] == 1:
                        split = True
                    if split and X_CU[cu][k] == 0:
                        return False
                for k in range(self.current_slice[0]-1, self.Fs_max-1):
                    if X_CU[cu][k] > 0:
                        return False
                required_capacity = sum(X_CU[cu][k] * self.CPU_per_instance[k+1] for k in range(self.current_slice[0]-1))
                if self.CU_av_capacity[self.timestep][cu][0] < required_capacity:
                    return False
            
        # If the minimum requirement of replicas count is not met or
        # if the allocated replicas exceed the maximum number of replicas per VNF
        # -> Invalid action
        if action > 0:
            for k in range(self.current_slice[0]-1):
                allocated_replicas = sum(X_DU[du][k] for du in range(self.n_DUs)) \
                                        + sum(X_CU[cu][k] for cu in range(self.n_CUs))
                required_replicas = self.current_slice[4][k+1][self.timestep - self.current_slice[1]]
                if self.C_max < allocated_replicas or allocated_replicas < required_replicas:
                    return False
        
        inactive_paths_prior = False
        for p in range(self.C_max):
            # If an inactive path is found before any active -> Invalid action
            # -To limit duplicate actions-
            if action > 0:
                if P[p][0] == 0:  
                    inactive_paths_prior = True
                else:
                    if inactive_paths_prior:
                        return False

            # If in a path a DU is not defined, but a CU is -> Invalid action
            if P[p][0] == 0:
                if P[p][1] != 0:
                    return False
            else:
                # If a CU is not defined in a path, all VNFs of the SFC 
                # must be deployed on the corresponding DU
                # Else -> Invalid action
                if P[p][1] == 0:
                    for k in range(self.current_slice[0]-1):
                        if X_DU[P[p][0]-1][k] == 0:
                            return False
                # In a path where both a DU and a CU are defined, each VNF of the SFC
                # must be assigned to exactly one of them
                else:
                    for k in range(self.current_slice[0]-1):
                        placed_in_EC = (X_DU[P[p][0] - 1][k] > 0)
                        placed_in_RC = (X_CU[P[p][1] - 1][k] > 0)
                        if placed_in_EC + placed_in_RC != 1:
                            return False

            for p2 in range(p+1, self.C_max):
                # If there are duplicate paths -> Invalid action
                if (P[p2][0] != 0 and P[p][0] == P[p2][0]) and \
                   (P[p][1] == P[p2][1]):
                    return False


        print(f"\nValid action: {action}")
        print("X_DU =", X_DU)
        print("X_CU =", X_CU)
        print("P    =", P)
        print(active)

        return True




    def invalid_action_masking(self):
        mask = np.array([None] * self.action_space_dim)
        print("Starting masking!")
        print(self.action_space_dim)
        #self.current_slice = [4]
        #self.req = [1,2,1]

        for action in range(self.action_space_dim):
            mask[action] = self.is_action_valid(action)

        return mask

if __name__ == "__main__":
    env = NetworkEnv()
    mask = env.invalid_action_masking()
    
    print("Shape of mask:", mask.shape)
    print("Number of valid actions:", np.sum(mask))
    print("Number of invalid actions:", np.sum(np.logical_not(mask)))