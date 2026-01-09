import numpy as np
import gymnasium as gym
from arrival import SliceReqGenerator
import logging
import pickle
import copy

class NetworkEnv(gym.Env):

    def __init__(self, Fs_max=4, n_DUs=8, n_CUs=2, c_DU=16, c_CU=64, C_max=2,
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
        # FH/BH latency per link (ms)
        # Each entry corresponds to the fronthaul delay between a DU and the RU
        self.FH_latency = [4, 5, 6, 7, 5, 8, 7, 6]
        # Each row represents a DU and each column corresponds to a CU
        # Therefore, the matrix has dimensions n_DUs × n_CUs, capturing backhaul delays
        # between every DU–CU pair
        self.BH_latency = np.array([ [12, 26], [28, 14], [16, 30], [32, 16],
                                     [20, 34], [36, 22], [24, 38], [26, 40] ])
        self.Gamma = Gamma                              # "Severity of delay violations" constant

        # Reward function coefficients
        self.reward_coeffs = [1,0,0]
        
        if training_mode:
            self.dataset_file = f"datasets/dataset{dataset}/historic_data.pkl"
            self.total_episodes = 1000
        else:
            self.dataset_file = f"datasets/dataset{dataset}/test_data.pkl"
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

        # Node computational utilization as function of time
        self.DU_utilization = np.full(shape=(self.horizon_length, self.n_DUs, 1), fill_value=0, dtype=np.float64)
        self.CU_utilization = np.full(shape=(self.horizon_length, self.n_CUs, 1), fill_value=0, dtype=np.float64)

        # Node power consumption as function of time
        self.DU_phi = np.full(shape=(self.horizon_length, self.n_DUs, 1), fill_value=0, dtype=np.float64)
        self.CU_phi = np.full(shape=(self.horizon_length, self.n_CUs, 1), fill_value=0, dtype=np.float64)

        # Path-allocation matrix
        self.assigned_paths = np.full(self.C_max*2, fill_value=0, dtype=int)
        # current VNF index
        self.vnf_idx = 1

        self.timestep = 0
        while self.current_scenario[self.timestep] == None:
            self.timestep += 1
        self.current_slice = self.current_scenario[self.timestep].pop()
        
        """
        State s_k = <AC_k=[AC_1(k), ..., AC_{|E|+|C|}(k)],
                     P_k = [P_1(k)_DU, P_1(k)_CU, ..., P_C_max(k)_DU, P_C_max(k)_CU]
                     SI_k=[F_s, i_k, C_{i,s}(t_k), delta_s, ht_s], 
                     t_k>

        Observation:
            - AC_k.size = n_DUs + n_CUs
              0 <= AC_i <= c_DU, if i = DU
              0 <= AC_i <= c_CU, if i = CU
            - P_k.size = 2*C_max
              0 <= P_i(k)_DU <= n_DUs
              0 <= P_i(k)_CU <= n_CUs
            - SI_k
              3 <= F_s <= Fs_max, size: 1
              1 <= i_k <= Fs_max, size: 1
              1 <= C_{i,s}(t_k) <= C_max, size: 1
              25 <= delta_s <= 100, size: 1
              0 <= ht_s <= H/2, size: 1 
              0 <= t_k <= H-1, size: 1
        """
        state_space_dim = self.n_DUs + self.n_CUs + (2*self.C_max) + 6
        lower_bds = []
        upper_bds = []

        # AC_k
        lower_bds += [0] * (self.n_DUs + self.n_CUs)
        upper_bds += [self.c_DU]*self.n_DUs + [self.c_CU]*self.n_CUs

        # P_k
        for _ in range(self.C_max):
            lower_bds += [0, 0]
            upper_bds += [self.n_DUs, self.n_CUs]

        # SI_k
        lower_bds += [3] #F_s
        upper_bds += [self.Fs_max]

        lower_bds += [1] #i_k
        upper_bds += [self.Fs_max]

        lower_bds += [1] #C_{i,s}(t_k)
        upper_bds += [self.C_max]

        lower_bds += [25] #delta_s
        upper_bds += [100]

        lower_bds += [0] #ht_s
        upper_bds += [int(self.horizon_length/2 - 1)]

        lower_bds += [0] #t_k
        upper_bds += [self.horizon_length - 1]

        lower_bds = np.array(lower_bds)
        upper_bds = np.array(upper_bds)

        self.observation_space = gym.spaces.Box(
            low=lower_bds, high=upper_bds, shape=(state_space_dim,), dtype=np.float64
        )
        """
        Action a_k = X(k)
        - X(k)
          C_max rows, for each row p:
            0 <= X(k)_{p,1} <= n_DUs + n_CUs
            0 <= X(k)_{p,2} <= C_max
            domain's size = ((n_DUs + n_CUs+1)(C_max+1))^C_max
          
        """
        self.action_space_dim = ((self.n_DUs + self.n_CUs + 1)*(self.C_max + 1)) ** self.C_max
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

        # Node computational utilization as function of time
        self.DU_utilization = np.full(shape=(self.horizon_length, self.n_DUs, 1), fill_value=0, dtype=np.float64)
        self.CU_utilization = np.full(shape=(self.horizon_length, self.n_CUs, 1), fill_value=0, dtype=np.float64)

        # Node power consumption as function of time
        self.DU_phi = np.full(shape=(self.horizon_length, self.n_DUs, 1), fill_value=0, dtype=np.float64)
        self.CU_phi = np.full(shape=(self.horizon_length, self.n_CUs, 1), fill_value=0, dtype=np.float64)

        # Path-allocation matrix
        self.assigned_paths = np.full(self.C_max*2, fill_value=0, dtype=int)
        # current VNF index
        self.vnf_idx = 1

        self.timestep = 0
        while self.current_scenario[self.timestep] == None:
            self.timestep += 1
        self.current_slice = self.current_scenario[self.timestep].pop()

        return self.get_state(), {}
    
    def get_state(self):
        t_s = self.current_slice[1]

        if self.timestep - t_s > 0:
            time_idx = self.timestep - t_s
        else:
            time_idx = 0
        state = np.concatenate([
            self.DU_av_capacity[self.timestep].flatten(),
            self.CU_av_capacity[self.timestep].flatten(),
            self.assigned_paths,
            np.array([self.current_slice[0]]),                                      # F_s
            np.array([self.vnf_idx]),                                               # i_k
            np.array([self.current_slice[4][self.vnf_idx][time_idx]]),              # C_{i,s}(t_k)
            np.array([self.current_slice[3]]),                                      # delta_s
            np.array([self.current_slice[2]]),                                      # ht_s
            np.array([self.timestep])                                               # t_k
        ])
        
        return state
    
    def get_info(self):
        info = {
            'time': self.timestep,
            'active': self.current_scenario_active_slices
        }
        return info
    
    def decode_action(self, action_idx):
        """
        Decode an action index into the C_max x 2 matrix X(k)
        
        - X
        Column 0: node index (0, ..., n_DUs+n_CUs)
        Column 1: number of replicas (0, ..., C_max)
        """
        X = np.zeros((self.C_max, 2), dtype=int)
        
        rem = action_idx
        for p in range(self.C_max):
            row = rem % ((self.n_DUs + self.n_CUs + 1) * (self.C_max + 1))
            node = row // (self.C_max + 1)
            replicas = row % (self.C_max + 1)
            X[p, 0] = node
            X[p, 1] = replicas
            
            rem = rem // ((self.n_DUs + self.n_CUs + 1) * (self.C_max + 1))
        return X

    def step(self, action):
        current_state =  self.get_state()
        #self.print_state_readable(current_state)
        reward = float('-inf')

        # if action == 0:
        #    if (self.timestep < self.horizon_length - 1) and \
        #       (self.timestep < self.current_slice[1] + self.current_slice[2]):
        #        if self.current_scenario[self.timestep + 1] == None:
        #            self.current_scenario[self.timestep + 1] = [self.current_slice.copy()]
        #        else:
        #            self.current_scenario[self.timestep + 1].append(self.current_slice.copy())
        #else:
        if action != 0:
            reward = 0

            # Decode action f(a_k)
            X_k = self.decode_action(action)
            replicas_per_node = np.zeros((self.n_DUs+self.n_CUs), dtype=int)
            for p in range(self.C_max):
                if X_k[p,0] > 0:
                    replicas_per_node[X_k[p,0] - 1] = X_k[p,1]

            # min/max utilization across all nodes
            max_util = 0; min_util = np.inf

            # for every time step from current time until slice's termination
            for t in range(self.current_slice[1] + self.current_slice[2] - self.timestep):
                for du in range(self.n_DUs):
                    # If t = t_0 or node idle in t-1: cold start overhead may be needed
                    prev_phi = self.DU_phi[self.timestep + t][du][0] if self.timestep + t == 0 else self.DU_phi[self.timestep + t - 1][du][0]
                    cold_start = (prev_phi == 0)

                    # Reduce the available resources according to the demand of each VNF on each node
                    self.DU_av_capacity[self.timestep + t][du][0] -= replicas_per_node[du] * self.CPU_per_instance[self.vnf_idx]
                    # Increase the power consumption of this node accordingly
                    self.DU_phi[self.timestep + t][du][0] += replicas_per_node[du] * self.CPU_per_instance[self.vnf_idx] * self.P_cpu
                    
                    # If cold start was considered and node is active now: apply cold start overhead
                    if self.DU_phi[self.timestep + t][du][0] > 0 and cold_start:
                        self.DU_phi[self.timestep + t][du][0] += self.P_start
                        #print(f"DU {du+1} cold start applied!")

                    # u^i_{node}(t) = 1 - (AC_i(t)/c_i)
                    self.DU_utilization[self.timestep + t][du][0] = 1 - (self.DU_av_capacity[self.timestep + t][du][0] / self.c_DU)
                    max_util = np.maximum(max_util, self.DU_utilization[self.timestep + t][du][0])
                    min_util = np.minimum(min_util, self.DU_utilization[self.timestep + t][du][0])

                # We act accordingly for CUs
                for cu in range(self.n_CUs):
                    prev_phi = self.CU_phi[self.timestep + t][cu][0] if self.timestep + t == 0 else self.CU_phi[self.timestep + t - 1][cu][0]
                    cold_start = (prev_phi == 0)
                    # During computing CU power consumption, we have to take into consideration their distributed workload
                    E_lambda = (self.theta_idle + self.llambda[cu] * (1-self.theta_idle)) * self.rho_lambda[cu]

                    self.CU_av_capacity[self.timestep + t][cu][0] -= replicas_per_node[self.n_DUs + cu] * self.CPU_per_instance[self.vnf_idx]

                    self.CU_phi[self.timestep + t][cu][0] += replicas_per_node[self.n_DUs + cu] * self.CPU_per_instance[self.vnf_idx] * self.P_cpu * E_lambda

                    if self.CU_phi[self.timestep + t][cu][0] > 0 and cold_start:
                        self.CU_phi[self.timestep + t][cu][0] += self.P_start
                        #print(f"CU {cu+1} cold start applied!")
                    
                    self.CU_utilization[self.timestep + t][cu][0] = 1 - (self.CU_av_capacity[self.timestep + t][cu][0] / self.c_CU)
                    max_util = np.maximum(max_util, self.CU_utilization[self.timestep + t][cu][0])
                    min_util = np.minimum(min_util, self.CU_utilization[self.timestep + t][cu][0])

            # Power consumption for each moment in current timestep
            for du in range(self.n_DUs):
                reward -= self.reward_coeffs[0] * self.DU_phi[self.timestep][du][0]
            for cu in range(self.n_CUs):
                reward -= self.reward_coeffs[0] * self.CU_phi[self.timestep][cu][0]

            # Maximum load imbalance
            reward -= self.reward_coeffs[1] * (max_util - min_util)

            # If the last VNF is placed
            if self.vnf_idx == self.current_slice[0]-1:
                # Compute the max E2E delay experienced across all paths using the path matrix P_k
                max_delay = 0
                for p in range(self.C_max):
                    delay = 0
                    assigned_du = self.assigned_paths[2*p]
                    #print(assigned_du)
                    if assigned_du > 0:
                        delay += self.FH_latency[assigned_du-1]
                        assigned_cu = X_k[p,0]-self.n_DUs
                        if assigned_cu > 0:
                            delay += self.BH_latency[assigned_du-1, assigned_cu-1]
                    max_delay = np.maximum(max_delay, delay)
                #print(f"Max delay for slice at time {self.timestep}: {max_delay} ms")
                # If SLA violation occurs, apply proportionate penalty
                if max_delay > self.current_slice[3]:
                    reward -= self.reward_coeffs[2] * (max_delay - self.current_slice[3]) * self.Gamma
                    #print(f"SLA violation! Delay: {max_delay} ms, SLA: {self.current_slice[3]} ms")

        self.scenario_reward += reward
        terminated = False

        # If there are more VNFs in the current slice
        if self.vnf_idx < self.current_slice[0]-1:
            # If VNF f_{1,s} is under consideration
            if self.vnf_idx == 1:
                # Update the DUs in the path allocation matrix
                for p in range(self.C_max):
                    self.assigned_paths[2*p] = X_k[p,0]
            else:
                for p in range(self.C_max):
                    # If a CU is assigned, update the path allocation matrix accordingly
                    if X_k[p,0] > self.n_DUs:
                        self.assigned_paths[2*p+1] = X_k[p,0] - self.n_DUs
            # Move to the next VNF
            self.vnf_idx += 1
        else: 
            while self.current_scenario[self.timestep] == None or self.current_scenario[self.timestep] == []:
                self.timestep += 1
                if self.timestep == self.horizon_length:
                    terminated = True
                    self.score_history.append(self.scenario_reward)
                    self.timestep = 0
                    break

            if not terminated:
                self.current_slice = self.current_scenario[self.timestep].pop()
                self.assigned_paths = np.full(self.C_max*2, fill_value=0, dtype=int)
                self.vnf_idx = 1

        next_state = self.get_state()
        #self.print_state_readable(next_state)

        return next_state, reward, terminated, False, self.get_info()
    
    def is_action_valid(self, action):
        # In this framework, we assume a_k != 0
        if action == 0:
            return False

        # Decode action f(a_k)
        X_k = self.decode_action(action)
        replicas_per_node = np.zeros((self.n_DUs+self.n_CUs), dtype=int)
        for p in range(self.C_max):
            if X_k[p,0] > 0:
                replicas_per_node[X_k[p,0] - 1] = X_k[p,1]

        # If the total computational demand is greater that the available capacity -> Invalid action
        for node in range(self.n_DUs+self.n_CUs):
            if (node+1 > self.n_DUs and replicas_per_node[node] > self.CU_av_capacity[self.timestep][node-self.n_DUs][0]):
                return False
            elif (node+1 <= self.n_DUs and replicas_per_node[node] > self.DU_av_capacity[self.timestep][node][0]):
                return False

        for p in range(self.C_max):
            # For every path with assignment in a shared node, the assigned replicas must be the same
            for p2 in range(p+1, self.C_max):
                if X_k[p,0] == X_k[p2,0] and X_k[p,1] != X_k[p2,1]:
                    return False
                
            # For an assigned node
            if X_k[p,0] > 0:
                # If no replicas are assigned -> Invalid action
                if X_k[p,1] == 0:
                    return False
                # If the corresponding path is inactive and the current VNF is not f_{1,s} -> Invalid action
                if self.assigned_paths[2*p] == 0 and self.vnf_idx > 1:
                    return False
            # For an unassigned node
            else:
                # If there are assigned replicas -> Invalid action
                if X_k[p,1] > 0:
                    return False
                # If the corresponding path is active -> Invalid action
                if self.assigned_paths[2*p] > 0:
                    return False

            du_assigned = self.assigned_paths[2*p]
            cu_assigned = self.assigned_paths[2*p+1]
            # For a path with only DU assigned
            if du_assigned > 0 and cu_assigned == 0:
                # If we assign the VNF to a different DU -> Invalid action
                if not(X_k[p,0] == du_assigned or X_k[p,0] > self.n_DUs):
                    return False
                
                for p2 in range(self.C_max):
                    if p2 == p:
                        continue
                    # If we assign the VNF to an already used CU -> Invalid action
                    if self.assigned_paths[2*p2+1] > 0 and X_k[p,0] == self.assigned_paths[2*p2+1] + self.n_DUs:
                        return False
                    # If it is a shared DU and one replica migrate in CU, while the other remain in DU -> Invalid action
                    if self.assigned_paths[2*p2] == du_assigned and self.assigned_paths[2*p2+1] == cu_assigned:
                        if not ((X_k[p,0] == du_assigned and X_k[p2,0] == du_assigned) or  
                                (X_k[p,0] > self.n_DUs and X_k[p2,0] > self.n_DUs)):
                            return False

            # If, for a path with CU assigned, we assign the VNF to any other node than this CU -> Invalid action
            if cu_assigned > 0 and X_k[p,0] != cu_assigned + self.n_DUs:
                return False
            
            # If we assign VNF f_{1,s} to a CU -> Invalid action
            if self.vnf_idx == 1 and X_k[p,0] > self.n_DUs:
                return False

        # If the minimum requirement of replicas count is not met or
        # if the allocated replicas exceed the maximum number of replicas per VNF
        # -> Invalid action
        required_replicas = self.current_slice[4][self.vnf_idx][self.timestep - self.current_slice[1]]
        deployed_replicas = sum(replicas_per_node)
        if self.C_max < deployed_replicas or required_replicas > deployed_replicas:
            return False
        
        #print(f"\nValid action: {action}")
        #print("X_k =", X_k)

        return True

    def invalid_action_masking(self):
        mask = np.zeros(self.action_space_dim, dtype=bool)
        
        for action in range(self.action_space_dim):
            mask[action] = self.is_action_valid(action)

        return mask

