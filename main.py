import argparse
from sb3_contrib import MaskablePPO
from environment import NetworkEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib.common.wrappers import ActionMasker
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
from arrival import SliceReqGenerator
import pandas as pd
import os

C_max = 2; n_DUs = 8; n_CUs = 2; Fs_max = 4; du_capacity = 16; cu_capacity = 64
P_cpu=3; P_start=16; theta_idle=0.4

def decode_action(action_idx):
        """
        Decode an action index into the C_max x 2 matrix X(k)
        
        - X
        Column 0: node index (0, ..., n_DUs+n_CUs)
        Column 1: number of replicas (0, ..., C_max)
        """
        X = np.zeros((C_max, 2), dtype=int)
        
        rem = action_idx
        for p in range(C_max):
            row = rem % ((n_DUs + n_CUs + 1) * (C_max + 1))
            node = row // (C_max + 1)
            replicas = row % (C_max + 1)
            X[p, 0] = node
            X[p, 1] = replicas
            
            rem = rem // ((n_DUs + n_CUs + 1) * (C_max + 1))
        return X

def unwrap_env(wrapped_env):
    env = wrapped_env
    if hasattr(env, "envs") and len(env.envs) > 0:
        env = env.envs[0]

    while hasattr(env, "env"):
        env = env.env

    return env

parser = argparse.ArgumentParser(description="CLI")

parser.add_argument("--training", action="store_true", help="Enable training mode")
parser.add_argument("--dataset",  type=int, choices=range(1,4), default=1)
parser.add_argument("--agent",  type=int, choices=range(1,8), default=1)
parser.add_argument("--test",  type=int, choices=range(1,4), default=1)

args = parser.parse_args()

training = args.training
dataset = args.dataset
agent = args.agent

if training:
    test = dataset
else:
    test = args.test

def mask_function(env):
    return env.invalid_action_masking()

env = NetworkEnv(training_mode=training, dataset=dataset)
env = ActionMasker(env, mask_function)
env = DummyVecEnv([lambda: env])

save_dir = f"results/train{dataset}/agent{agent}/"
os.makedirs(save_dir, exist_ok=True)

if training:
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = MaskablePPO("MlpPolicy", env, gamma=0.99, verbose=2, tensorboard_log=None, seed=12)

    model.learn(total_timesteps=100_000, progress_bar=True)
    model.save(save_dir + "agent.zip")
    env.save(save_dir + "vec_normalize.pkl")

    wrapped_env = env.envs[0]
    while hasattr(wrapped_env, 'env'):
        wrapped_env = wrapped_env.env

    with open(save_dir + "score_history.pkl", "wb") as f:
        pickle.dump(wrapped_env.score_history, f)
else:
    env = VecNormalize.load(save_dir + "vec_normalize.pkl", env)
    env.norm_reward = True

    model = MaskablePPO.load(save_dir + "agent.zip")
    test_episodes = 1

    # Unwrap environment to access NetworkEnv attributes
    base_env = unwrap_env(env.envs[0])
    llambda = np.array(base_env.llambda, dtype=float)                                   # Load lambda
    rho_lambda = np.array(base_env.rho_lambda, dtype=float)                             # Load workload distribution           
    E_lambda = ((theta_idle + llambda * (1-theta_idle)) * rho_lambda).reshape(-1, 1)    # Expected workload factor
    FH_latency = np.array(base_env.FH_latency, dtype=int)                               # Fronthaul links latency                           
    BH_latency = np.array(base_env.BH_latency, dtype=int)                               # Backhaul links latency

    test_score_history = []
    state = env.reset()

    for episode in range(test_episodes):
        done = False
        score = 0
        print(f"Start of episode {episode+1}")
        power_cons = np.zeros(12, dtype=np.float32)         # Power consumption over time
        sla_violations = np.zeros(12, dtype=int)            # SLA violations over time
        sla_severity = np.zeros(12, dtype=int)              # Extra delay of SLA violations over time
        cpu_du = np.zeros(shape=(n_DUs, 12), dtype=int)     # CPU usage per DU over time
        cpu_cu = np.zeros(shape=(n_CUs, 12), dtype=int)     # CPU usage per CU over time
        active = np.zeros(12, dtype=int)                    # Active slices over time

        while not done:
            action_mask = env.get_attr("action_masks")[0]()
            action, _ = model.predict(state, action_masks=action_mask, deterministic=True)

            unnormalized_state = env.unnormalize_obs(state).flatten()

            # If a VNF is being deployed
            if action != 0:
                X_k = decode_action(action)
                # Assign replicas to nodes
                replicas_per_node = np.zeros(n_DUs + n_CUs, dtype=int)
                for p in range(C_max):
                    if X_k[p,0] > 0:
                        replicas_per_node[X_k[p,0]-1] = X_k[p,1]

                # For the slice's lifetime, update active slices and CPU usage            
                for t in range(round(unnormalized_state[-1]), round(unnormalized_state[-1]) + round(unnormalized_state[-2])):
                    active[t] += (round(unnormalized_state[15]) == 1)
                    
                    for node in range(n_DUs + n_CUs):
                        if node < n_DUs:
                            cpu_du[node, t] += replicas_per_node[node]
                        else:
                            cpu_cu[node-n_DUs, t] += replicas_per_node[node]
                    
                # If the last VNF is being placed, check for SLA violations
                if round(unnormalized_state[-5]) == round(unnormalized_state[-6])-1:
                    paths = np.round(unnormalized_state[n_DUs+n_CUs : n_DUs+n_CUs + 2*C_max]).astype(int)
                    # Compute the max E2E delay experienced across all paths using the path matrix
                    max_delay = 0
                    for p in range(C_max):
                        delay = 0
                        assigned_du = paths[2*p]
                        if assigned_du > 0:
                            delay += FH_latency[assigned_du-1]
                            assigned_cu = X_k[p,0]-n_DUs
                            if assigned_cu > 0:
                                delay += BH_latency[assigned_du-1, assigned_cu-1]
                        max_delay = np.maximum(max_delay, delay)
                    # If SLA violation occurs, record it across the slice's lifetime
                    if max_delay > round(unnormalized_state[-3]):
                        print(f"SLA violation! Delay: {max_delay} ms, SLA: {round(unnormalized_state[-3])} ms")
                        for t in range(round(unnormalized_state[-1]), round(unnormalized_state[-1]) + round(unnormalized_state[-2])):
                            sla_violations[t] += 1
                            sla_severity[t] += max_delay - round(unnormalized_state[-3])

            state_, reward, done, info = env.step(action)
            if isinstance(reward, list):
                reward = reward
            score += reward     # Accumulate reward
            state = state_      # Move to next state

        # Compute node utilizations
        util_du = cpu_du / du_capacity
        util_cu = cpu_cu / cu_capacity
        # Merge node utilizations
        node_util = np.vstack([util_du, util_cu])

        # Compute max-min load over time
        max_load = np.max(node_util, axis=0)
        min_load = np.min(node_util, axis=0)

        # Compute power consumption
        power_cons = P_cpu * np.sum(util_du * du_capacity, axis=0) + P_cpu * np.sum(util_cu * cu_capacity * E_lambda, axis=0)
        # Add cold start power overhead
        for t in range(12):
            if t == 0:
                for n in range(n_DUs + n_CUs):
                    if node_util[n, t] > 0:
                        power_cons[t] += P_start
            else:
                for n in range(n_DUs + n_CUs):
                    if node_util[n, t] > 0 and node_util[n, t-1] == 0:
                        power_cons[t] += P_start


        data = {
                "Time": np.arange(12),
                "Active": active,
                "Power_Consumption": power_cons,
                "Max_Load_Imbalance": max_load - min_load,
                "SLA_Violations": sla_violations,
                "SLA_Severity": sla_severity / np.maximum(sla_violations, 1)
        }

        df = pd.DataFrame(data)
        os.makedirs(save_dir + f"test{test}/", exist_ok=True)
        df.to_csv(save_dir + f"test{test}/case_{episode}.csv")
        score = score.item()
        test_score_history.append(score)

        print(f'Test episode {episode}, score {score:.1f}')

        time = np.arange(12)
        # Plot DU utilizations
        plt.figure(figsize=(10,5))
        for d in range(n_DUs):
            plt.plot(time, util_du[d], label=f"DU {d+1}")
        plt.xlabel("Time")
        plt.ylabel("Utilization")
        plt.title("DU Utilization Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot CUs utilizations
        plt.figure(figsize=(10,5))
        for c in range(n_CUs):
            plt.plot(time, util_cu[c], label=f"CU {c+1}")
        plt.xlabel("Time")
        plt.ylabel("Utilization")
        plt.title("CU Utilization Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    avg_test_score = np.mean(test_score_history)
    print(f'Average test score over {test_episodes} episodes: {avg_test_score:.1f}')
