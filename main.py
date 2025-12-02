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

C_max = 2; n_DUs = 8; n_CUs = 2; Fs_max = 4
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

parser = argparse.ArgumentParser(description="CLI")

parser.add_argument("--training", action="store_true", help="Enable training mode")
parser.add_argument("--dataset",  type=int, choices=range(1,4), default=1)
parser.add_argument("--agent",  type=int, choices=range(1,6), default=1)
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

env = NetworkEnv(training_mode=training, dataset=test)
env = ActionMasker(env, mask_function)
env = DummyVecEnv([lambda: env])

save_dir = f"results/train{dataset}/agent{agent}/"
os.makedirs(save_dir, exist_ok=True)

if training:
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    #model = MaskablePPO("MlpPolicy", env, gamma=0.99, verbose=2, tensorboard_log="./ppo_tensorboard/", seed=12)
    model = MaskablePPO("MlpPolicy", env, gamma=0.99, verbose=2, tensorboard_log=None, seed=12)

    print("Learning is about to begin!")
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
    env.norm_reward = False

    model = MaskablePPO.load(save_dir + "agent.zip")
    test_episodes = 1

    test_score_history = []
    state = env.reset()
    for episode in range(test_episodes):
        done = False
        score = 0
        print(f"Start of episode {episode+1}")
        power_cons = np.zeros(12, dtype=np.float32)
        util_imbalance = np.zeros(12, dtype=np.float32)
        sla_violations = np.zeros(12, dtype=np.float32)
        cpu_du = np.zeros(shape=(n_DUs, 12), dtype=np.float32)
        cpu_cu = np.zeros(shape=(n_CUs, 12), dtype=np.float32)
        cr_cpu = np.zeros(12, dtype=np.float32)
        # accepted = np.zeros(12)
        active = np.zeros(12)
        # acc_ratio = np.zeros(12, dtype=np.float32)

        while not done:
            action_mask = env.get_attr("action_masks")[0]()
            action, _ = model.predict(state, action_masks=action_mask, deterministic=True)

            unnormalized_state = env.unnormalize_obs(state).flatten()
            print("State length:", len(unnormalized_state))
            for i, v in enumerate(unnormalized_state):
                print(i, v)

            if action != 0:
                X_k = decode_action(action)
                replicas_per_node = np.zeros(n_DUs + n_CUs, dtype=int)
                for p in range(C_max):
                    if X_k[p,0] > 0:
                        replicas_per_node[X_k[p,0]-1] = X_k[p,1]
            
                for t in range(round(unnormalized_state[-1]), round(unnormalized_state[-1]) + round(unnormalized_state[18])):
                    active[t] += (round(unnormalized_state[15]) == 1)
                    
                    for p in range(C_max):
                        node = X_k[p,0]
                        if node > 0:
                            if node <= n_DUs:
                                cpu_du[node-1, t] += replicas_per_node[node-1]
                            else:
                                cpu_cu[node - n_DUs -1, t] += replicas_per_node[node-1]
                            cr_cpu[t] += replicas_per_node[node-1]

            state_, reward, done, info = env.step(action)
            if isinstance(reward, list):
                reward = reward
            score += reward
            state = state_

        data = {
                "Time": np.arange(12),
                "Active": active
        }

        df = pd.DataFrame(data)
        os.makedirs(save_dir + f"test{test}/", exist_ok=True)
        df.to_csv(save_dir + f"test{test}/case_{episode}.csv")
        score = score.item()
        test_score_history.append(score)

        print(f'Test episode {episode}, score {score:.1f}')

    avg_test_score = np.mean(test_score_history)
    print(f'Average test score over {test_episodes} episodes: {avg_test_score:.1f}')
