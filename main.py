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

C_max = 2 
n_DUs = 2 
n_CUs = 2 
Fs_max = 4

def decode_path_matrix(encoded_p):
    """
    Encoding used: sum_{i=1...C_max} (P(k){i,0}*(n_CUs+1) + P(k){i,1}) * ((n_DUs+1)(n_CUs+1))^(i-1)
    Decode encoded path matrix into P(k) with shape C_max*2
    P[i][0] = DU index in [0,n_DUs]
    P[i][1] = CU index in [0,n_CUs]
    """
    base = (n_DUs+1) * (n_CUs+1)
    P = [[0,0] for _ in range(C_max)]

    temp = encoded_p
    for i in range(C_max):
        var = temp % base
        temp //= base

        P[i][0] = var // (n_CUs + 1)
        P[i][1] = var %  (n_CUs + 1)

    return P

def decode_allocation_matrix(self, encoded_x, N):
    """
    Encoding used: sum_{i=1...n__} sum_{j=1...Fs_max-1} X_s(k)_{i,j} * (C_max+1)^{(i-1)(Fs_max-1)+(j-1)}
    Decoder for allocation matrices X^{DU}(k) and X^{CU}(k) with shape n_DUs*(Fs_max-1) and n_CUs*(Fs_max-1), respectively
    X{i,j} in [0,C_max] which denotes how many replicas of VNF j+1 does node i hosts
    """
    X = [[0]*(Fs_max-1) for _ in range(N)]
    temp = encoded_x

    for i in range(N):
        for j in range(Fs_max-1):
            X[i][j] = temp % (C_max + 1)
            temp //= (C_max + 1)

    return X

def decode_allocation_DU(self, encoded_du):
    return self.decode_allocation_matrix(encoded_du, n_DUs)

def decode_allocation_CU(self, encoded_cu):
    return self.decode_allocation_matrix(encoded_cu, n_CUs)

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

if training:
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = MaskablePPO("MlpPolicy", env, gamma=0.99, verbose=2, tensorboard_log="./ppo_tensorboard/", seed=12)

    print("Learning is about to begin!")
    model.learn(total_timesteps=1_000_000, progress_bar=True)
    model.save(f"results/train{dataset}/agent{agent}/agent.zip")
    env.save(f"results/train{dataset}/agent{agent}/vec_normalize.pkl")

    wrapped_env = env.envs[0]
    while hasattr(wrapped_env, 'env'):
        wrapped_env = wrapped_env.env

    with open(f"results/train{dataset}/agent{agent}/score_history.pkl", "wb") as f:
        pickle.dump(wrapped_env.score_history, f)
else:
    env = VecNormalize.load(f"results/train{dataset}/agent{agent}/vec_normalize.pkl", env)
    env.norm_reward = False

    model = MaskablePPO.load(f"results/train{dataset}/agent{agent}/agent.zip")
    test_episodes = 1

    test_score_history = []
    state = env.reset()
    for episode in range(test_episodes):
        done = False
        score = 0
        print(f"Start of episode {episode+1}")
        power_cons = np.zeros(12, dtype=np.float32)
        cpu_du = np.zeros(shape=(n_DUs, 12), dtype=np.float32)
        cpu_cu = np.zeros(shape=(n_CUs, 12), dtype=np.float32)
        cr_cpu = np.zeros(12, dtype=np.float32)
        accepted = np.zeros(12)
        active = np.zeros(12)
        acc_ratio = np.zeros(12, dtype=np.float32)
        power_eff = np.zeros(12, dtype=np.float32)

        # Weights in f(a_k)-action encoding function per block
        cu_w = (C_max + 1)**(n_DUs*(Fs_max-1))               # "Allocation in CUs - X^{CU}" block
        p_w  = cu_w * ((C_max + 1)**(n_CUs*(Fs_max-1)))      # "Path" block
        sp_w = p_w * (((n_DUs + 1)*(n_CUs + 1)) ** C_max)    # "Split point" block

        while not done:
            action_mask = env.get_attr("action_masks")[0]()
            action, _ = model.predict(state, action_masks=action_mask, deterministic=True)

            unnormalized_state = env.unnormalize_obs(state).flatten()
            print("State length:", len(unnormalized_state))
            for i, v in enumerate(unnormalized_state):
                print(i, v)

            """if action != 0:
                # Finding encoding for each block - Factorization of action f(a_k)
                #encoded_sp = action // self.sp_w; rem = action % self.sp_w
                encoded_p  = action // p_w;     rem = action % p_w
                encoded_cu = rem // cu_w
                encoded_du = rem % cu_w

                # Decode each number in order to find the respective matrix
                #SP_k   = self.decode_split_matrix(encoded_sp)
                P_k    = decode_path_matrix(encoded_p)
                X_DU_k = decode_allocation_DU(encoded_du)
                X_CU_k = decode_allocation_CU(encoded_cu)"""