import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

score_history = []
for i in range(1, 4):
    with open(f"results/train{i}/agentX/score_history.pkl", "rb") as f:
        sh = pickle.load(f)
        print(len(sh))
        score_history.append(moving_average(sh, window_size=1000))

max_list = []
min_list = []
main_list = []

for i in range(626):
    min_list.append(min(score_history[0][i], score_history[1][i], score_history[2][i]))
    max_list.append(max(score_history[0][i], score_history[1][i], score_history[2][i]))
    main_list.append((score_history[0][i] + score_history[1][i] + score_history[2][i]) / 3)

main_array = np.array(main_list)
min_array = np.array(min_list)
max_array = np.array(max_list)
x = np.arange(len(main_array))

plt.figure(figsize=(6, 4))

# Plot main curve
plt.plot(x, main_array, color='blue',linewidth=2)

# Fill area between min and max
plt.fill_between(x, min_array, max_array, color='blue', alpha=0.2)

# Labels and styling
plt.xlabel('Episode', fontsize=12, fontweight='bold')
plt.ylabel('Cumulative Reward', fontsize=12, fontweight='bold')
plt.xticks(np.arange(7) * 100, fontweight='bold')
plt.yticks(fontweight='bold')
plt.grid(True)
plt.tight_layout()
plt.show()
