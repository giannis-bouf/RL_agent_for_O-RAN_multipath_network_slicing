import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from arrival import SliceReqGenerator

with open("datasets/dataset2/historic_data.pkl", "rb") as f:
    dataset = pickle.load(f)

arrivals = np.zeros(12)
arr_time = []
act_time = []
active = np.zeros(12)
configs = np.zeros(12)

for inst in dataset:
    for sl in inst.arriving_reqs:
        arrivals[sl[1]] += 1
        arr_time.append(sl[1]+1)
        for t in range(sl[1], sl[1] + sl[2]):
            act_time.append(t+1)
    active += inst.active
    configs += inst.configurations

plt.figure(figsize=(6, 4))
sns.histplot(x=arr_time, label='Dataset', kde=False, stat='count', bins=30, weights=np.ones_like(arr_time) / 1000, color='blue')

#plt.title('Empirical Distributions')
plt.xlabel('Time Slot', fontweight='bold')
plt.xticks(np.arange(1, 13, 1), fontweight='bold')
plt.ylabel('Average Number of Arrivals', fontweight='bold')
plt.yticks(fontweight='bold')
plt.legend(fontsize='medium', frameon=True).set_title(None)  # Remove legend title
for text in plt.gca().get_legend().get_texts():
    text.set_fontweight('bold')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(x=act_time, label='Dataset', kde=False, stat='count', bins=30, weights=np.ones_like(act_time) / 1000, color='blue')

#plt.title('Empirical Distributions')
plt.xlabel('Time Slot', fontweight='bold')
plt.xticks(np.arange(1, 13, 1), fontweight='bold')
plt.ylabel('Average Number of Active Slices', fontweight='bold')
plt.yticks(fontweight='bold')
plt.legend(fontsize='medium', frameon=True).set_title(None)  # Remove legend title
for text in plt.gca().get_legend().get_texts():
    text.set_fontweight('bold')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.bar(np.arange(1, 13), configs / 1000)

plt.xlabel('Time Slot', fontweight='bold')
plt.xticks(np.arange(1, 13, 1), fontweight='bold')
plt.ylabel('Average Number of Configurations', fontweight='bold')
plt.yticks(fontweight='bold')
plt.grid(True)
plt.tight_layout()
plt.show()
