import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_results(dataset, agent_id):
    # Load all case_*.csv files from an agent and return averaged metrics

    dir = f"results/train{dataset}/agent{agent_id}"
    dfs = []
    for test in range(1,4):
        for file in os.listdir(dir + f"/test{test}"):
            df = pd.read_csv(os.path.join(dir + f"/test{test}", file))
            dfs.append(df)

    if len(dfs) == 0:
        raise ValueError(f"No case files found in {dir}")
    
    Df = pd.concat(dfs, axis=0)
    
    return  Df.groupby("Time").mean()

agentA = 3
agentB = 5
dataset = 2

A = load_results(dataset, agentA)
B = load_results(dataset, agentB)

agent_orientations = {
    1: "Power Consumption-oriented",
    2: "Load Balance-oriented",
    3: "Delay Violations-oriented",
    4: "Load Balance and Delay-oriented",
    5: "Power Consumption and Delay-oriented",
    6: "Multi-oriented"
}

time = np.arange(12)

# Plot Power Consumption
plt.figure(figsize=(10, 5))
plt.plot(time, A["Power_Consumption"], label=f"{agent_orientations[agentA]} Agent")
plt.plot(time, B["Power_Consumption"], label=f"{agent_orientations[agentB]} Agent")
plt.xlabel("Time")
plt.ylabel("Power Consumption")
plt.title("Average Power Consumption Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Load Imbalance
plt.figure(figsize=(10, 5))
plt.plot(time, A["Max_Load_Imbalance"], label=f"{agent_orientations[agentA]} Agent")
plt.plot(time, B["Max_Load_Imbalance"], label=f"{agent_orientations[agentB]} Agent")
plt.xlabel("Time")
plt.ylabel("Load Imbalance")
plt.title("Average Load Imbalance Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot SLA Violations
plt.figure(figsize=(10, 5))
plt.plot(time, A["SLA_Violations"], label=f"{agent_orientations[agentA]} Agent")
plt.plot(time, B["SLA_Violations"], label=f"{agent_orientations[agentB]} Agent")
plt.xlabel("Time")
plt.ylabel("SLA Violations")
plt.title("Average SLA Violations Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot SLA Severity
plt.figure(figsize=(10, 5))
plt.plot(time, A["SLA_Severity"], label=f"{agent_orientations[agentA]} Agent")
plt.plot(time, B["SLA_Severity"], label=f"{agent_orientations[agentB]} Agent")
plt.xlabel("Time")
plt.ylabel("SLA Severity (ms)")
plt.title("Average SLA Severity Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

