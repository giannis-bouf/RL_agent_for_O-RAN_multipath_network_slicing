import numpy as np
import pickle

class SliceReqGenerator:
    def __init__(self, Fs_max, C_max):
        """
        Slice arrival generator for simulating slice requests
        Fs_max: Maximum size of Service Function Chain (maximum number of VNF in a slice)
        C_max: Maximum number of replicas per VNF
        """
        self.Fs_max = Fs_max
        self.C_max = C_max
        self.arriving_reqs = []
        self.active = np.zeros(12)
        total = np.random.randint(20, 30)
        self.generate_reqs(S=total, H=12, t_0=0)

    # Choosing slice type based on given distribution
    def slice_type(self, p_embb, p_urllc, p_mmtc):
        types = ["eMBB", "URLLC", "mMTC"]
        return np.random.choice(types, p=[p_embb, p_urllc, p_mmtc])

    def generate_reqs(self, S, H, t_0):
        p_embb = 0.5
        p_urllc = 0.3
        p_mmtc = 0.2

        for _ in range(S):
            type_s = self.slice_type(p_embb, p_urllc, p_mmtc)
            # Slice's E2E delay requirement according to its type
            if type_s == 'URLLC':
                delta_s = 25
            elif type_s == 'eMBB':
                delta_s = 50
            elif type_s == 'mMTC': 
                delta_s = 100

            # # Arriving time of slice request
            # Normal Distribution
            # mu = (H+t_0)/2-1
            # sigma = 0.9
            # t_s = int(np.clip(np.random.normal(loc=mu, scale=sigma), t_0, H + t_0-1))

            # Poisson distribution
            # t_s = np.clip(np.random.poisson(lam=mu/2), t_0, H + t_0)

            # Exponential distribution
            # t_s = int(np.clip(np.random.exponential(scale=(H+t_0)/4), t_0, H + t_0-1))

            # Beta Distribution
            # a, b = 10, 2
            # t_s = int(np.clip(np.random.beta(a, b) * (H + t_0 - 1), t_0, H + t_0 - 1))

            # Uniform Distribution
            t_s = np.random.randint(t_0, H + t_0)

            ht_s = min(np.random.randint(H / 4, H / 2), t_0 + H - t_s) # Slice's holding time
            for t in range(t_s, t_s + ht_s):
                self.active[t] = self.active[t] + 1

            F_s = np.random.randint(3,self.Fs_max + 1) # SFC's size
            C_s = [np.int64(1)] # Computational demand for each VNF (demand for VNF 0 is always 1)
            # The distribution of replica count for the following VNFs 
            # is designed to favor lower demands greater than 1, as an exponential decay 
            # is applied to values from 2 to C_max
            if self.C_max > 1:
                p1 = 1 / (self.C_max + 2)
                remaining_indices = np.arange(self.C_max - 1)
                exp_weights = np.exp(-remaining_indices / 3.0) 
                remaining_p = 1 - p1
                exp_weights_normalized = (exp_weights / exp_weights.sum()) * remaining_p
                replica_p = np.concatenate(([p1], exp_weights_normalized))
            else:
                replica_p = np.array([1.0])
            for _ in range(1, F_s):
                C_s.append(np.random.choice(range(1, self.C_max+1), p=replica_p))

            self.arriving_reqs.append([F_s, t_s, ht_s, delta_s, C_s])

def load():
    with open('datasets/datasetX/historic_data.pkl', 'rb') as f:
        loaded_episodes = pickle.load(f)
        print(f"Loaded {len(loaded_episodes)} episodes.")
        print(loaded_episodes[0].arriving_reqs)  # Print the first episode for verification

if __name__ == '__main__':
    all_episodes, test_episodes = [], []
    num_episodes = 1000
    tests = 1
    Fs_max = 4
    C_max = 2
    np.random.seed(443)

    # Generate 500 episodes
    for _ in range(num_episodes):
        episode = SliceReqGenerator(Fs_max, C_max)
        all_episodes.append(episode)

    # Save episodes to pickle file
    with open('datasets/datasetX/historic_data.pkl', 'wb') as f:
        pickle.dump(all_episodes, f)

    for _ in range(tests):
        episode = SliceReqGenerator(Fs_max, C_max)
        test_episodes.append(episode)

    with open('datasets/datasetX/test_data.pkl', 'wb') as f:
       pickle.dump(test_episodes, f)

    load()

