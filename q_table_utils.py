import pickle
import collections
import numpy as np

def save_q_table(q_table, path):
    # Save Q-table as a dictionary
    with open(path, "wb") as f:
        pickle.dump(dict(q_table), f)

def load_q_table(path, action_size):

    with open(path, "rb") as f:
        loaded = pickle.load(f)

    return collections.defaultdict(lambda: np.zeros(action_size), loaded)