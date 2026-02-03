import numpy as np

def generate_data(n=2000):
    X = np.random.uniform(-1, 1, (n, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    return X, y
