import numpy as np

def compute_reliability(model, x, noise_std=0.02, trials=10):
    preds = []
    for _ in range(trials):
        noisy_x = x + np.random.normal(0, noise_std, size=x.shape)
        p = model.predict(noisy_x.reshape(1, -1), verbose=0)
        preds.append(p)

    preds = np.array(preds)
    variance = np.var(preds, axis=0).mean()

    reliability = np.exp(-variance * 50.0)
    return float(np.clip(reliability, 0.0, 1.0))
