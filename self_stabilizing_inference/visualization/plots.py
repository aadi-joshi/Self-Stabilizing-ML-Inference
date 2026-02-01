import matplotlib.pyplot as plt

def plot_reliability(df):
    plt.figure(figsize=(10, 4))
    plt.plot(df["reliability"], label="Reliability")
    plt.plot(df["degraded"].astype(int), label="Degraded", alpha=0.4)
    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Self-Stabilizing Inference Reliability")
    plt.tight_layout()
    plt.savefig("reliability.png")
    plt.show()
