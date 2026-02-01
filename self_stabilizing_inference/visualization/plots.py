import matplotlib.pyplot as plt

def plot_reliability(df):
    plt.figure()
    plt.plot(df["reliability"], label="Reliability")
    plt.plot(df["degraded"].astype(int), label="Degraded", alpha=0.5)
    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Inference Reliability Over Time")
    plt.show()
