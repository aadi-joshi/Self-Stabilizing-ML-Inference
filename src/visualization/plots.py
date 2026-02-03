import matplotlib.pyplot as plt
import os

def plot_all(df, outdir):
    # Reliability plot
    plt.figure(figsize=(12, 5))
    plt.plot(df['reliability'], alpha=0.3, label='Raw Reliability')
    plt.plot(df['smoothed_reliability'], linewidth=2, label='Smoothed Reliability')
    plt.xlabel('Step')
    plt.ylabel('Reliability')
    plt.title('Raw vs Smoothed Reliability')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'reliability.png'))
    plt.close()

    # Latency plot
    plt.figure(figsize=(12, 5))
    plt.plot(df['latency'], alpha=0.3, label='Raw Latency')
    plt.plot(df['smoothed_latency'], linewidth=2, label='Smoothed Latency')
    plt.xlabel('Step')
    plt.ylabel('Latency (s)')
    plt.title('Raw vs Smoothed Latency')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'latency.png'))
    plt.close()

    # Active model plot
    plt.figure(figsize=(12, 3))
    plt.plot(df['active_model'], label='Active Model (fast=fast, robust=robust)')
    plt.xlabel('Step')
    plt.ylabel('Model')
    plt.title('Active Model Over Time')
    plt.yticks([0, 1], ['fast', 'robust'])
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'active_model.png'))
    plt.close()

    # Controller state plot
    plt.figure(figsize=(12, 3))
    plt.plot(df['controller_state'], label='Controller State')
    plt.xlabel('Step')
    plt.ylabel('State')
    plt.title('Controller State Over Time')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'controller_state.png'))
    plt.close()
