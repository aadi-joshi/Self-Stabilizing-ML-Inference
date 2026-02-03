import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# =========================================================
# 1. DATA GENERATION
# =========================================================

def generate_data(n=2000):
    X = np.random.uniform(-1, 1, (n, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    return X, y

X_train, y_train = generate_data()

# =========================================================
# 2. MODEL DEFINITIONS
# =========================================================

def build_fast_model():
    model = Sequential([
        Dense(8, activation="relu", input_shape=(2,)),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss="binary_crossentropy"
    )
    return model

def build_robust_model():
    model = Sequential([
        Dense(32, activation="relu", input_shape=(2,)),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy"
    )
    return model

fast_model = build_fast_model()
robust_model = build_robust_model()

fast_model.fit(X_train, y_train, epochs=5, verbose=0)
robust_model.fit(X_train, y_train, epochs=10, verbose=0)

# =========================================================
# 3. ENVIRONMENT DEGRADATION MODEL
# =========================================================

def environment_noise(step):
    if step < 150:
        return 0.01      # healthy
    elif step < 300:
        return 0.15      # degraded
    else:
        return 0.03      # recovery

# =========================================================
# 4. RELIABILITY METRIC
# =========================================================

def compute_reliability(model, x, noise_std, trials=12):
    preds = []
    for _ in range(trials):
        noisy_x = x + np.random.normal(0, noise_std, size=x.shape)
        p = model.predict(noisy_x.reshape(1, -1), verbose=0)
        preds.append(p[0][0])

    preds = np.array(preds)
    variance = np.var(preds)

    reliability = np.exp(-variance * 40.0)
    return float(np.clip(reliability, 0.0, 1.0))

# =========================================================
# 5. CONTROLLER PARAMETERS
# =========================================================

LOW_THRESHOLD = 0.40
HIGH_THRESHOLD = 0.70

EMA_ALPHA = 0.1
MIN_DWELL_STEPS = 15

active_model = "fast"
last_switch_step = -100
smoothed_reliability = 1.0

# =========================================================
# 6. MAIN CONTROL LOOP
# =========================================================

steps = 500

reliability_log = []
smoothed_log = []
model_log = []

for step in range(steps):
    x = np.random.uniform(-1, 1, size=(2,))
    noise = environment_noise(step)

    if active_model == "fast":
        reliability = compute_reliability(fast_model, x, noise)
    else:
        reliability = compute_reliability(robust_model, x, noise)

    # Exponential smoothing
    smoothed_reliability = (
        EMA_ALPHA * reliability
        + (1 - EMA_ALPHA) * smoothed_reliability
    )

    # Controller with hysteresis + dwell time
    if (
        active_model == "fast"
        and smoothed_reliability < LOW_THRESHOLD
        and step - last_switch_step > MIN_DWELL_STEPS
    ):
        active_model = "robust"
        last_switch_step = step
        print(f"[STEP {step}] Degradation detected → switching to ROBUST model")

    elif (
        active_model == "robust"
        and smoothed_reliability > HIGH_THRESHOLD
        and step - last_switch_step > MIN_DWELL_STEPS
    ):
        active_model = "fast"
        last_switch_step = step
        print(f"[STEP {step}] Stability recovered → switching to FAST model")

    reliability_log.append(reliability)
    smoothed_log.append(smoothed_reliability)
    model_log.append(0 if active_model == "fast" else 1)

    if step % 25 == 0:
        print(
            f"[STEP {step}] Model={active_model} | "
            f"Reliability={reliability:.3f} | "
            f"Smoothed={smoothed_reliability:.3f}"
        )

# =========================================================
# 7. VISUALIZATION
# =========================================================

plt.figure(figsize=(11, 5))

plt.plot(reliability_log, alpha=0.3, label="Raw Reliability")
plt.plot(smoothed_log, linewidth=2, label="Smoothed Reliability")
plt.plot(model_log, linestyle="--", label="Active Model (0=Fast, 1=Robust)")

plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title("Self-Stabilizing ML Inference with Reliability Smoothing")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Experiment finished.")
