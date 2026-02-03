# Reliability Smoothing and Thresholding: Control Logic & Data Flow Summary

## Overview
This document summarizes where and how reliability smoothing and thresholding are applied in the codebase, and outlines the control logic and data flow for these mechanisms.

---

## 1. Locations of Reliability Smoothing & Thresholding

### A. `src/main.py`
- **Smoothing:**
  - `ExponentialSmoother` is instantiated for reliability and latency.
  - `smoothed_reliability = reliability_smoother.update(reliability)`
- **Thresholding:**
  - `DualSignalController` is initialized with `reliability_threshold` and `latency_threshold` from config.
  - Controller logic uses smoothed reliability and latency to decide model switching.

### B. `src/metrics/smoothing.py`
- **Smoothing Implementation:**
  - `ExponentialSmoother` class implements EWMA (exponential weighted moving average).

### C. `src/controller/dual_controller.py`
- **Thresholding Logic:**
  - `DualSignalController` uses thresholds to switch between 'fast' and 'robust' models based on smoothed reliability/latency.

### D. `src/metrics/reliability.py`
- **Reliability Calculation:**
  - `ReliabilityMetric` computes reliability as an exponential function of output variance.

### E. `self_stabilizing_inference/main.py`
- **Smoothing:**
  - Manual EWMA smoothing of reliability in main loop.
- **Thresholding:**
  - Hardcoded `LOW_THRESHOLD` and `HIGH_THRESHOLD` for switching logic.

### F. `self_stabilizing_inference/detection/degradation.py`
- **Combined Smoothing & Thresholding:**
  - `DegradationDetector` class applies EWMA smoothing and checks if smoothed value is below threshold.

---

## 2. Control Logic & Data Flow (High-Level)

1. **Data Generation:**
   - Input data is generated or sampled.
2. **Model Selection:**
   - Active model is either 'fast' or 'robust', determined by controller logic.
3. **Reliability Calculation:**
   - For each step, reliability is computed using model predictions under noise.
4. **Smoothing:**
   - Raw reliability is smoothed using EWMA (ExponentialSmoother or manual formula).
5. **Thresholding & Control:**
   - Smoothed reliability (and latency) are compared to thresholds.
   - If reliability drops below threshold (or latency exceeds threshold), controller switches to robust model.
   - If reliability recovers above threshold (and latency is low), controller switches back to fast model.
   - Hysteresis and dwell time are used to prevent rapid switching.
6. **Logging & Visualization:**
   - Reliability, smoothed reliability, latency, and model state are logged and visualized.

---

## 3. Key Classes/Functions
- `ExponentialSmoother` (src/metrics/smoothing.py): Implements EWMA.
- `ReliabilityMetric` (src/metrics/reliability.py): Computes reliability.
- `DualSignalController` (src/controller/dual_controller.py): Applies thresholding logic for model switching.
- `DegradationDetector` (self_stabilizing_inference/detection/degradation.py): Combines smoothing and thresholding.

---

## 4. Data Flow Diagram (Textual)

```
[Input Data] → [Model] → [ReliabilityMetric] → [ExponentialSmoother] → [Controller/Threshold] → [Model Switch]
```

---

*This summary is for documentation and review purposes. No code functionality has been changed.*
