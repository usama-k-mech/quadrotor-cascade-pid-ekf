# quadcopter-cascade-PID-EKF-closed-loop-GNC-simulation
A closed-loop Guidance, Navigation, and Control (GNC) simulation of a nonlinear quadcopter with cascade PID control and EKF-based state estimation.

---

## What This Demonstrates
- **Cascade PID**: outer position (50 Hz) → attitude (100 Hz) → rate (200 Hz)
- **12-state EKF**: analytical Jacobian, Joseph-form covariance, innovation gating
- **Closed-loop**: controller feeds off EKF estimates, not true state
- **NEES validation**: nominal NEES = 12.72 vs ideal = 12.0
- **Sensor fusion**: GPS (10 Hz) · Gyro (200 Hz) · Magnetometer (50 Hz) · Accelerometer (200 Hz)
- **Stress test**: adaptive GPS R handles 5× noise degradation gracefully

---

## Results
| Metric | Nominal | Stress GPS×5 |
|--------|---------|--------------|
| Position RMSE X | 0.187 m | 0.985 m |
| Position RMSE Y | 0.161 m | 0.709 m |
| Position RMSE Z | 0.191 m | 1.213 m |
| NEES            | 12.72   | 36.16   |

---

## Visualizations

![EKF State Estimation](src/figures/fig1_ekf_state_estimation.png)

![Stress Test (5× GPS Noise)](src/figures/fig2_stress_test.png)

![NEES Consistency](src/figures/fig3_nees.png)

![Innovation Sequence](src/figures/fig4_innovations.png)

![Covariance](src/figures/fig5_covariance.png)

---

## Files
- `quad_pid_utils.py` — dynamics, cascade PID, trajectory
- `quad_ekf.py` — EKF, Jacobian, sensor simulator
- `quad_ekf_run.py` — closed-loop simulation runner

---

## Dependencies
```bash
pip install numpy scipy matplotlib c4dynamics
```
