"""
quad_ekf.py
===========
Extended Kalman Filter (EKF) for the quadcopter defined in quad_pid_utils.py.

This file contains:
  - jacobian_F()     : analytical 12x12 process Jacobian  dF/dX
  - EKF class        : predict + multi-sensor update steps
  - SensorSimulator  : adds realistic noise to the true simulation state

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THEORETICAL BACKGROUND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY EKF AND NOT STANDARD KALMAN FILTER?
────────────────────────────────────────
The standard (linear) Kalman Filter assumes:
    x_{k+1} = A * x_k + B * u_k + w_k       (linear dynamics)
    z_k     = H * x_k + v_k                  (linear measurement)

The quadcopter dynamics are NONLINEAR — the thrust appears multiplied by
sin(phi)*cos(theta), rotation matrices involve products of trig functions,
and the Euler kinematics couple phi, theta, and the rates p, q, r in a
nonlinear way. A linear Kalman Filter cannot handle this.

The EKF solves this by LINEARIZING the nonlinear dynamics around the
current state estimate at each timestep using the Jacobian matrix.

THE FULL EKF EQUATIONS
──────────────────────
Let:
    x  ∈ R^12  — state vector  [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
    f(x, u)    — nonlinear dynamics function (dynamics() from quad_pid_utils)
    h(x)       — nonlinear measurement function
    F          — Jacobian of f w.r.t. x  (12x12)
    H          — Jacobian of h w.r.t. x  (meas_dim x 12)
    P          — state covariance matrix  (12x12)
    Q          — process noise covariance (12x12)
    R          — measurement noise covariance (meas_dim x meas_dim)
    K          — Kalman gain
    I          — 12x12 identity matrix

PREDICT STEP  (propagates state and uncertainty forward in time)
────────────────────────────────────────────────────────────────
    x_pred = x + dt * f(x, u)              ← Euler integration of nonlinear dynamics
    F      = ∂f/∂x  evaluated at x         ← Jacobian linearization at current state
    P_pred = F * P * F^T + Q               ← covariance propagation (linear approx)

    Interpretation of P_pred = F*P*F^T + Q:
      - F*P*F^T  : how the current uncertainty (P) is transformed by the dynamics
      - Q        : new uncertainty added at each step due to process noise
                   (unmodeled aerodynamics, wind, vibration, etc.)

UPDATE STEP  (corrects the prediction using a sensor measurement)
─────────────────────────────────────────────────────────────────
    y  = z - h(x_pred)                     ← innovation: difference between
                                              actual measurement z and predicted
                                              measurement h(x_pred)
    S  = H * P_pred * H^T + R              ← innovation covariance
    K  = P_pred * H^T * S^{-1}             ← Kalman gain: how much to trust
                                              the measurement vs the prediction
    x  = x_pred + K * y                    ← corrected state estimate
    P  = (I - K * H) * P_pred              ← corrected covariance (Joseph form
                                              used in code for numerical stability)

    Interpretation of K:
      - Large K → trust the measurement more  (P large or R small)
      - Small K → trust the prediction more   (P small or R large)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE JACOBIAN — WHAT IT IS AND HOW IT IS COMPUTED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT IS A JACOBIAN?
───────────────────
A Jacobian is a matrix of ALL first-order partial derivatives of a vector
function. If f : R^n → R^n maps an n-dimensional input to an n-dimensional
output, the Jacobian F is:

         ┌ ∂f₁/∂x₁   ∂f₁/∂x₂  ···  ∂f₁/∂xₙ ┐
         │ ∂f₂/∂x₁   ∂f₂/∂x₂  ···  ∂f₂/∂xₙ │
    F =  │    ⋮                          ⋮  │
         └ ∂fₙ/∂x₁   ∂fₙ/∂x₂  ···  ∂fₙ/∂xₙ ┘

Each row i, column j entry answers: "how does output fᵢ change when I
nudge input xⱼ by a tiny amount, holding everything else constant?"

WHY WE NEED IT IN THE EKF
──────────────────────────
The covariance propagation equation P_pred = F*P*F^T + Q comes from a
first-order Taylor expansion of the nonlinear dynamics around the current
estimate x:

    f(x + δx) ≈ f(x) + F * δx      where F = ∂f/∂x

This approximation is valid when δx is small — i.e., when the estimate
is already reasonably close to the true state. This is the fundamental
assumption of the EKF.

HOW THE JACOBIAN IS COMPUTED HERE
──────────────────────────────────
We differentiate the dynamics equations from quad_pid_utils.py analytically
and hardcode the result. This is the standard approach in
embedded GNC systems — it runs in microseconds vs numerical differentiation.

Our state vector and the index mapping:
    Index : State variable
    ──────────────────────
      0   : x      (inertial position X)
      1   : y      (inertial position Y)
      2   : z      (inertial position Z)
      3   : vx     (inertial velocity X)
      4   : vy     (inertial velocity Y)
      5   : vz     (inertial velocity Z)
      6   : phi    (roll angle)
      7   : theta  (pitch angle)
      8   : psi    (yaw angle)
      9   : p      (roll rate  — body frame)
     10   : q      (pitch rate — body frame)
     11   : r      (yaw rate   — body frame)

The full 12×12 Jacobian has many zero entries because most states don't
directly affect most derivatives. The non-trivial non-zero blocks are:

  Block 1: ∂[dx,dy,dz]/∂[vx,vy,vz]  →  identity (positions integrate velocities)
  Block 2: ∂[dvx,dvy,dvz]/∂[phi,theta,psi]  →  thrust projection trig derivatives
  Block 3: ∂[dvx,dvy,dvz]/∂[vx,vy,vz]  →  diagonal drag terms  -Ax/m, -Ay/m, -Az/m
  Block 4: ∂[dphi,dtheta,dpsi]/∂[phi,theta]  →  Euler kinematics coupling
  Block 5: ∂[dphi,dtheta,dpsi]/∂[p,q,r]  →  Euler kinematics rates
  Block 6: ∂[dp,dq,dr]/∂[p,q,r]  →  inertia coupling + drag  (Euler's equations)

All other entries are zero and are not stored (the matrix is initialized to zeros).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SENSOR MODELS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Four simulated sensors, each measuring a subset of the state:

  GPS          : measures [x, y, z]               → H_gps   is 3×12  (linear)
  IMU (gyro)   : measures [p, q, r]               → H_gyro  is 3×12  (linear)
  Magnetometer : measures [psi]                   → H_mag   is 1×12  (linear)
  Accelerometer: measures [ax_body, ay_body]      → nonlinear h(x), linearized H

The first three measurement functions h(x) are LINEAR — they simply pick out
specific states — so their Jacobians H are constant matrices (ones in the
right positions, zeros elsewhere). No linearization needed for H.

The ACCELEROMETER is different: it measures the projection of gravity onto
the body frame, which is a nonlinear function of phi and theta:

    ax_body ≈  g * sin(theta)
    ay_body ≈ -g * sin(phi) * cos(theta)

Its Jacobian H_acc(x) must be recomputed at each update step (it depends
on the current state estimate). This is a true EKF update — the only one
in this implementation that requires a state-dependent H.

WHY THE ACCELEROMETER IS SO VALUABLE:
The GPS, gyro, and magnetometer together leave roll (phi) and pitch (theta)
unobservable except through the weak coupling in the translational dynamics.
The accelerometer directly observes the gravity vector orientation, giving
phi and theta a strong, high-rate (200 Hz) measurement — exactly what is
needed to suppress the attitude noise visible without it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
from collections import deque


# ============================================================
#  JACOBIAN  F = ∂f/∂x   (12 × 12)
# ============================================================

def jacobian_F(x, T, Omega, params):
    """
    Analytical Jacobian of the quadcopter dynamics f(x) with respect to
    the state vector x.

    This is the matrix F = ∂f/∂x, evaluated at the current state estimate x.
    It is used in the EKF predict step to propagate the covariance:
        P_pred = F * P * F^T + Q

    Parameters
    ----------
    x      : np.ndarray (12,)
        Current state estimate [x,y,z, vx,vy,vz, phi,theta,psi, p,q,r]
    T      : float
        Total thrust force [N] at the current timestep
    Omega  : float
        Net rotor speed w1-w2+w3-w4 [rad/s] for gyroscopic coupling
    params : dict
        Quadcopter physical parameters

    Returns
    -------
    F : np.ndarray (12, 12)
        Jacobian matrix evaluated at state x
    """

    # ── Unpack state ────────────────────────────────────────────────────────
    _, _, _, vx, vy, vz, phi, theta, psi, p, q, r = x

    # ── Unpack physical parameters ──────────────────────────────────────────
    m   = params['m']
    Ixx = params['Ixx']
    Iyy = params['Iyy']
    Izz = params['Izz']
    Ax  = params['Ax']
    Ay  = params['Ay']
    Az  = params['Az']
    Ar  = params['Ar']
    IR  = params['IR']

    # ── Pre-compute trig functions ──────────────────────────────────────────
    sp = np.sin(phi);    cp = np.cos(phi)
    st = np.sin(theta);  ct = np.cos(theta)
    ss = np.sin(psi);    cs = np.cos(psi)

    # Guard against Euler-angle singularity at theta = ±90°.
    # The Jacobian has 1/cos(theta) and tan(theta) terms (Blocks 4 & 5) that
    # diverge at ±90°.  We clamp |ct| to a small positive floor so the filter
    # degrades gracefully rather than producing NaN/Inf.  For nominal flight
    # (|theta| << 90°) this clamp is never active.
    if abs(ct) < 1e-3:
        ct = np.sign(ct) * 1e-3
    tt = st / ct   # tan(theta) — computed after clamping ct

    # ── Initialize Jacobian to zero ─────────────────────────────────────────
    F = np.zeros((12, 12))

    # ════════════════════════════════════════════════════════════════════════
    #  BLOCK 1:  ∂[dx, dy, dz] / ∂[vx, vy, vz]  →  identity
    # ════════════════════════════════════════════════════════════════════════
    F[0, 3] = 1.0
    F[1, 4] = 1.0
    F[2, 5] = 1.0

    # ════════════════════════════════════════════════════════════════════════
    #  BLOCK 2:  ∂[dvx, dvy, dvz] / ∂[vx, vy, vz]  →  drag terms
    # ════════════════════════════════════════════════════════════════════════
    F[3, 3] = -Ax / m
    F[4, 4] = -Ay / m
    F[5, 5] = -Az / m

    # ════════════════════════════════════════════════════════════════════════
    #  BLOCK 3:  ∂[dvx, dvy, dvz] / ∂[phi, theta, psi]  →  thrust projection
    #
    #  dvx/dt = (sin(phi)*sin(psi) + cos(phi)*sin(theta)*cos(psi)) * T/m - Ax/m*vx
    #  dvy/dt = (-sin(phi)*cos(psi) + cos(phi)*sin(theta)*sin(psi)) * T/m - Ay/m*vy
    #  dvz/dt = -g + cos(phi)*cos(theta) * T/m - Az/m*vz
    # ════════════════════════════════════════════════════════════════════════
    Tm = T / m

    F[3, 6] = ( cp*ss - sp*st*cs) * Tm    # ∂dvx/∂phi
    F[3, 7] = ( cp*ct*cs        ) * Tm    # ∂dvx/∂theta
    F[3, 8] = ( sp*cs - cp*st*ss) * Tm    # ∂dvx/∂psi

    F[4, 6] = (-cp*cs - sp*st*ss) * Tm    # ∂dvy/∂phi
    F[4, 7] = ( cp*ct*ss        ) * Tm    # ∂dvy/∂theta
    F[4, 8] = ( sp*ss + cp*st*cs) * Tm    # ∂dvy/∂psi

    F[5, 6] = (-sp*ct           ) * Tm    # ∂dvz/∂phi
    F[5, 7] = (-cp*st           ) * Tm    # ∂dvz/∂theta

    # ════════════════════════════════════════════════════════════════════════
    #  BLOCK 4:  ∂[dphi, dtheta, dpsi] / ∂[phi, theta]  →  Euler kinematics
    #
    #  dphi/dt   = p + sin(phi)*tan(theta)*q + cos(phi)*tan(theta)*r
    #  dtheta/dt = cos(phi)*q - sin(phi)*r
    #  dpsi/dt   = sin(phi)/cos(theta)*q + cos(phi)/cos(theta)*r
    # ════════════════════════════════════════════════════════════════════════
    sec2_theta = 1.0 / (ct**2)

    F[6, 6] = (cp*tt*q - sp*tt*r)
    F[6, 7] = (sp*q + cp*r) * sec2_theta

    F[7, 6] = (-sp*q - cp*r)

    F[8, 6] = ( cp/ct*q - sp/ct*r)
    F[8, 7] = (sp*q + cp*r)*tt/ct

    # ════════════════════════════════════════════════════════════════════════
    #  BLOCK 5:  ∂[dphi, dtheta, dpsi] / ∂[p, q, r]  →  Euler kinematics rates
    # ════════════════════════════════════════════════════════════════════════
    F[6, 9]  = 1.0
    F[6, 10] = sp * tt
    F[6, 11] = cp * tt

    F[7, 10] = cp
    F[7, 11] = -sp

    F[8, 10] = sp / ct
    F[8, 11] = cp / ct

    # ════════════════════════════════════════════════════════════════════════
    #  BLOCK 6:  ∂[dp, dq, dr] / ∂[p, q, r]  →  Euler's equations
    #
    #  dp/dt = ((Iyy-Izz)/Ixx)*q*r - (IR/Ixx)*q*Omega + M_phi/Ixx - (Ar/Ixx)*p
    #  dq/dt = ((Izz-Ixx)/Iyy)*p*r + (IR/Iyy)*p*Omega + M_theta/Iyy - (Ar/Iyy)*q
    #  dr/dt = ((Ixx-Iyy)/Izz)*p*q + M_psi/Izz - (Ar/Izz)*r
    # ════════════════════════════════════════════════════════════════════════
    c_pqr  = (Iyy - Izz) / Ixx
    c_pqr2 = (Izz - Ixx) / Iyy
    c_pqr3 = (Ixx - Iyy) / Izz

    F[9,  9]  = -Ar / Ixx
    F[9,  10] = c_pqr * r - (IR / Ixx) * Omega   # IR gyroscopic term
    F[9,  11] = c_pqr * q

    F[10, 9]  = c_pqr2 * r + (IR / Iyy) * Omega  # IR gyroscopic term
    F[10, 10] = -Ar / Iyy
    F[10, 11] = c_pqr2 * p

    F[11, 9]  = c_pqr3 * q
    F[11, 10] = c_pqr3 * p
    F[11, 11] = -Ar / Izz

    return F


# ============================================================
#  MEASUREMENT MATRICES H  (constant — linear measurement functions)
# ============================================================

H_GPS = np.zeros((3, 12))
H_GPS[0, 0] = 1.0    # x
H_GPS[1, 1] = 1.0    # y
H_GPS[2, 2] = 1.0    # z

H_GYRO = np.zeros((3, 12))
H_GYRO[0, 9]  = 1.0  # p
H_GYRO[1, 10] = 1.0  # q
H_GYRO[2, 11] = 1.0  # r

H_MAG = np.zeros((1, 12))
H_MAG[0, 8] = 1.0    # psi


# ============================================================
#  SENSOR SIMULATOR
# ============================================================

class SensorSimulator:
    """
    Simulates realistic noisy sensor measurements from the true state.

    Sensors with update rates:
      GPS          — 10  Hz  — measures position [x, y, z]
      IMU (gyro)   — 200 Hz  — measures body rates [p, q, r]
      Magnetometer — 50  Hz  — measures yaw [psi]
      Accelerometer— 200 Hz  — measures specific force [ax_body, ay_body]

    Noise standard deviations (1-sigma):
      GPS  :  σ_pos = 0.5   m
      Gyro :  σ_pqr = 0.01  rad/s
      Mag  :  σ_psi = 0.05  rad
      Acc  :  σ_acc = 0.05  m/s²
    """

    def __init__(self, noise_params=None):
        p = noise_params or {}
        self.gps_std  = p.get('gps_std',  0.5 )
        self.gyro_std = p.get('gyro_std', 0.01)
        self.mag_std  = p.get('mag_std',  0.05)
        self.acc_std  = p.get('acc_std',  0.05)

    def gps(self, x_true):
        return x_true[0:3] + np.random.randn(3) * self.gps_std

    def gyro(self, x_true):
        return x_true[9:12] + np.random.randn(3) * self.gyro_std

    def magnetometer(self, x_true):
        return x_true[8:9] + np.random.randn(1) * self.mag_std

    def accelerometer(self, x_true, x_true_prev=None, dt=0.005, g=9.81):
        """
        Simulate body-frame accelerometer measurement.

        A real accelerometer measures SPECIFIC FORCE — the sum of the gravity
        projection AND the vehicle's inertial (translational) acceleration:

            a_measured = R_body_inertial^T * (a_inertial - g_vec)

        During static or near-hover flight, inertial acceleration ≈ 0 and the
        measurement reduces to the gravity projection alone.  But on a figure-8
        trajectory with 4 m amplitude, lateral inertial accelerations reach
        ~0.3–0.5 m/s², which is LARGER than the gravity signal at small tilt
        angles (~g*sin(5°) ≈ 0.85 m/s²).

        We approximate the inertial component by finite-differencing the true
        inertial velocities and rotating to the body frame.  This is the
        physically correct simulator model and is what a real IMU sees.

        The EKF measurement model h(x) still uses ONLY the gravity projection
        (it doesn't know the inertial acceleration).  The mismatch between what
        the sensor sees and what h(x) predicts appears as effective noise, which
        is why R_acc must be set large enough to account for it (σ ≈ 0.5 m/s²
        rather than the sensor's electronic noise of ~0.05 m/s²).
        """
        phi   = x_true[6]
        theta = x_true[7]
        psi   = x_true[8]
        sp = np.sin(phi);  cp = np.cos(phi)
        st = np.sin(theta); ct = np.cos(theta)
        ss = np.sin(psi);  cs = np.cos(psi)

        # Gravity projection in body frame (what h(x) models)
        ax_grav =  g * st
        ay_grav = -g * sp * ct

        # Inertial acceleration in body frame (what h(x) does NOT model)
        # Rotate inertial accel into body frame via R^T (3-2-1 Euler)
        ax_inertial = ay_inertial = 0.0
        if x_true_prev is not None:
            dvx = (x_true[3] - x_true_prev[3]) / dt
            dvy = (x_true[4] - x_true_prev[4]) / dt
            # Project inertial XY acceleration onto body X and Y axes
            # (body-frame rows of the rotation matrix R)
            ax_inertial = (ct*cs)*dvx + (ct*ss)*dvy
            ay_inertial = (sp*st*cs - cp*ss)*dvx + (sp*st*ss + cp*cs)*dvy

        ax_true = ax_grav + ax_inertial
        ay_true = ay_grav + ay_inertial
        return np.array([ax_true, ay_true]) + np.random.randn(2) * self.acc_std


# ============================================================
#  EXTENDED KALMAN FILTER
# ============================================================

class EKF:
    """
    Extended Kalman Filter for the 12-state quadcopter.

    Sensor update schedule:
      - IMU   (gyro)        : 200 Hz — every predict step
      - Accelerometer       : 200 Hz — every predict step
      - Magnetometer        : 50  Hz — every 4th predict step
      - GPS                 : 10  Hz — every 20th predict step

    State vector (12 states):
        [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
         0  1  2   3   4   5   6     7     8   9  10  11
    """

    def __init__(self, x0, P0, Q, R_gps, R_gyro, R_mag, R_acc, quad_params):
        self.x      = x0.copy()
        self.P      = P0.copy()
        self.Q      = Q.copy()
        self.R_gps  = R_gps.copy()
        self.R_gyro = R_gyro.copy()
        self.R_mag  = R_mag.copy()
        self.R_acc  = R_acc.copy()
        self.params = quad_params
        self.I12    = np.eye(12)

        # ── Adaptive R (GPS) state ───────────────────────────────────────
        # Scale R_gps based on observed innovation spread.
        # When GPS degrades, innovations grow — inflate R so the filter
        # down-weights bad measurements instead of forcing them onto state.
        # Reference: Mehra (1970) "On the identification of variances and
        # adaptive Kalman filtering", IEEE TAC.
        self._R_gps_base      = R_gps.copy()
        self._r_scale         = 1.0
        self._r_scale_pending = 1.0     # applied at START of next GPS update
        self._innov_window    = deque(maxlen=10)  # auto-evicts oldest entry
        self._innov_maxlen    = 10      # 2 s at 10 Hz

        # ── Adaptive Q_vel state ─────────────────────────────────────────
        # Scale velocity process noise when controller commands high accel.
        self._Q_vel_base  = np.diag(Q[3:6, 3:6]).copy()
        self._vel_q_scale = 1.0

    def predict(self, f_dynamics, dt, quad_obj, rotor_speeds):
        """
        EKF PREDICT STEP
        ─────────────────
        Equations:
            x_pred = x + dt * f(x, u)          <- Euler integration
            F_d    = I + dt * F_cont + dt2/2 * F_cont^2  <- 2nd-order discrete
            P_pred = F_d * P * F_d^T + Q_adaptive
        """
        # 1. Euler integration
        x_dot  = f_dynamics(0, self.x, quad_obj, rotor_speeds)
        self.x = self.x + dt * x_dot

        # wrap psi to [-pi, pi] after every integration step.
        # Without this psi drifts unboundedly, growing mag innovations until
        # they exceed the gate and yaw corrections silently stop.
        self.x[8] = np.arctan2(np.sin(self.x[8]), np.cos(self.x[8]))

        # 2. Adaptive Q_vel: scale velocity process noise with accel magnitude.
        # Threshold 0.4 m/s^2 (figure-8 peak ~0.16 m/s^2 with margin).
        # Max scale 2x — enough to open P during turns without destabilising.
        accel_mag = np.linalg.norm(x_dot[3:6])
        if accel_mag > 0.4:
            self._vel_q_scale = min(2.0, 1.0 + (accel_mag - 0.4) * 1.5)
        else:
            self._vel_q_scale = 1.0
        self.Q[3, 3] = self._Q_vel_base[0] * self._vel_q_scale
        self.Q[4, 4] = self._Q_vel_base[1] * self._vel_q_scale
        self.Q[5, 5] = self._Q_vel_base[2] * self._vel_q_scale

        # 3. Total thrust and net rotor speed for Jacobian
        kT    = self.params['kT']
        T     = kT * np.sum(rotor_speeds**2)
        Omega = rotor_speeds[0] - rotor_speeds[1] + rotor_speeds[2] - rotor_speeds[3]

        # 4. Analytical Jacobian at current state (includes IR gyroscopic terms)
        F_cont = jacobian_F(self.x, T, Omega, self.params)

        # 5. Discretise via 2nd-order Taylor expansion of matrix exponential:
        #        F_d ≈ I + dt*F_cont + (dt^2/2)*F_cont^2
        #    Reduces discretization error O(dt^2) -> O(dt^3) for one extra
        #    matrix multiply — worth it for the coupled rotational states.
        F_d = self.I12 + dt * F_cont + (0.5 * dt * dt) * (F_cont @ F_cont)

        # 6. Covariance propagation
        self.P = F_d @ self.P @ F_d.T + self.Q

    def _update(self, z, H, R, gate_threshold=None):
        """
        Generic EKF update (shared by all sensors).

        Uses the Joseph form for numerical stability:
            P = (I - KH) P (I - KH)^T + K R K^T

        Innovation gating (Mahalanobis distance test):
        ───────────────────────────────────────────────
        Before applying the update we compute the normalised innovation
        squared (NIS):

            NIS = y^T S^{-1} y

        where y = z - H*x is the innovation and S = H*P*H^T + R is the
        innovation covariance.  NIS follows a chi-squared distribution
        with m degrees of freedom (m = measurement dimension) when the
        filter is consistent.

        If NIS > gate_threshold we reject the measurement entirely —
        it is statistically inconsistent with the current state estimate
        and is most likely an outlier (GPS multipath, spike, etc.).

        Typical gate thresholds (chi2 95th percentile):
            m=1 (mag)   →  3.84
            m=2 (acc)   →  5.99
            m=3 (GPS, gyro) → 7.81

        Parameters
        ----------
        gate_threshold : float or None
            NIS threshold.  None disables gating (default, backwards-compatible).
        """
        y   = z - H @ self.x
        S   = H @ self.P @ H.T + R

        # ── Innovation gate ───────────────────────────────────────────────
        if gate_threshold is not None:
            try:
                nis = float(y @ np.linalg.solve(S, y))
                if nis > gate_threshold:
                    return   # reject outlier measurement — do not update
            except np.linalg.LinAlgError:
                return       # singular S — skip update safely

        # K = P H^T S^{-1}  →  solve S K^T = H P^T for K^T, then transpose
        # np.linalg.solve(A, B) solves A @ X = B  →  X = S^{-1} (H P)
        K   = np.linalg.solve(S, H @ self.P).T    # (12 x m)
        self.x = self.x + K @ y
        IKH    = self.I12 - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T               # Joseph form

    # ── Chi-squared innovation gate thresholds ───────────────────────────────
    # Degrees of freedom = measurement dimension m.
    # Gyro/acc/mag use the 95th percentile: chi2.ppf(0.95, df=m) gives
    # m=1 -> 3.84, m=2 -> 5.99, m=3 -> 7.81.
    # The GPS gate is deliberately looser, at the 99.9th percentile
    # (chi2.ppf(0.999, df=3) = 16.27): under sustained GPS degradation,
    # adaptive R should down-weight the measurements smoothly rather than
    # the gate rejecting them outright and leaving position dead-reckoned.
    _GATE_GPS  = 16.27   # 3-DOF  (x, y, z position)
    _GATE_GYRO = 7.81   # 3-DOF  (p, q, r rates)
    _GATE_MAG  = 3.84   # 1-DOF  (yaw)
    _GATE_ACC  = 5.99   # 2-DOF  (ax_body, ay_body)

    def update_gps(self, z_gps):
        """
        GPS position update with innovation gating and adaptive R.

        Correct two-phase design:
          Phase A (always runs): collect innovation, compute new R scale,
                                 store as pending for NEXT call.
          Phase B (gate check):  apply pending R from PREVIOUS call,
                                 gate with consistent S, update if passes.

        This separation ensures S used for gating and S used for the Kalman
        update are computed with the SAME R value, eliminating the
        inconsistency in the previous implementation where R was updated
        mid-step and the gate used stale S.

        Adaptive R parameters:
          alpha     = 0.20  — EMA smoothing (was 0.50, too reactive)
          max_scale = 10    — R_max = 10 x R_base (was 100, caused GPS lockout)
          window    = 20    — 2 s of GPS history at 10 Hz
        """
        y = z_gps - H_GPS @ self.x

        # ── PHASE A: always collect innovation, compute pending R scale ───
        # This runs unconditionally — even if the gate later rejects this
        # measurement, we still want to observe the innovation for R estimation.
        self._innov_window.append(y.copy())
       

        if len(self._innov_window) == self._innov_maxlen:
            innov_arr  = np.array(self._innov_window)
            C_innov    = (innov_arr.T @ innov_arr) / len(innov_arr)
            # Use current R_gps
            S_expected = H_GPS @ self.P @ H_GPS.T + self.R_gps
            ratio      = np.trace(C_innov) / np.trace(S_expected)

            alpha     = 0.30                         
            new_scale = max(1.0, min(30.0, ratio))   
            self._r_scale_pending = ((1 - alpha) * self._r_scale
                                     + alpha * new_scale)

        # ── PHASE B: apply pending scale from PREVIOUS step, then gate ───
        # R_gps is updated to the scale computed last call, so both the gate
        # S and the Kalman update S are consistent with the same R.
        self._r_scale = self._r_scale_pending
        self.R_gps    = self._R_gps_base * self._r_scale

        S = H_GPS @ self.P @ H_GPS.T + self.R_gps

        try:
            nis = float(y @ np.linalg.solve(S, y))
            if nis > self._GATE_GPS:
                return   # reject outlier — R was already updated for next step
        except np.linalg.LinAlgError:
            return

        # ── Apply update ──────────────────────────────────────────────────
        K      = np.linalg.solve(S, H_GPS @ self.P).T
        self.x = self.x + K @ y
        IKH    = self.I12 - K @ H_GPS
        self.P = IKH @ self.P @ IKH.T + K @ self.R_gps @ K.T   # Joseph form

    def update_gyro(self, z_gyro):
        self._update(z_gyro, H_GYRO, self.R_gyro, gate_threshold=self._GATE_GYRO)

    def update_magnetometer(self, z_mag):
        """
        Magnetometer (yaw) update with correct circular-statistics innovation.

        The wrapped innovation is fed directly into the gain equations so that
        z_mag never touches the state vector — eliminating the correlation bias
        that arises when z_wrapped is constructed as x[8] + innov.
        """
        innov = np.arctan2(
            np.sin(z_mag[0] - self.x[8]),
            np.cos(z_mag[0] - self.x[8])
        )                                                    # scalar ∈ [-π, π]
        S      = H_MAG @ self.P @ H_MAG.T + self.R_mag      # (1,1)

        # ── Innovation gate (1-DOF chi2 95% = 3.84) ──────────────────────
        nis = float(innov**2 / S[0, 0])
        if nis > self._GATE_MAG:
            return

        K      = np.linalg.solve(S, H_MAG @ self.P).T       # (12,1)
        self.x = self.x + K[:, 0] * innov
        IKH    = self.I12 - K @ H_MAG
        self.P = IKH @ self.P @ IKH.T + K @ self.R_mag @ K.T   # Joseph form

    def update_accelerometer(self, z_acc, g=9.81):
        """
        Accelerometer update — the only nonlinear measurement in this filter.

        MEASUREMENT MODEL
        ─────────────────
        A body-frame accelerometer at near-hover measures the reaction to
        gravity (specific force).  For small-to-moderate angles:

            h(x) = [ g * sin(theta)           ]   ← ax_body
                   [-g * sin(phi) * cos(theta) ]   ← ay_body

        LINEARISED H  (state-dependent — recomputed every call)
        ────────────────────────────────────────────────────────
        H_acc = ∂h/∂x  evaluated at current estimate:

            ∂ax/∂theta =  g * cos(theta)
            ∂ay/∂phi   = -g * cos(phi) * cos(theta)
            ∂ay/∂theta =  g * sin(phi) * sin(theta)

        All other partial derivatives are zero (ax and ay don't depend on
        position, velocity, yaw, or body rates in this model).

        WHY THIS HELPS
        ──────────────
        Without the accelerometer, phi and theta are only observable through
        the weak coupling between attitude and translational acceleration in
        the GPS/velocity states.  At 10 Hz GPS and small tilt angles that
        signal is buried in noise.

        The accelerometer observes the gravity vector directly at 200 Hz,
        giving the filter a strong, frequent correction for both roll and
        pitch simultaneously.

        Parameters
        ----------
        z_acc : np.ndarray (2,)
            Accelerometer measurement [ax_body, ay_body] in m/s²
        g     : float
            Gravitational acceleration (default 9.81 m/s²)
        """
        phi   = self.x[6]
        theta = self.x[7]

        sp = np.sin(phi);  cp = np.cos(phi)
        st = np.sin(theta); ct = np.cos(theta)

        # Predicted measurement h(x)
        h = np.array([
             g * st,
            -g * sp * ct
        ])

        # Innovation
        y = z_acc - h

        # State-dependent H matrix (2 × 12) — only phi (6) and theta (7) columns
        H_acc = np.zeros((2, 12))
        H_acc[0, 7] =  g * ct          # ∂ax/∂theta
        H_acc[1, 6] = -g * cp * ct     # ∂ay/∂phi
        H_acc[1, 7] =  g * sp * st     # ∂ay/∂theta

        # Standard EKF update with Joseph-form covariance and innovation gate
        S      = H_acc @ self.P @ H_acc.T + self.R_acc    # (2,2)

        # ── Innovation gate (2-DOF chi2 95% = 5.99) ──────────────────────
        try:
            nis = float(y @ np.linalg.solve(S, y))
            if nis > self._GATE_ACC:
                return
        except np.linalg.LinAlgError:
            return

        K      = np.linalg.solve(S, H_acc @ self.P).T     # (12,2)
        self.x = self.x + K @ y
        IKH    = self.I12 - K @ H_acc
        self.P = IKH @ self.P @ IKH.T + K @ self.R_acc @ K.T   # Joseph form
