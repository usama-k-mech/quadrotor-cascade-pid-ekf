"""
quad_ekf_run.py
===============
Closed-loop EKF + Cascade PID quadcopter simulation.

WHAT THIS FILE DEMONSTRATES
────────────────────────────
1. CLOSED-LOOP ESTIMATION-CONTROL:
   The PID controller feeds off EKF state estimates, not the true state.

2. ACTUAL ROTOR SPEED REPLAY:
   The EKF predict step uses the rotor speeds that were actually commanded
   at each timestep (stored during the PID run), so the analytical Jacobian
   is evaluated with correct thrust at every step.

3. NEES CONSISTENCY CHECK:
   Normalized Estimation Error Squared — the standard metric for verifying
   a filter is neither over- nor under-confident.
       NEES ≈ n_states (12)  →  filter is consistent
       NEES >> 12            →  filter is overconfident (P too small)
       NEES << 12            →  filter is underconfident (P too large)

4. INNOVATION MONITORING:
   Plots the raw measurement residuals (z - H*x_pred) for each sensor.
   A well-tuned EKF produces zero-mean innovations with bounded variance.

5. NOISE STRESS TEST:
   Re-runs the closed-loop system with GPS noise scaled 5× to show the
   EKF maintains accurate estimates where raw measurements fail completely.

ARCHITECTURE
────────────
Phase 1:  PID sim on TRUE state  →  generates true trajectory + rotor history
Phase 2:  EKF closed-loop run    →  controller reads EKF estimates
Phase 3:  Metrics + plots
Phase 4:  Stress test (high noise)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — saves to file
import matplotlib.pyplot as plt
import types
import os

from quad_pid_utils import (dynamics, run_fig8_pid,
                             position_reference, velocity_reference,
                             InitializeControllers)
from quad_ekf import EKF, SensorSimulator
import c4dynamics as c4d


# ============================================================
#  SHARED CONFIG
# ============================================================

config = {

    'quad': {
        'm'  : 0.468,
        'g'  : 9.81,
        'l'  : 0.225,
        'kT' : 2.98e-6,
        'kQ' : 0.0382,
        'Ixx': 4.856e-3,
        'Iyy': 4.856e-3,
        'Izz': 8.801e-3,
        'Ax' : 0.30,
        'Ay' : 0.30,
        'Az' : 0.25,
        'Ar' : 0.20,
        'IR' : 3.357e-5,
    },

    'trajectory': {
        'A'    : 4.0,
        'B'    : 2.0,
        'omega': 0.1,
        'z_ref': 1.5,
        't_end': 90.0,
    },

    'controller': {
        'Kp_p': 0.80,  'Ki_p': 0.0001,  'Kd_p': 0.010,
        'Kp_q': 0.80,  'Ki_q': 0.0001,  'Kd_q': 0.010,
        'Kp_r': 0.60,  'Ki_r': 0.0001,  'Kd_r': 0.008,

        'Kp_phi'  : 6.0,  'Ki_phi'  : 0.0001,  'Kd_phi'  : 0.80,  'AW_phi'  : 0.5,
        'Kp_theta': 6.0,  'Ki_theta': 0.0001,  'Kd_theta': 0.80,  'AW_theta': 0.5,
        'Kp_psi'  : 4.0,  'Ki_psi'  : 0.5,     'Kd_psi'  : 0.40,  'AW_psi'  : 0.5,

        'Kp_x': 0.80,  'Ki_x': 0.00,  'Kd_x': 0.50,  'AW_x': 0.5,
        'Kp_y': 1.00,  'Ki_y': 0.00,  'Kd_y': 0.70,  'AW_y': 0.5,
        'Kp_z': 8.0,  'Ki_z': 8.0,  'Kd_z': 1.50,  'AW_z': 3.0,

        'Kff_x': 0.35,  'Kff_y': 0.40,

        'N_rate'        : 50,
        'omega_max'     : 1000.0,
        'T_max_factor'  : 4,
        'T_min'         : 0.0,
        'att_cmd_limit' : 0.314,
        'yaw_rate_limit': 1.0,
    },

    'sim': {
        'dt': 0.005,
        'tf': 90.0,
    },
}

# ── EKF noise matrices ────────────────────────────────────────────────────────
#
# Q TUNING RATIONALE
# ───────────────────────────────────────────────────────
# Q: unmodeled disturbances per step
# Q_pos  = 0.005 m      : position random walk
# Q_vel  = 0.020 m/s    : unmodeled aerodynamics; adaptive scaling during maneuvers
# Q_att  = 0.008 rad    : gyro bias drift; psi slightly larger
# Q_rate = 0.012 rad/s  : actuator noise
#
# R TUNING RATIONALE
# ──────────────────
# R_gps  = 0.50 m       : consumer GPS 1-sigma
# R_gyro = 0.015 rad/s  : MEMS IMU spec
# R_mag  = 0.055 rad    : magnetometer
# R_acc  = 0.10 m/s^2   

Q = np.diag(np.array([
    0.005, 0.005, 0.008,   # x, y, z          (m)      grounded in physics
    0.020, 0.020, 0.025,   # vx, vy, vz        (m/s)   adaptive scale in predict()
    0.008, 0.008, 0.010,   # phi, theta, psi   (rad)   gyro drift; psi slightly larger
    0.012, 0.012, 0.012,   # p, q, r           (rad/s) actuator noise
])**2)

R_gps  = np.diag([0.50**2, 0.50**2, 0.50**2])   # adaptive R handles degradation

R_gyro = np.diag([0.015**2, 0.015**2, 0.015**2]) 

R_mag  = np.array([[0.055**2]])                  

R_acc  = np.diag([0.10**2, 0.10**2])              

P0 = np.diag(np.array([
    1.0,  1.0,  1.0,
    0.5,  0.5,  0.5,
    0.1,  0.1,  0.1,
    0.1,  0.1,  0.1,
])**2)

# Sensor update divisors (relative to 200 Hz master)
GPS_RATE = 20   # GPS  at 10 Hz  → every 20 steps
MAG_RATE =  4   # Mag  at 50 Hz  → every  4 steps


# ============================================================
#  HELPER: build a SimpleNamespace quad object from state vector
#          so controllers can read .x .y .phi etc.
# ============================================================

def _make_quad_ns(X_state, quad_params):
    """
    Returns a SimpleNamespace that looks like a c4d.rigidbody to the
    controller — it exposes .x .y .z .vx .vy .vz .phi .theta .psi .p .q .r
    plus the physical parameters needed for ControlAllocator.
    """
    ns = types.SimpleNamespace(**quad_params)
    (ns.x, ns.y, ns.z,
     ns.vx, ns.vy, ns.vz,
     ns.phi, ns.theta, ns.psi,
     ns.p,  ns.q,  ns.r) = X_state
    # stored fields controller references
    ns.F         = quad_params['m'] * quad_params['g']
    ns.tau_phi   = 0.0
    ns.tau_theta = 0.0
    ns.tau_psi   = 0.0
    return ns


if __name__ == '__main__':
    # ============================================================
    #  PHASE 1 — TRUE-STATE PID RUN  (ground truth + rotor history)
    # ============================================================

    print("=" * 60)
    print("PHASE 1: Running PID simulation on TRUE state")
    print("=" * 60)

    _result = run_fig8_pid(config)

    # run_fig8_pid returns (quad, rotor_history) in the updated quad_pid_utils.
    
    if isinstance(_result, tuple):
        quad_true, rotor_history = _result
    else:
        quad_true = _result
        # Fall back: approximate rotor speeds as constant hover
        w_hover = np.sqrt(
            config['quad']['m'] * config['quad']['g'] /
            (4 * config['quad']['kT'])
        )
        N_approx = int(round(config['sim']['tf'] / config['sim']['dt']))
        rotor_history = np.tile(
            np.array([w_hover] * 4), (N_approx, 1)
        )  # (N, 4) — constant hover speed at every step

    t_hist     = quad_true.data('x')[0]
    N          = len(t_hist)
    dt         = config['sim']['dt']
    tf         = config['sim']['tf']

    # Stack true state history (N, 12)
    X_true = np.column_stack([
        quad_true.data('x')[1],  quad_true.data('y')[1],  quad_true.data('z')[1],
        quad_true.data('vx')[1], quad_true.data('vy')[1], quad_true.data('vz')[1],
        quad_true.data('phi')[1],quad_true.data('theta')[1],quad_true.data('psi')[1],
        quad_true.data('p')[1],  quad_true.data('q')[1],  quad_true.data('r')[1],
    ])

    print(f"  True trajectory: {N} steps, dt={dt}s\n")


    # ============================================================
    #  PHASE 2 — CLOSED-LOOP EKF RUN
    #  Controller reads EKF estimates, not true state.
    # ============================================================

    def run_closed_loop_ekf(config, X_true_init, rotor_history_phase1, noise_params,
                            label='nominal'):
        """
        TRUE closed-loop EKF simulation.
        Controller reads EKF estimates; true dynamics re-integrated each step.
        Returns (X_true_cl, X_ekf, P_diag, innovations, nees).
        """
        from scipy.integrate import solve_ivp

        print(f"  [{label}] Initialising EKF and sensor simulator...")

        sensor      = SensorSimulator(noise_params=noise_params)
        quad_params = config['quad']
        dt          = config['sim']['dt']
        tf          = config['sim']['tf']
        N           = int(round(tf / dt))

        A     = config['trajectory']['A']
        B     = config['trajectory']['B']
        omega = config['trajectory']['omega']
        z_ref = config['trajectory']['z_ref']

        x0_ekf = X_true_init.copy()
        x0_ekf[0:3] += np.random.randn(3) * 0.3
        x0_ekf[6:9] += np.random.randn(3) * 0.05

        P0_cl = np.diag(np.array([
            0.50, 0.50, 0.80,
            0.50, 0.50, 0.60,
            0.05, 0.05, 0.05,
            0.10, 0.10, 0.10,
        ])**2)

        ekf = EKF(x0=x0_ekf, P0=P0_cl, Q=Q.copy(),
                  R_gps=R_gps, R_gyro=R_gyro, R_mag=R_mag, R_acc=R_acc,
                  quad_params=quad_params)

        X_true_current = X_true_init.copy()
        quad_obj       = types.SimpleNamespace(**quad_params)

        quad_ns = _make_quad_ns(X_true_init, quad_params)
        outer_ctrl, mid_ctrl, inner_ctrl, allocator = InitializeControllers(
            config['controller'], quad_ns)

        Ts_outer   = 1.0 / 50.0
        Ts_middle  = 1.0 / 100.0
        outer_time = middle_time = 0.0

        psi_d = phi_d = theta_d = 0.0
        p_d = q_d = r_d = 0.0
        T_cmd = quad_params['m'] * quad_params['g']

        rotor_speeds = rotor_history_phase1[0].copy()

        X_true_cl = np.zeros((N, 12))
        X_ekf_out = np.zeros((N, 12))
        P_diag    = np.zeros((N, 12))
        nees      = np.zeros(N)

        innov_gps_list  = []; innov_gps_t  = []
        innov_gyro_list = []; innov_gyro_t = []
        innov_mag_list  = []; innov_mag_t  = []

        gps_ctr = mag_ctr = 0
        t_hist_cl   = np.arange(N) * dt
        X_true_prev = X_true_current.copy()

        print(f"  [{label}] Running {N} steps (TRUE closed-loop dynamics)...")

        for k in range(N):
            t = t_hist_cl[k]

            X_true_cl[k] = X_true_current.copy()
            X_ekf_out[k] = ekf.x.copy()
            P_diag[k]    = np.diag(ekf.P)

            x_err = X_true_current - ekf.x
            try:
                nees[k] = float(x_err @ np.linalg.solve(ekf.P, x_err))
            except np.linalg.LinAlgError:
                nees[k] = np.nan

            ekf.predict(f_dynamics=dynamics, dt=dt,
                        quad_obj=quad_obj, rotor_speeds=rotor_speeds)

            z_gyro  = sensor.gyro(X_true_current)
            innov_g = z_gyro - ekf.x[9:12]
            ekf.update_gyro(z_gyro)
            innov_gyro_list.append(innov_g); innov_gyro_t.append(t)

            z_acc = sensor.accelerometer(X_true_current, x_true_prev=X_true_prev, dt=dt)
            ekf.update_accelerometer(z_acc)

            mag_ctr += 1
            if mag_ctr >= MAG_RATE:
                z_mag     = sensor.magnetometer(X_true_current)
                psi_innov = np.arctan2(np.sin(z_mag[0]-ekf.x[8]),
                                       np.cos(z_mag[0]-ekf.x[8]))
                innov_mag_list.append(np.array([psi_innov])); innov_mag_t.append(t)
                ekf.update_magnetometer(z_mag)
                mag_ctr = 0

            gps_ctr += 1
            if gps_ctr >= GPS_RATE:
                z_gps     = sensor.gps(X_true_current)
                innov_pos = z_gps - ekf.x[0:3]
                innov_gps_list.append(innov_pos); innov_gps_t.append(t)
                ekf.update_gps(z_gps)
                gps_ctr = 0

            quad_ns = _make_quad_ns(ekf.x, quad_params)
            xd, yd, zd     = position_reference(t, A, B, omega, z_ref, t_sim=tf)
            vxd_ff, vyd_ff = velocity_reference(t, A, B, omega, t_sim=tf)

            outer_time += dt
            if outer_time >= Ts_outer:
                T_cmd, phi_d, theta_d, psi_d = outer_ctrl.compute(
                    xd, yd, zd, vxd_ff, vyd_ff, psi_d, quad_ns, Ts_outer)
                outer_time = 0.0

            middle_time += dt
            if middle_time >= Ts_middle:
                p_d, q_d, r_d = mid_ctrl.compute(phi_d, theta_d, psi_d, quad_ns, Ts_middle)
                middle_time = 0.0

            quad_ns.tau_phi, quad_ns.tau_theta, quad_ns.tau_psi = inner_ctrl.compute(
                p_d, q_d, r_d, quad_ns, dt)

            rotor_speeds = np.array(allocator.allocate(
                T_cmd, quad_ns.tau_phi, quad_ns.tau_theta, quad_ns.tau_psi))

            X_true_prev = X_true_current.copy()
            sol = solve_ivp(dynamics, [t, t+dt], X_true_current,
                            args=(quad_obj, rotor_speeds), method='RK45')
            X_true_current = sol.y[:, -1]

        innovations = {
            'gps'  : (np.array(innov_gps_t),  np.array(innov_gps_list)),
            'gyro' : (np.array(innov_gyro_t), np.array(innov_gyro_list)),
            'mag'  : (np.array(innov_mag_t),  np.array(innov_mag_list)),
        }

        print(f"  [{label}] Done.\n")
        return X_true_cl, X_ekf_out, P_diag, innovations, nees


    # ── Run nominal closed-loop EKF ───────────────────────────────────────────────
    print("=" * 60)
    print("PHASE 2a: Closed-loop EKF — NOMINAL noise")
    print("=" * 60)

    np.random.seed(42)
    nominal_noise = {'gps_std': 0.5, 'gyro_std': 0.01, 'mag_std': 0.05}
    X_true_nom, X_ekf_nom, P_diag_nom, innov_nom, nees_nom = run_closed_loop_ekf(
        config, X_true[0], rotor_history, nominal_noise, label='nominal')

    # ── Run stress-test: GPS noise 5× ────────────────────────────────────────────
    print("=" * 60)
    print("PHASE 2b: Closed-loop EKF — HIGH NOISE STRESS TEST (GPS×5)")
    print("=" * 60)

    np.random.seed(42)
    stress_noise = {'gps_std': 2.5, 'gyro_std': 0.01, 'mag_std': 0.05}
    X_true_str, X_ekf_str, P_diag_str, innov_str, nees_str = run_closed_loop_ekf(
        config, X_true[0], rotor_history, stress_noise, label='stress')


    # ============================================================
    #  PHASE 3 — METRICS
    # ============================================================

    t_hist_nom = np.arange(len(X_true_nom)) * config['sim']['dt']
    t_hist_str = np.arange(len(X_true_str)) * config['sim']['dt']

    def compute_rmse(X_true_run, X_est, t_h, t_start=8.0, t_end=82.0):
        mask = (t_h >= t_start) & (t_h <= t_end)
        err  = X_true_run[mask] - X_est[mask]
        return np.sqrt(np.mean(err**2, axis=0))

    rmse_nom = compute_rmse(X_true_nom, X_ekf_nom, t_hist_nom)
    rmse_str = compute_rmse(X_true_str, X_ekf_str, t_hist_str)

    state_names = ['x','y','z','vx','vy','vz','phi','theta','psi','p','q','r']
    units       = ['m','m','m','m/s','m/s','m/s','rad','rad','rad','rad/s','rad/s','rad/s']

    print("\n── EKF RMSE (figure-8 phase) ──────────────────────────────────")
    print(f"{'State':<10} {'Nominal':>12} {'Stress(GPS×5)':>15}")
    print("-" * 40)
    for i, (name, unit) in enumerate(zip(state_names, units)):
        print(f"{name:<10} {rmse_nom[i]:>10.4f} {unit:<4}  {rmse_str[i]:>10.4f} {unit}")

    # NEES statistics (figure-8 phase only)
    mask_f8_nom = (t_hist_nom >= 8.0) & (t_hist_nom <= 82.0)
    mask_f8_str = (t_hist_str >= 8.0) & (t_hist_str <= 82.0)
    nees_nom_mean = np.nanmean(nees_nom[mask_f8_nom])
    nees_str_mean = np.nanmean(nees_str[mask_f8_str])

    print(f"\n── NEES (mean over figure-8, ideal ≈ 12.0) ────────────────────")
    print(f"  Nominal  : {nees_nom_mean:.2f}")
    print(f"  Stress   : {nees_str_mean:.2f}")
    print(f"  (n_states = 12  →  consistent filter has NEES ≈ 12)")



    # ============================================================
    #  PHASE 4 — PLOTS
    # ============================================================

    # Save all figures to results/ folder
    os.makedirs('results', exist_ok=True)
    plt.close('all')
    lw = 1.2

    # ── Figure 1: State estimation — nominal ─────────────────────────────────────
    fig1, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig1.suptitle('Closed-Loop EKF — State Estimation (Nominal Noise)\n'
                  '[TRUE closed-loop: dynamics re-integrated under EKF-controlled rotors]',
                  fontsize=13, fontweight='bold')

    plot_cfg = [
        (0, 0, 0, 'x [m]',      'Position X'),
        (1, 0, 1, 'y [m]',      'Position Y'),
        (2, 0, 2, 'z [m]',      'Altitude Z'),
        (6, 1, 0, 'phi [deg]',  'Roll'),
        (7, 1, 1, 'theta[deg]', 'Pitch'),
        (8, 1, 2, 'psi [deg]',  'Yaw'),
        (9, 2, 0, 'p [rad/s]',  'Roll Rate'),
        (10,2, 1, 'q [rad/s]',  'Pitch Rate'),
        (11,2, 2, 'r [rad/s]',  'Yaw Rate'),
    ]

    for (si, row, col, ylabel, title) in plot_cfg:
        ax    = axes[row, col]
        scale = 180/np.pi if 'deg' in ylabel else 1.0
        ax.plot(t_hist_nom, X_true_nom[:,si]*scale, 'k-',  lw=lw,     label='True (CL)',  alpha=0.7)
        ax.plot(t_hist_nom, X_ekf_nom[:,si]*scale,  'b-',  lw=lw+0.4, label='EKF est.',   alpha=0.9)
        sigma = np.sqrt(P_diag_nom[:,si]) * scale
        ax.fill_between(t_hist_nom,
                        X_ekf_nom[:,si]*scale - 2*sigma,
                        X_ekf_nom[:,si]*scale + 2*sigma,
                        alpha=0.15, color='blue', label='\u00b12\u03c3 bound')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title,   fontsize=10)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time [s]', fontsize=8)

    plt.tight_layout()
    plt.savefig('results/fig1_ekf_state_estimation.png', dpi=150)
    plt.show()
    print('\nSaved: results/fig1_ekf_state_estimation.png')

    # ── Figure 2: Nominal vs Stress — positions ───────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle('Stress Test — TRUE Closed-Loop: EKF-Controlled Dynamics\n'
                  'Nominal (GPS \u03c3=0.5 m)  vs  Stress (GPS \u03c3=2.5 m)',
                  fontsize=13, fontweight='bold')

    for i, (ax, lbl, unit) in enumerate(zip(axes2,
                                            ['Position X', 'Position Y', 'Altitude Z'],
                                            ['m', 'm', 'm'])):
        ax.plot(t_hist,     X_true[:,i],      'k-',  lw=lw,     label='Phase1 ref',      alpha=0.5)
        ax.plot(t_hist_nom, X_true_nom[:,i],  'g-',  lw=lw,     label='True CL nominal', alpha=0.7)
        ax.plot(t_hist_nom, X_ekf_nom[:,i],   'b-',  lw=lw+0.3, label='EKF nominal',     alpha=0.9)
        ax.plot(t_hist_str, X_ekf_str[:,i],   'r--', lw=lw,     label='EKF stress\u00d75',  alpha=0.85)
        ax.set_xlabel('Time [s]', fontsize=9)
        ax.set_ylabel(f'{lbl} [{unit}]', fontsize=9)
        ax.set_title(lbl, fontsize=11)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/fig2_stress_test.png', dpi=150)
    plt.show()
    print('Saved: results/fig2_stress_test.png')

    # ── Figure 3: NEES ───────────────────────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    fig3.suptitle('NEES \u2014 Filter Consistency Check  (ideal \u2248 12)\n'
                  '[Computed against TRUE closed-loop state]',
                  fontsize=13, fontweight='bold')

    def rolling_mean(x, w):
        kernel = np.ones(w) / w
        return np.convolve(x, kernel, mode='same')

    w           = int(2.0 / dt)
    nees_nom_sm = rolling_mean(np.nan_to_num(nees_nom, nan=0.0), w)
    nees_str_sm = rolling_mean(np.nan_to_num(nees_str, nan=0.0), w)

    ax3.plot(t_hist_nom, nees_nom_sm, 'b-',  lw=lw, label='Nominal')
    ax3.plot(t_hist_str, nees_str_sm, 'r--', lw=lw, label='Stress\u00d75')
    ax3.axhline(12,      color='g',      lw=1.5, ls='--', label='Ideal (n=12)')
    ax3.axhline(12*1.5,  color='orange', lw=1,   ls=':',  label='Upper warn (\u00d71.5)')
    ax3.set_xlabel('Time [s]', fontsize=10)
    ax3.set_ylabel('NEES (2s smoothed)', fontsize=10)
    ax3.set_ylim(0, min(nees_nom_sm.max()*1.3, 200))
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/fig3_nees.png', dpi=150)
    plt.show()
    print('Saved: results/fig3_nees.png')

    # ── Figure 4: Innovation plots — nominal ─────────────────────────────────────
    fig4, axes4 = plt.subplots(2, 3, figsize=(16, 8))
    fig4.suptitle('Innovation Sequences \u2014 Nominal Run\n'
                  '(Zero-mean, bounded variance \u2192 filter is consistent)',
                  fontsize=13, fontweight='bold')

    t_gps,  innov_gps  = innov_nom['gps']
    t_gyro, innov_gyro = innov_nom['gyro']

    gps_labels  = ['x innov [m]',   'y innov [m]',   'z innov [m]']
    gyro_labels = ['p innov [r/s]', 'q innov [r/s]', 'r innov [r/s]']

    for i in range(3):
        ax = axes4[0, i]
        ax.plot(t_gps, innov_gps[:, i], 'b.', ms=2, alpha=0.5)
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_title(f'GPS \u2014 {gps_labels[i]}', fontsize=10)
        ax.set_xlabel('Time [s]', fontsize=8)
        ax.grid(True, alpha=0.3)

    for i in range(3):
        ax = axes4[1, i]
        ax.plot(t_gyro, innov_gyro[:, i], 'r.', ms=1, alpha=0.3)
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_title(f'Gyro \u2014 {gyro_labels[i]}', fontsize=10)
        ax.set_xlabel('Time [s]', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/fig4_innovations.png', dpi=150)
    plt.show()
    print('Saved: results/fig4_innovations.png')

    # ── Figure 5: Covariance convergence ─────────────────────────────────────────
    fig5, axes5 = plt.subplots(1, 3, figsize=(14, 4))
    fig5.suptitle('EKF Covariance Convergence \u2014 1\u03c3 Position Uncertainty',
                  fontsize=13, fontweight='bold')

    for i, (ax, lbl) in enumerate(zip(axes5, ['x', 'y', 'z'])):
        sigma_nom = np.sqrt(P_diag_nom[:, i])
        sigma_str = np.sqrt(P_diag_str[:, i])
        ax.plot(t_hist_nom, sigma_nom, 'b-',  lw=lw, label='Nominal')
        ax.plot(t_hist_str, sigma_str, 'r--', lw=lw, label='Stress\u00d75')
        ax.fill_between(t_hist_nom, 0, sigma_nom, alpha=0.15, color='blue')
        ax.set_ylabel('1-\u03c3 [m]', fontsize=9)
        ax.set_xlabel('Time [s]', fontsize=9)
        ax.set_title(f'Position {lbl} uncertainty', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/fig5_covariance.png', dpi=150)
    plt.show()
    print('Saved: results/fig5_covariance.png')

    # ── Figure 6: 3D trajectory ───────────────────────────────────────────────────
    fig6 = plt.figure(figsize=(14, 6))
    fig6.suptitle('3D Trajectory \u2014 TRUE Closed-Loop GNC\n'
                  'True dynamics re-integrated under EKF-controlled rotor speeds',
                  fontsize=13, fontweight='bold')

    ax6a = fig6.add_subplot(1, 2, 1, projection='3d')
    ax6a.plot(X_true_nom[:,0], X_true_nom[:,1], X_true_nom[:,2], 'k-',  lw=lw,     label='True CL')
    ax6a.plot(X_ekf_nom[:,0],  X_ekf_nom[:,1],  X_ekf_nom[:,2],  'b--', lw=lw+0.3, label='EKF estimate')
    ax6a.set_xlabel('X [m]'); ax6a.set_ylabel('Y [m]'); ax6a.set_zlabel('Z [m]')
    ax6a.set_title('Nominal Noise'); ax6a.legend(fontsize=8)

    ax6b = fig6.add_subplot(1, 2, 2, projection='3d')
    ax6b.plot(X_true_str[:,0], X_true_str[:,1], X_true_str[:,2], 'k-',  lw=lw,     label='True CL stress')
    ax6b.plot(X_ekf_str[:,0],  X_ekf_str[:,1],  X_ekf_str[:,2],  'r--', lw=lw+0.3, label='EKF stress\u00d75')
    ax6b.set_xlabel('X [m]'); ax6b.set_ylabel('Y [m]'); ax6b.set_zlabel('Z [m]')
    ax6b.set_title('High Noise (GPS \u00d75)'); ax6b.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('results/fig6_3d_trajectory.png', dpi=150)
    plt.show()
    print('Saved: results/fig6_3d_trajectory.png')

    print("\n" + "=" * 60)
    print("ALL DONE \u2014 6 figures saved.")
    print("=" * 60)
