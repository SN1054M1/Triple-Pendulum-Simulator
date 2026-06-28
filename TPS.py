# -*- coding: utf-8 -*-
"""
Triple Pendulum Simulator (NumPy-only, Memory-optimized + Angle Window Modes)

Features
- Triple pendulum dynamics (Lagrangian form, NumPy-only)
- Integrator: RK4 / semi-implicit Euler
- Precompute + render
- GIF / MP4 export
- Physical statistics panel (energy-first)
- Angle chart modes: unwrap / sin / cos / wrapped
- Angle history modes: full / rolling window
"""

import os
import json
import hashlib
import colorsys
import gc

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

# =========================
# Config (edit here only)
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- Physical parameters ----------
L1, L2, L3 = 1.0, 1.0, 1.0
M1, M2, M3 = 1.0, 1.0, 1.0
G = 10.0
DAMPING = 0.0

# ---------- Simulation parameters ----------
T_TOTAL = 10.0
SIM_DT = 0.1
FPS = 30
PLAYBACK_SPEED = 1.0  # <1 slow, >1 fast

# ---------- Numerical integration ----------
INTEGRATOR = "rk4"  # "rk4" | "semi_euler"
SUBSTEPS = 1

# ---------- Initial conditions ----------
USE_CONSOLE_INPUT = True
DEFAULT_ANGLES_DEG = [10.0, 20.0, 30.0]
INITIAL_OMEGA = [0.0, 0.0, 0.0]

# ---------- Visualization ----------
SHOW_GRID = True
SHOW_INFO = True
SHOW_TRAIL = True
TRAIL_SOURCE = "sim"  # "sim" | "display"
TRAIL_LENGTH = 40     # display frames
TRAIL_ALPHA = 0.85
TRAIL_MAX_POINTS = 1000
TRAIL_SMOOTH_FACTOR = 3
ROD_COLOR = "white"
PLOT_MARGIN = 0.2

# ---------- Figure ----------
FIG_SIZE = (18.0, 10.2)
OUTPUT_DPI = 200

# ---------- Background color ----------
# Available:
# "sin_smooth", "cos_phase", "atan_soft", "linear_wrap",
# "hsv_cycle", "triad_soft", "cmap_theta1", "energy_heat"
COLOR_MODE = "sin_smooth"
COLOR_CMAP = "turbo"
COLOR_DYNAMIC = True
COLOR_TEMPORAL_SMOOTH = True
COLOR_SMOOTH_ALPHA = 0.20

# ---------- Angle chart ----------
# mode:
# "unwrap": eliminate +/-pi jump by adding/subtracting 2pi
# "sin"/"cos": naturally continuous periodic projection
# "wrapped": direct wrapped angle in [-pi, pi]
ANGLE_PLOT_MODE = "unwrap"   # "unwrap" | "sin" | "cos" | "wrapped"
ANGLE_PLOT_YLABEL = {
    "unwrap": "theta (rad, unwrapped)",
    "sin": "sin(theta)",
    "cos": "cos(theta)",
    "wrapped": "theta (rad, wrapped [-pi, pi])",
}

# history display mode:
# "full"   -> show from t=0 to current frame
# "window" -> rolling window [t-ANGLE_WINDOW_SECONDS, t]
ANGLE_HISTORY_MODE = "window"   # "full" | "window"
ANGLE_WINDOW_SECONDS = 5.0

# ---------- Export ----------
SHOW_WINDOW = False
EXPORT_GIF = True
EXPORT_MP4 = False
GIF_NAME = os.path.join(BASE_DIR, "triple_pendulum_numpy_final.gif")
MP4_NAME = os.path.join(BASE_DIR, "triple_pendulum_numpy_final.mp4")
MP4_BITRATE = 7000

# Export subsample:
# 1 means every frame, 2 means every 2nd frame ...
EXPORT_FRAME_STRIDE = 1

# ---------- Cache / dump ----------
USE_TRAJ_CACHE = False
TRAJ_CACHE_DIR = os.path.join(BASE_DIR, "traj_cache")
SAVE_NPZ = False
NPZ_NAME = os.path.join(BASE_DIR, "triple_pendulum_numpy_final_data.npz")

# ---------- Progress ----------
USE_PROGRESS = True
PROGRESS_MIN_INTERVAL = 0.5

# ---------- Numeric guard ----------
STRICT_NUMERIC_CHECK = True

# ---------- Memory options ----------
# Float32 buffer can notably reduce memory footprint
USE_FLOAT32_BUFFERS = True
BUFFER_DTYPE = np.float32 if USE_FLOAT32_BUFFERS else np.float64

# Whether to precompute geometry/trail source arrays.
# Turn off to reduce RAM at cost of extra CPU in update().
PRECOMPUTE_GEOMETRY = True
PRECOMPUTE_TRAIL_SOURCE = True

# -------------------------
# Progress
# -------------------------
try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False


class SimpleProgress:
    def __init__(self, total, desc):
        self.total = max(int(total), 1)
        self.desc = desc
        self.last_percent = -1
        self.closed = False
        self.n = 0

    def update(self, n_done):
        if self.closed:
            return
        self.n = min(int(n_done), self.total)
        p = int(100.0 * self.n / self.total)
        if p != self.last_percent:
            print(f"\r{self.desc}: {p:3d}%", end="", flush=True)
            self.last_percent = p

    def close(self):
        if not self.closed:
            print()
            self.closed = True


def iter_with_progress(iterable, total, desc):
    if not USE_PROGRESS:
        for x in iterable:
            yield x
        return

    if HAS_TQDM:
        for x in tqdm(
            iterable,
            total=total,
            desc=desc,
            ncols=100,
            mininterval=PROGRESS_MIN_INTERVAL,
            leave=True,
            dynamic_ncols=True,
            smoothing=0.05,
        ):
            yield x
        return

    p = SimpleProgress(total, desc)
    for i, x in enumerate(iterable, start=1):
        p.update(i)
        yield x
    p.close()


def make_export_callback(label, total, reserve_final_step=True):
    if not USE_PROGRESS:
        return None, None

    display_total = int(total) + (1 if reserve_final_step else 0)

    if HAS_TQDM:
        bar = tqdm(
            total=display_total,
            desc=label,
            ncols=100,
            mininterval=PROGRESS_MIN_INTERVAL,
            leave=True,
            dynamic_ncols=True,
            smoothing=0.05,
        )

        def cb(i, n):
            target = min(i + 1, total)
            delta = target - bar.n
            if delta > 0:
                bar.update(delta)

        return cb, bar

    p = SimpleProgress(display_total, label)

    def cb(i, n):
        target = min(i + 1, total)
        p.update(target)

    return cb, p


# -------------------------
# Math / Physics
# -------------------------
def wrap_to_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def build_tail_masses(m):
    tail = np.zeros(3, dtype=float)
    tail[2] = m[2]
    tail[1] = m[1] + tail[2]
    tail[0] = m[0] + tail[1]
    return tail


def mass_matrix(q, l, tail):
    M = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            s_ij = tail[max(i, j)]
            M[i, j] = s_ij * l[i] * l[j] * np.cos(q[i] - q[j])
    return M


def dM_dq(q, l, tail):
    dm = np.zeros((3, 3, 3), dtype=float)
    for k in range(3):
        for i in range(3):
            for j in range(3):
                if i == j:
                    dm[k, i, j] = 0.0
                    continue
                s_ij = tail[max(i, j)]
                base = s_ij * l[i] * l[j] * np.sin(q[i] - q[j])
                if k == i:
                    dm[k, i, j] += -base
                if k == j:
                    dm[k, i, j] += base
    return dm


def coriolis_vector(q, dq, l, tail):
    dm = dM_dq(q, l, tail)
    c = np.zeros(3, dtype=float)
    for i in range(3):
        s = 0.0
        for j in range(3):
            for k in range(3):
                gamma = 0.5 * (dm[k, i, j] + dm[j, i, k] - dm[i, j, k])
                s += gamma * dq[j] * dq[k]
        c[i] = s
    return c


def grad_potential(q, l, g, tail):
    return tail * g * l * np.sin(q)


def forward_kinematics(theta, l):
    t1, t2, t3 = theta
    x1 = l[0] * np.sin(t1)
    y1 = -l[0] * np.cos(t1)
    x2 = x1 + l[1] * np.sin(t2)
    y2 = y1 - l[1] * np.cos(t2)
    x3 = x2 + l[2] * np.sin(t3)
    y3 = y2 - l[2] * np.cos(t3)
    return np.array([0.0, x1, x2, x3]), np.array([0.0, y1, y2, y3])


def forward_velocity(theta, omega, l):
    t1, t2, t3 = theta
    w1, w2, w3 = omega

    vx1 = l[0] * np.cos(t1) * w1
    vy1 = l[0] * np.sin(t1) * w1

    vx2 = vx1 + l[1] * np.cos(t2) * w2
    vy2 = vy1 + l[1] * np.sin(t2) * w2

    vx3 = vx2 + l[2] * np.cos(t3) * w3
    vy3 = vy2 + l[2] * np.sin(t3) * w3
    return np.array([vx1, vx2, vx3]), np.array([vy1, vy2, vy3])


def split_energy(q, dq, l, m, g, tail):
    x, y = forward_kinematics(q, l)
    vx, vy = forward_velocity(q, dq, l)
    t_parts = 0.5 * m * (vx * vx + vy * vy)
    v_parts = m * g * y[1:]
    T = float(np.sum(t_parts))
    V = float(np.sum(v_parts))
    return T, V, t_parts, v_parts, x[1:], y[1:], vx, vy


def energy_total(q, dq, l, m, g, tail):
    T, V, _, _, _, _, _, _ = split_energy(q, dq, l, m, g, tail)
    return T + V


def angular_momentum_z(xm, ym, vxm, vym, m):
    return float(np.sum(m * (xm * vym - ym * vxm)))


def stats_at_state(q, dq, l, m, g, tail):
    T, V, T_parts, V_parts, xm, ym, vxm, vym = split_energy(q, dq, l, m, g, tail)
    E = T + V
    Lz = angular_momentum_z(xm, ym, vxm, vym, m)
    omega_rms = float(np.sqrt(np.mean(dq * dq)))
    tip_speed = float(np.hypot(vxm[2], vym[2]))
    tip_radius = float(np.hypot(xm[2], ym[2]))
    kin_ratio = float(T / (abs(V) + 1e-12))
    diss_power = float(-DAMPING * np.sum(dq * dq))

    E1 = float(T_parts[0] + V_parts[0])
    E2 = float(T_parts[1] + V_parts[1])
    E3 = float(T_parts[2] + V_parts[2])

    return {
        "T": T,
        "V": V,
        "E": E,
        "Lz": Lz,
        "omega_rms": omega_rms,
        "tip_speed": tip_speed,
        "tip_radius": tip_radius,
        "kin_ratio": kin_ratio,
        "diss_power": diss_power,
        "T1": float(T_parts[0]),
        "T2": float(T_parts[1]),
        "T3": float(T_parts[2]),
        "V1": float(V_parts[0]),
        "V2": float(V_parts[1]),
        "V3": float(V_parts[2]),
        "E1": E1,
        "E2": E2,
        "E3": E3,
    }


def dynamics(state, l, m, g, damping, tail):
    q = state[:3]
    dq = state[3:]

    M = mass_matrix(q, l, tail)
    c = coriolis_vector(q, dq, l, tail)
    gvec = grad_potential(q, l, g, tail)
    d = damping * dq

    rhs = -(c + gvec + d)
    ddq = np.linalg.solve(M, rhs)

    out = np.zeros(6, dtype=float)
    out[:3] = dq
    out[3:] = ddq
    return out


def step_state(state, dt, l, m, g, damping, tail):
    if INTEGRATOR == "semi_euler":
        d = dynamics(state, l, m, g, damping, tail)
        q = state[:3].copy()
        dq = state[3:].copy()
        dq = dq + dt * d[3:]
        q = q + dt * dq
        return np.concatenate([q, dq])

    k1 = dynamics(state, l, m, g, damping, tail)
    k2 = dynamics(state + 0.5 * dt * k1, l, m, g, damping, tail)
    k3 = dynamics(state + 0.5 * dt * k2, l, m, g, damping, tail)
    k4 = dynamics(state + dt * k3, l, m, g, damping, tail)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def check_finite(arr, name):
    if STRICT_NUMERIC_CHECK and (not np.all(np.isfinite(arr))):
        raise FloatingPointError(f"Non-finite values detected in {name}")


def build_cache_key(theta0_deg):
    payload = {
        "L": [L1, L2, L3],
        "M": [M1, M2, M3],
        "G": G,
        "DAMPING": DAMPING,
        "T_TOTAL": T_TOTAL,
        "SIM_DT": SIM_DT,
        "INTEGRATOR": INTEGRATOR,
        "SUBSTEPS": SUBSTEPS,
        "INITIAL_OMEGA": INITIAL_OMEGA,
        "THETA0_DEG": theta0_deg,
    }
    s = json.dumps(payload, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def simulate(theta0_deg):
    l = np.array([L1, L2, L3], dtype=float)
    m = np.array([M1, M2, M3], dtype=float)
    tail = build_tail_masses(m)

    cache_hit = False
    if USE_TRAJ_CACHE:
        os.makedirs(TRAJ_CACHE_DIR, exist_ok=True)
        key = build_cache_key(theta0_deg)
        cache_file = os.path.join(TRAJ_CACHE_DIR, f"traj_{key}.npz")
        if os.path.exists(cache_file):
            data = np.load(cache_file)
            ts = data["ts"]
            states = data["states"]
            cache_hit = True
            print(f"[INFO] Loaded trajectory cache: {cache_file}")
            return ts, states, l, m, tail, cache_hit

    theta0 = np.deg2rad(np.array(theta0_deg, dtype=float))
    omega0 = np.array(INITIAL_OMEGA, dtype=float)
    state = np.concatenate([theta0, omega0])

    n_steps = int(np.floor(T_TOTAL / SIM_DT))
    ts = np.linspace(0.0, n_steps * SIM_DT, n_steps + 1, dtype=float)
    states = np.zeros((n_steps + 1, 6), dtype=BUFFER_DTYPE)
    states[0] = state.astype(BUFFER_DTYPE, copy=False)

    sub = max(int(SUBSTEPS), 1)
    dt_sub = SIM_DT / sub

    for i in iter_with_progress(range(n_steps), n_steps, "Simulating"):
        for _ in range(sub):
            state = step_state(state, dt_sub, l, m, G, DAMPING, tail)
            check_finite(state, "state")
        states[i + 1] = state.astype(BUFFER_DTYPE, copy=False)

    if USE_TRAJ_CACHE:
        key = build_cache_key(theta0_deg)
        cache_file = os.path.join(TRAJ_CACHE_DIR, f"traj_{key}.npz")
        np.savez_compressed(cache_file, ts=ts, states=states)
        print(f"[OK] Saved trajectory cache: {cache_file}")

    return ts, states, l, m, tail, cache_hit


# -------------------------
# Rendering / Color / Stats
# -------------------------
def angle_to_channel(angle, mode):
    if mode == "sin_smooth":
        return 0.5 + 0.5 * np.sin(angle)
    if mode == "cos_phase":
        return 0.5 + 0.5 * np.cos(angle)
    if mode == "atan_soft":
        return 0.5 + np.arctan(angle) / np.pi
    w = wrap_to_pi(angle)
    return (w + np.pi) / (2.0 * np.pi)


def theta_to_rgb(theta, mode, energy_value=None, e_min=None, e_max=None):
    t1, t2, t3 = theta

    if mode in ("sin_smooth", "cos_phase", "atan_soft", "linear_wrap"):
        return (
            angle_to_channel(t1, mode),
            angle_to_channel(t2, mode),
            angle_to_channel(t3, mode),
        )

    if mode == "hsv_cycle":
        hue = ((t1 + 0.5 * t2 - 0.25 * t3) / (2.0 * np.pi)) % 1.0
        sat = 0.65 + 0.25 * np.clip(abs(np.sin(t2 - t1)), 0.0, 1.0)
        val = 0.55 + 0.40 * np.clip(abs(np.cos(t3 - t2)), 0.0, 1.0)
        return colorsys.hsv_to_rgb(hue, min(sat, 1.0), min(val, 1.0))

    if mode == "triad_soft":
        r = 0.5 + 0.5 * np.sin(t1)
        g = 0.5 + 0.5 * np.sin(t2 + 2.0 * np.pi / 3.0)
        b = 0.5 + 0.5 * np.sin(t3 + 4.0 * np.pi / 3.0)
        return (r, g, b)

    if mode == "cmap_theta1":
        cmap = plt.get_cmap(COLOR_CMAP)
        u = (wrap_to_pi(t1) + np.pi) / (2.0 * np.pi)
        base = np.array(cmap(u)[:3], dtype=float)
        mod = 0.85 + 0.15 * np.array([
            0.5 + 0.5 * np.sin(t2),
            0.5 + 0.5 * np.sin(t3),
            0.5 + 0.5 * np.sin(t2 - t3),
        ])
        return tuple(np.clip(base * mod, 0.0, 1.0).tolist())

    if mode == "energy_heat":
        cmap = plt.get_cmap(COLOR_CMAP)
        if (energy_value is None) or (e_min is None) or (e_max is None) or (e_max <= e_min):
            u = 0.5
        else:
            u = float(np.clip((energy_value - e_min) / (e_max - e_min), 0.0, 1.0))
        return tuple(cmap(u)[:3])

    return (
        angle_to_channel(t1, "sin_smooth"),
        angle_to_channel(t2, "sin_smooth"),
        angle_to_channel(t3, "sin_smooth"),
    )


def build_background_colors(q_out, energy_out=None):
    n = len(q_out)
    bg = np.zeros((n, 3), dtype=float)

    e_min = None
    e_max = None
    if energy_out is not None and len(energy_out) > 0:
        e_min = float(np.min(energy_out))
        e_max = float(np.max(energy_out))

    for i in iter_with_progress(range(n), n, "Color mapping"):
        bg[i] = theta_to_rgb(
            q_out[i],
            COLOR_MODE,
            None if energy_out is None else energy_out[i],
            e_min,
            e_max,
        )

    if COLOR_TEMPORAL_SMOOTH and n > 1:
        alpha = float(np.clip(COLOR_SMOOTH_ALPHA, 0.0, 1.0))
        for i in range(1, n):
            bg[i] = alpha * bg[i] + (1.0 - alpha) * bg[i - 1]

    if not COLOR_DYNAMIC:
        bg[:] = bg[0]

    return bg


def build_angle_series(q_out, mode):
    if mode == "sin":
        return np.sin(q_out)
    if mode == "cos":
        return np.cos(q_out)
    if mode == "unwrap":
        return np.unwrap(q_out, axis=0)
    out = np.empty_like(q_out)
    for k in range(q_out.shape[1]):
        out[:, k] = wrap_to_pi(q_out[:, k])
    return out


def sample_for_render(ts_sim, states_sim, l, m, tail):
    frame_dt = (1.0 / FPS) * max(int(EXPORT_FRAME_STRIDE), 1)
    t_out = np.arange(0.0, ts_sim[-1] + 1e-12, frame_dt, dtype=float)
    t_lookup = np.clip(t_out * PLAYBACK_SPEED, ts_sim[0], ts_sim[-1])

    q_sim = states_sim[:, :3].astype(float, copy=False)
    dq_sim = states_sim[:, 3:].astype(float, copy=False)

    n = len(t_out)
    q_out = np.zeros((n, 3), dtype=BUFFER_DTYPE)
    dq_out = np.zeros((n, 3), dtype=BUFFER_DTYPE)
    for j in range(3):
        q_out[:, j] = np.interp(t_lookup, ts_sim, q_sim[:, j]).astype(BUFFER_DTYPE)
        dq_out[:, j] = np.interp(t_lookup, ts_sim, dq_sim[:, j]).astype(BUFFER_DTYPE)

    if PRECOMPUTE_GEOMETRY:
        xs = np.zeros((n, 4), dtype=BUFFER_DTYPE)
        ys = np.zeros((n, 4), dtype=BUFFER_DTYPE)
        for i in iter_with_progress(range(n), n, "Frame geometry"):
            xx, yy = forward_kinematics(q_out[i].astype(float), l)
            xs[i] = xx.astype(BUFFER_DTYPE)
            ys[i] = yy.astype(BUFFER_DTYPE)
    else:
        xs = None
        ys = None

    stats = {
        "T": np.zeros(n, dtype=BUFFER_DTYPE),
        "V": np.zeros(n, dtype=BUFFER_DTYPE),
        "E": np.zeros(n, dtype=BUFFER_DTYPE),
        "Lz": np.zeros(n, dtype=BUFFER_DTYPE),
        "omega_rms": np.zeros(n, dtype=BUFFER_DTYPE),
        "tip_speed": np.zeros(n, dtype=BUFFER_DTYPE),
        "tip_radius": np.zeros(n, dtype=BUFFER_DTYPE),
        "kin_ratio": np.zeros(n, dtype=BUFFER_DTYPE),
        "diss_power": np.zeros(n, dtype=BUFFER_DTYPE),
        "T1": np.zeros(n, dtype=BUFFER_DTYPE),
        "T2": np.zeros(n, dtype=BUFFER_DTYPE),
        "T3": np.zeros(n, dtype=BUFFER_DTYPE),
        "V1": np.zeros(n, dtype=BUFFER_DTYPE),
        "V2": np.zeros(n, dtype=BUFFER_DTYPE),
        "V3": np.zeros(n, dtype=BUFFER_DTYPE),
        "E1": np.zeros(n, dtype=BUFFER_DTYPE),
        "E2": np.zeros(n, dtype=BUFFER_DTYPE),
        "E3": np.zeros(n, dtype=BUFFER_DTYPE),
    }

    for i in iter_with_progress(range(n), n, "Physical stats"):
        st = stats_at_state(q_out[i].astype(float), dq_out[i].astype(float), l, m, G, tail)
        for k in stats:
            stats[k][i] = st[k]

    E0 = float(stats["E"][0])
    stats["dE"] = (stats["E"] - E0).astype(BUFFER_DTYPE)
    stats["dE_rel"] = (stats["dE"] / (abs(E0) + 1e-12)).astype(BUFFER_DTYPE)

    bg = build_background_colors(q_out.astype(float), stats["E"].astype(float)).astype(BUFFER_DTYPE)
    angle_series = build_angle_series(q_out.astype(float), ANGLE_PLOT_MODE).astype(BUFFER_DTYPE)

    if SHOW_TRAIL and PRECOMPUTE_TRAIL_SOURCE:
        if TRAIL_SOURCE == "display":
            mass_x = np.zeros((n, 3), dtype=BUFFER_DTYPE)
            mass_y = np.zeros((n, 3), dtype=BUFFER_DTYPE)
            for i in iter_with_progress(range(n), n, "Trail source prep"):
                xx, yy = forward_kinematics(q_out[i].astype(float), l)
                mass_x[i] = xx[1:].astype(BUFFER_DTYPE)
                mass_y[i] = yy[1:].astype(BUFFER_DTYPE)
            ts_trail = t_lookup.copy()
        else:
            n_sim = len(ts_sim)
            mass_x = np.zeros((n_sim, 3), dtype=BUFFER_DTYPE)
            mass_y = np.zeros((n_sim, 3), dtype=BUFFER_DTYPE)
            for i in iter_with_progress(range(n_sim), n_sim, "Trail source prep"):
                xx, yy = forward_kinematics(q_sim[i], l)
                mass_x[i] = xx[1:].astype(BUFFER_DTYPE)
                mass_y[i] = yy[1:].astype(BUFFER_DTYPE)
            ts_trail = ts_sim.copy()
    else:
        mass_x = None
        mass_y = None
        ts_trail = None

    return t_out, t_lookup, q_out, dq_out, xs, ys, bg, ts_trail, mass_x, mass_y, stats, angle_series


def finalize_progress(progress_obj):
    if progress_obj is None:
        return
    remaining = getattr(progress_obj, "total", 0) - getattr(progress_obj, "n", 0)
    if remaining > 0:
        progress_obj.update(remaining)
    progress_obj.close()


def save_with_progress(ani, path, writer, label, total_frames):
    cb, progress_obj = make_export_callback(label, total_frames, reserve_final_step=True)

    print(f"[INFO] {label}: writing frames...")
    if cb is None:
        ani.save(path, writer=writer, dpi=OUTPUT_DPI)
    else:
        ani.save(path, writer=writer, dpi=OUTPUT_DPI, progress_callback=cb)

    print(f"[INFO] {label}: finalizing file...")
    finalize_progress(progress_obj)


# 物理参数含义说明（用于打印面板）
# Meanings of printed physical parameters in side panel
PHYS_PARAM_MEANINGS = {
    "L": "pendulum lengths of 3 links",
    "M": "point masses of 3 bobs",
    "g": "gravity acceleration",
    "damping": "linear damping on generalized angular velocity",
    "theta0": "initial 3 angles in degree",
    "omega0": "initial 3 angular velocities in rad/s",
    "integrator": "time integrator type",
    "substeps": "inner integration substeps",
    "T_total": "total simulated physical time",
    "dt": "simulation time step",
    "FPS": "render frame rate",
    "playback": "display-time speed multiplier",
}


def build_side_panel_text(input_angles_deg):
    lines = [
        "Physical / Numerical Parameters",
        "-------------------------------",
        f"L = [{L1:.5g}, {L2:.5g}, {L3:.5g}]  # {PHYS_PARAM_MEANINGS['L']}",
        f"M = [{M1:.5g}, {M2:.5g}, {M3:.5g}]  # {PHYS_PARAM_MEANINGS['M']}",
        f"g = {G:.5g}  # {PHYS_PARAM_MEANINGS['g']}",
        f"damping = {DAMPING:.5g}  # {PHYS_PARAM_MEANINGS['damping']}",
        "",
        f"theta0(deg) = [{input_angles_deg[0]:.5g}, {input_angles_deg[1]:.5g}, {input_angles_deg[2]:.5g}]  # {PHYS_PARAM_MEANINGS['theta0']}",
        f"omega0(rad/s) = [{INITIAL_OMEGA[0]:.5g}, {INITIAL_OMEGA[1]:.5g}, {INITIAL_OMEGA[2]:.5g}]  # {PHYS_PARAM_MEANINGS['omega0']}",
        "",
        f"Integrator = {INTEGRATOR}  # {PHYS_PARAM_MEANINGS['integrator']}",
        f"Substeps = {SUBSTEPS}  # {PHYS_PARAM_MEANINGS['substeps']}",
        f"T_total = {T_TOTAL:.5g} s  # {PHYS_PARAM_MEANINGS['T_total']}",
        f"dt = {SIM_DT:.5g} s  # {PHYS_PARAM_MEANINGS['dt']}",
        f"FPS = {FPS}  # {PHYS_PARAM_MEANINGS['FPS']}",
        f"Playback = {PLAYBACK_SPEED:.4g}x  # {PHYS_PARAM_MEANINGS['playback']}",
        "",
        f"Angle chart mode = {ANGLE_PLOT_MODE}",
        f"Angle history mode = {ANGLE_HISTORY_MODE}",
        "",
        "Energy view = E1 / E2 / E3",
        "Main view = triple pendulum",
    ]
    return "\n".join(lines)


def render_and_export(
    t_out, t_lookup, q_out, dq_out, xs, ys, bg,
    ts_trail, mass_x, mass_y, stats, angle_series, input_angles_deg, l
):
    fig = plt.figure(figsize=FIG_SIZE)
    fig.subplots_adjust(left=0.02, right=0.995, top=0.972, bottom=0.04)

    gs = fig.add_gridspec(
        3, 2,
        width_ratios=[1.50, 0.82],
        height_ratios=[0.31, 0.27, 0.42],
        wspace=0.045, hspace=0.085
    )

    ax = fig.add_subplot(gs[:, 0])
    ax_angle = fig.add_subplot(gs[0, 1])
    ax_energy = fig.add_subplot(gs[1, 1])

    right_bottom = gs[2, 1].subgridspec(2, 1, height_ratios=[0.60, 0.40], hspace=0.10)
    ax_panel = fig.add_subplot(right_bottom[0, 0])
    ax_live = fig.add_subplot(right_bottom[1, 0])

    ax_panel.axis("off")
    ax_live.axis("off")

    rmax = L1 + L2 + L3 + PLOT_MARGIN
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("W")
    ax.set_title("Triple Pendulum | Main View", fontsize=12, pad=6)
    ax.grid(SHOW_GRID, alpha=0.25)

    ax_panel.text(
        0.01, 0.99, build_side_panel_text(input_angles_deg),
        ha="left", va="top", family="monospace", fontsize=6.2,
        linespacing=1.02
    )

    side_dynamic = ax_live.text(
        0.01, 0.99, "",
        ha="left", va="top", family="monospace", fontsize=6.6,
        linespacing=1.04
    )

    line, = ax.plot([], [], "o-", lw=2.2, ms=6.0, color=ROD_COLOR, antialiased=True)
    trail1, = ax.plot([], [], "-", lw=1.5, color=(1.0, 0.2, 0.2), alpha=TRAIL_ALPHA,
                      antialiased=True, solid_capstyle="round", solid_joinstyle="round")
    trail2, = ax.plot([], [], "-", lw=1.5, color=(0.2, 1.0, 0.2), alpha=TRAIL_ALPHA,
                      antialiased=True, solid_capstyle="round", solid_joinstyle="round")
    trail3, = ax.plot([], [], "-", lw=1.5, color=(0.2, 0.5, 1.0), alpha=TRAIL_ALPHA,
                      antialiased=True, solid_capstyle="round", solid_joinstyle="round")

    ax_angle.set_title("Angles", pad=2, fontsize=8.4)
    ax_angle.set_xlabel("display time (s)", fontsize=8)
    ax_angle.set_ylabel(ANGLE_PLOT_YLABEL.get(ANGLE_PLOT_MODE, "value"), fontsize=8)
    ax_angle.tick_params(labelsize=7)
    ax_angle.grid(True, alpha=0.25)
    ax_angle.set_xlim(0.0, max(float(ANGLE_WINDOW_SECONDS), 1e-9))

    y_min = float(np.min(angle_series))
    y_max = float(np.max(angle_series))
    if np.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    pad = 0.08 * (y_max - y_min)
    ax_angle.set_ylim(y_min - pad, y_max + pad)

    c1 = (1.0, 0.35, 0.35)
    c2 = (0.35, 1.0, 0.45)
    c3 = (0.45, 0.62, 1.0)

    a1_line, = ax_angle.plot([], [], "-", lw=1.6, color=c1, label="theta1")
    a2_line, = ax_angle.plot([], [], "-", lw=1.6, color=c2, label="theta2")
    a3_line, = ax_angle.plot([], [], "-", lw=1.6, color=c3, label="theta3")
    a1_dot, = ax_angle.plot([], [], "o", ms=4.6, color=c1)
    a2_dot, = ax_angle.plot([], [], "o", ms=4.6, color=c2)
    a3_dot, = ax_angle.plot([], [], "o", ms=4.6, color=c3)
    ax_angle.legend(
        loc="upper left",
        ncol=3,
        framealpha=0.18,
        fontsize=6.1,
        borderpad=0.25,
        handlelength=1.8,
        columnspacing=0.9
    )

    e1_line, = ax_energy.plot([], [], "-", lw=1.6, color=c1, label="E1")
    e2_line, = ax_energy.plot([], [], "-", lw=1.6, color=c2, label="E2")
    e3_line, = ax_energy.plot([], [], "-", lw=1.6, color=c3, label="E3")
    ax_energy.set_title("Energies", pad=2, fontsize=8.4)
    ax_energy.set_xlabel("display time (s)", fontsize=8)
    ax_energy.set_ylabel("energy", fontsize=8)
    ax_energy.tick_params(labelsize=7)
    ax_energy.grid(True, alpha=0.25)
    ax_energy.set_xlim(0.0, max(float(ANGLE_WINDOW_SECONDS), 1e-9))

    e_min = min(float(np.min(stats["E1"])), float(np.min(stats["E2"])), float(np.min(stats["E3"])))
    e_max = max(float(np.max(stats["E1"])), float(np.max(stats["E2"])), float(np.max(stats["E3"])))
    if np.isclose(e_min, e_max):
        e_min -= 1.0
        e_max += 1.0
    epad = 0.08 * (e_max - e_min)
    ax_energy.set_ylim(e_min - epad, e_max + epad)
    ax_energy.legend(
        loc="upper left",
        ncol=3,
        framealpha=0.18,
        fontsize=6.1,
        borderpad=0.25,
        handlelength=1.8,
        columnspacing=0.9
    )

    def init():
        line.set_data([], [])
        trail1.set_data([], [])
        trail2.set_data([], [])
        trail3.set_data([], [])

        a1_line.set_data([], [])
        a2_line.set_data([], [])
        a3_line.set_data([], [])
        a1_dot.set_data([], [])
        a2_dot.set_data([], [])
        a3_dot.set_data([], [])

        e1_line.set_data([], [])
        e2_line.set_data([], [])
        e3_line.set_data([], [])

        side_dynamic.set_text("")
        return (
            line, trail1, trail2, trail3,
            a1_line, a2_line, a3_line,
            a1_dot, a2_dot, a3_dot,
            e1_line, e2_line, e3_line,
            side_dynamic
        )

    def densify_polyline_local(x, y, factor):
        if factor <= 1 or len(x) < 2:
            return x, y
        n = len(x)
        src = np.arange(n, dtype=float)
        dst = np.linspace(0.0, n - 1, (n - 1) * factor + 1)
        xd = np.interp(dst, src, x)
        yd = np.interp(dst, src, y)
        return xd, yd

    def trail_from_dynamic_source(mass_idx, t_cur, t_span):
        t_axis = t_lookup
        q_axis = q_out.astype(float)

        t0 = max(float(t_axis[0]), float(t_cur - t_span))
        i0 = np.searchsorted(t_axis, t0, side="left")
        i1 = np.searchsorted(t_axis, t_cur, side="right")
        if i1 <= i0:
            i1 = min(i0 + 1, len(t_axis))

        x = []
        y = []
        for k in range(i0, i1):
            xx, yy = forward_kinematics(q_axis[k], l)
            x.append(xx[mass_idx + 1])
            y.append(yy[mass_idx + 1])

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if len(x) > TRAIL_MAX_POINTS:
            stride = int(np.ceil(len(x) / TRAIL_MAX_POINTS))
            x = x[::stride]
            y = y[::stride]

        x, y = densify_polyline_local(x, y, TRAIL_SMOOTH_FACTOR)
        return x, y

    def get_trail_segment_cached(mass_idx, t_cur, t_span):
        t0 = max(float(ts_trail[0]), float(t_cur - t_span))
        i0 = np.searchsorted(ts_trail, t0, side="left")
        i1 = np.searchsorted(ts_trail, t_cur, side="right")
        if i1 <= i0:
            i1 = min(i0 + 1, len(ts_trail))

        x = mass_x[i0:i1, mass_idx].astype(float, copy=False)
        y = mass_y[i0:i1, mass_idx].astype(float, copy=False)

        if len(x) > TRAIL_MAX_POINTS:
            stride = int(np.ceil(len(x) / TRAIL_MAX_POINTS))
            x = x[::stride]
            y = y[::stride]

        x, y = densify_polyline_local(x, y, TRAIL_SMOOTH_FACTOR)
        return x, y

    def update(i):
        ax.set_facecolor(bg[i])

        if PRECOMPUTE_GEOMETRY and xs is not None and ys is not None:
            cur_x = xs[i]
            cur_y = ys[i]
        else:
            cur_x, cur_y = forward_kinematics(q_out[i].astype(float), l)

        line.set_data(cur_x, cur_y)

        t_cur = float(t_out[i])

        if SHOW_TRAIL:
            trail_seconds = (TRAIL_LENGTH / FPS) * max(PLAYBACK_SPEED, 1e-12)

            if PRECOMPUTE_TRAIL_SOURCE and (mass_x is not None) and (mass_y is not None):
                x1, y1 = get_trail_segment_cached(0, t_cur, trail_seconds)
                x2, y2 = get_trail_segment_cached(1, t_cur, trail_seconds)
                x3, y3 = get_trail_segment_cached(2, t_cur, trail_seconds)
            else:
                x1, y1 = trail_from_dynamic_source(0, t_cur, trail_seconds)
                x2, y2 = trail_from_dynamic_source(1, t_cur, trail_seconds)
                x3, y3 = trail_from_dynamic_source(2, t_cur, trail_seconds)

            trail1.set_data(x1, y1)
            trail2.set_data(x2, y2)
            trail3.set_data(x3, y3)
        else:
            trail1.set_data([], [])
            trail2.set_data([], [])
            trail3.set_data([], [])

        window_seconds = max(float(ANGLE_WINDOW_SECONDS), 1e-9)

        if ANGLE_HISTORY_MODE == "window":
            if t_cur < window_seconds:
                x0 = 0.0
                x1 = window_seconds
                i0 = 0
            else:
                x1 = t_cur
                x0 = t_cur - window_seconds
                i0 = int(np.searchsorted(t_out, x0, side="left"))
        else:
            x0 = 0.0
            x1 = max(float(t_out[-1]), 1e-9)
            i0 = 0

        tt = t_out[i0:i + 1]

        ax_angle.set_xlim(x0, x1)
        ax_energy.set_xlim(x0, x1)

        a1_line.set_data(tt, angle_series[i0:i + 1, 0])
        a2_line.set_data(tt, angle_series[i0:i + 1, 1])
        a3_line.set_data(tt, angle_series[i0:i + 1, 2])
        a1_dot.set_data([t_out[i]], [angle_series[i, 0]])
        a2_dot.set_data([t_out[i]], [angle_series[i, 1]])
        a3_dot.set_data([t_out[i]], [angle_series[i, 2]])

        e1_line.set_data(tt, stats["E1"][i0:i + 1])
        e2_line.set_data(tt, stats["E2"][i0:i + 1])
        e3_line.set_data(tt, stats["E3"][i0:i + 1])

        if SHOW_INFO:
            deg = np.rad2deg(q_out[i].astype(float))
            side_dynamic.set_text(
                "Runtime\n"
                "------\n"
                f"t = {t_out[i]:.4f} s\n"
                f"E = {float(stats['E'][i]): .7e}\n"
                f"dE/E0 = {100.0 * float(stats['dE_rel'][i]): .4e} %\n"
                f"T = {float(stats['T'][i]): .7e}\n"
                f"V = {float(stats['V'][i]): .7e}\n"
                f"tip_speed = {float(stats['tip_speed'][i]): .7e}\n"
            )
        else:
            side_dynamic.set_text("")

        return (
            line, trail1, trail2, trail3,
            a1_line, a2_line, a3_line,
            a1_dot, a2_dot, a3_dot,
            e1_line, e2_line, e3_line,
            side_dynamic
        )

    ani = FuncAnimation(
        fig,
        update,
        frames=len(t_out),
        init_func=init,
        interval=1000.0 / FPS,
        blit=False,
        cache_frame_data=False,
    )

    total_frames = len(t_out)

    if EXPORT_GIF:
        print("[INFO] GIF export is two-phase: frame rendering, then final file commit.")
        save_with_progress(ani, GIF_NAME, PillowWriter(fps=FPS), "Export GIF", total_frames)
        print(f"[OK] GIF exported: {GIF_NAME}")

    if EXPORT_MP4:
        print("[INFO] MP4 export is two-phase: frame rendering, then final file commit.")
        save_with_progress(
            ani,
            MP4_NAME,
            FFMpegWriter(fps=FPS, bitrate=MP4_BITRATE),
            "Export MP4",
            total_frames,
        )
        print(f"[OK] MP4 exported: {MP4_NAME}")

    if SHOW_WINDOW:
        plt.show()
    else:
        plt.close(fig)


def read_initial_angles():
    if not USE_CONSOLE_INPUT:
        return DEFAULT_ANGLES_DEG

    s = input("Enter theta1 theta2 theta3 in degrees (example: 10 20 30): ").strip()
    parts = s.split()
    if len(parts) != 3:
        raise ValueError("Please input exactly 3 numbers.")
    return [float(parts[0]), float(parts[1]), float(parts[2])]


def main():
    if EXPORT_GIF and (T_TOTAL > 120 or FPS > 40):
        print("[WARN] GIF can be memory-heavy. Prefer MP4 or increase EXPORT_FRAME_STRIDE.")

    input_angles_deg = read_initial_angles()

    print("[INFO] Precomputing NumPy dynamics...")
    ts_sim, states_sim, l, m, tail, cache_hit = simulate(input_angles_deg)
    print(f"[INFO] Simulation done. Steps = {len(ts_sim)} | Cache hit = {cache_hit}")

    print("[INFO] Preparing render buffers...")
    (
        t_out, t_lookup, q_out, dq_out, xs, ys, bg,
        ts_trail, mass_x, mass_y, stats, angle_series
    ) = sample_for_render(ts_sim, states_sim, l, m, tail)
    print(f"[INFO] Render frames prepared. Frames = {len(t_out)} | Speed = {PLAYBACK_SPEED:.2f}x")

    if SAVE_NPZ:
        np.savez(
            NPZ_NAME,
            ts_sim=ts_sim,
            states_sim=states_sim,
            t_out=t_out,
            t_lookup=t_lookup,
            q_out=q_out,
            dq_out=dq_out,
            xs=xs if xs is not None else np.array([]),
            ys=ys if ys is not None else np.array([]),
            bg=bg,
            ts_trail=ts_trail if ts_trail is not None else np.array([]),
            mass_x=mass_x if mass_x is not None else np.array([]),
            mass_y=mass_y if mass_y is not None else np.array([]),
            angle_series=angle_series,
            stats_T=stats["T"],
            stats_V=stats["V"],
            stats_E=stats["E"],
            stats_dE=stats["dE"],
            stats_dE_rel=stats["dE_rel"],
            stats_Lz=stats["Lz"],
            stats_omega_rms=stats["omega_rms"],
            stats_tip_speed=stats["tip_speed"],
            stats_tip_radius=stats["tip_radius"],
            stats_kin_ratio=stats["kin_ratio"],
            stats_diss_power=stats["diss_power"],
            stats_T1=stats["T1"], stats_T2=stats["T2"], stats_T3=stats["T3"],
            stats_V1=stats["V1"], stats_V2=stats["V2"], stats_V3=stats["V3"],
            stats_E1=stats["E1"], stats_E2=stats["E2"], stats_E3=stats["E3"],
        )
        print(f"[OK] Precomputed data saved: {NPZ_NAME}")

    render_and_export(
        t_out, t_lookup, q_out, dq_out, xs, ys, bg,
        ts_trail, mass_x, mass_y, stats, angle_series, input_angles_deg, l
    )

    del ts_sim, states_sim, q_out, dq_out, bg, angle_series, stats
    del t_out, t_lookup, xs, ys, ts_trail, mass_x, mass_y
    gc.collect()


if __name__ == "__main__":
    main()