# -*- coding: utf-8 -*-
"""
# Triple Pendulum Simulator (NumPy-only)

## Quick Start
1. Install:
   pip install numpy matplotlib tqdm
2. Run:
   python test.py
3. Optional:
   - Set USE_CONSOLE_INPUT = False to use DEFAULT_ANGLES_DEG
   - Set EXPORT_GIF / EXPORT_MP4 to control outputs
   - Set SHOW_WINDOW = True for preview window

## Model
Generalized coordinates:
q = [theta1, theta2, theta3]^T
Lagrangian equation:
M(q) * ddq + C(q, dq) + G(q) + D(dq) = 0
Numerical integrator:
RK4 (default) or semi-implicit Euler
"""

import os
import json
import hashlib
import colorsys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

# =========================
# Config (edit here only)
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- 物理参数 ----------
# ---------- Physical parameters ----------
L1, L2, L3 = 1.0, 1.0, 1.0
M1, M2, M3 = 1.0, 1.0, 1.0
G = 10.0
DAMPING = 0.0

# ---------- 仿真参数 ----------
# ---------- Simulation parameters ----------
T_TOTAL = 10.0
SIM_DT = 0.01
FPS = 30
PLAYBACK_SPEED = 1.0  # <1 slow, >1 fast

# ---------- 数值积分 ----------
# ---------- Numerical integration ----------
INTEGRATOR = "rk4"     # "rk4" | "semi_euler"
SUBSTEPS = 1           # internal substeps per SIM_DT

# ---------- 初值 ----------
# ---------- Initial conditions ----------
USE_CONSOLE_INPUT = True
DEFAULT_ANGLES_DEG = [10.0, 20.0, 30.0]
INITIAL_OMEGA = [0.0, 0.0, 0.0]

# ---------- 可视化 ----------
# ---------- Visualization ----------
SHOW_GRID = True
SHOW_INFO = True
SHOW_TRAIL = True
TRAIL_SOURCE = "sim"   # "sim" | "display"
TRAIL_LENGTH = 40      # display frames
TRAIL_ALPHA = 0.85
TRAIL_MAX_POINTS = 1000
TRAIL_SMOOTH_FACTOR = 3
ROD_COLOR = "white"
PLOT_MARGIN = 0.2

# ---------- 输出分辨率 ----------
# ---------- Output resolution ----------
FIG_SIZE = (14.0, 8.6)
OUTPUT_DPI = 240

# ---------- 背景配色 ----------
# ---------- Background color ----------
# Available:
# "sin_smooth", "cos_phase", "atan_soft", "linear_wrap",
# "hsv_cycle", "triad_soft", "cmap_theta1", "energy_heat"
COLOR_MODE = "sin_smooth"
COLOR_CMAP = "turbo"
COLOR_DYNAMIC = True
COLOR_TEMPORAL_SMOOTH = True
COLOR_SMOOTH_ALPHA = 0.20

# ---------- 输出 ----------
# ---------- Output ----------
SHOW_WINDOW = False    # no auto preview by default
EXPORT_GIF = True
EXPORT_MP4 = False
GIF_NAME = os.path.join(BASE_DIR, "triple_pendulum_numpy.gif")
MP4_NAME = os.path.join(BASE_DIR, "triple_pendulum_numpy.mp4")
MP4_BITRATE = 7000

# ---------- 缓存与存档 ----------
# ---------- Cache and data dump ----------
USE_TRAJ_CACHE = False
TRAJ_CACHE_DIR = os.path.join(BASE_DIR, "traj_cache")
SAVE_NPZ = False
NPZ_NAME = os.path.join(BASE_DIR, "triple_pendulum_numpy_data.npz")

# ---------- 进度 ----------
# ---------- Progress ----------
USE_PROGRESS = True
PROGRESS_MIN_INTERVAL = 0.5

# ---------- 数值检查 ----------
# ---------- Numeric guard ----------
STRICT_NUMERIC_CHECK = True

# -------------------------
# Progress system
# -------------------------
try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False


class SimpleProgress:
    # 中文：无 tqdm 的单行进度显示，不重复刷屏
    # English: single-line fallback progress display without flooding
    def __init__(self, total, desc):
        self.total = max(int(total), 1)
        self.desc = desc
        self.last_percent = -1
        self.closed = False

    def update(self, n_done):
        if self.closed:
            return
        percent = int(100.0 * n_done / self.total)
        if percent != self.last_percent:
            print(f"\r{self.desc}: {percent:3d}%", end="", flush=True)
            self.last_percent = percent

    def close(self):
        if not self.closed:
            print()
            self.closed = True


def iter_with_progress(iterable, total, desc):
    # 中文：统一进度入口，确保单通路输出
    # English: unified progress entry, single output path
    if not USE_PROGRESS:
        for x in iterable:
            yield x
        return

    if HAS_TQDM:
        for x in tqdm(iterable, total=total, desc=desc, ncols=100, mininterval=PROGRESS_MIN_INTERVAL):
            yield x
        return

    prog = SimpleProgress(total, desc)
    for i, x in enumerate(iterable, start=1):
        prog.update(i)
        yield x
    prog.close()


def make_export_callback(label, total):
    # 中文：导出动画时的进度回调
    # English: progress callback during animation export
    if not USE_PROGRESS:
        return None, None

    if HAS_TQDM:
        bar = tqdm(total=total, desc=label, ncols=100, mininterval=PROGRESS_MIN_INTERVAL)

        def cb(i, n):
            target = i + 1
            delta = target - bar.n
            if delta > 0:
                bar.update(delta)

        return cb, bar

    prog = SimpleProgress(total, label)

    def cb(i, n):
        prog.update(i + 1)

    return cb, prog


# -------------------------
# Math / Physics
# -------------------------
def wrap_to_pi(a):
    # 中文：角度归一到 [-pi, pi]
    # English: wrap angle into [-pi, pi]
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def build_tail_masses(m):
    # 中文：tail[i] = 从 i 到末端的累积质量
    # English: tail[i] = cumulative mass from i-th link to end
    tail = np.zeros(3, dtype=float)
    tail[2] = m[2]
    tail[1] = m[1] + tail[2]
    tail[0] = m[0] + tail[1]
    return tail


def mass_matrix(q, l, tail):
    # 中文：M_ij = S_ij * l_i * l_j * cos(q_i - q_j)
    # English: closed-form mass matrix for serial point-mass links
    M = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            Sij = tail[max(i, j)]
            M[i, j] = Sij * l[i] * l[j] * np.cos(q[i] - q[j])
    return M


def dM_dq(q, l, tail):
    # 中文：dM/dq_k，用于计算 Christoffel 项
    # English: dM/dq_k used by Christoffel-based Coriolis vector
    DM = np.zeros((3, 3, 3), dtype=float)  # k, i, j
    for k in range(3):
        for i in range(3):
            for j in range(3):
                if i == j:
                    DM[k, i, j] = 0.0
                    continue
                Sij = tail[max(i, j)]
                base = Sij * l[i] * l[j] * np.sin(q[i] - q[j])
                if k == i:
                    DM[k, i, j] += -base
                if k == j:
                    DM[k, i, j] += base
    return DM


def coriolis_vector(q, dq, l, tail):
    # 中文：C_i = sum_jk Gamma_ijk dq_j dq_k
    # English: C_i = sum_jk Gamma_ijk dq_j dq_k
    DM = dM_dq(q, l, tail)
    cvec = np.zeros(3, dtype=float)
    for i in range(3):
        s = 0.0
        for j in range(3):
            for k in range(3):
                gamma = 0.5 * (DM[k, i, j] + DM[j, i, k] - DM[i, j, k])
                s += gamma * dq[j] * dq[k]
        cvec[i] = s
    return cvec


def grad_potential(q, l, g, tail):
    # 中文：dV/dq_i = tail_i * g * l_i * sin(q_i)
    # English: gravity generalized force from potential gradient
    return tail * g * l * np.sin(q)


def energy_total(q, dq, l, m, g, tail):
    # 中文：E = T + V
    # English: total mechanical energy E = T + V
    M = mass_matrix(q, l, tail)
    T = 0.5 * float(dq @ M @ dq)
    y1 = -l[0] * np.cos(q[0])
    y2 = y1 - l[1] * np.cos(q[1])
    y3 = y2 - l[2] * np.cos(q[2])
    V = m[0] * g * y1 + m[1] * g * y2 + m[2] * g * y3
    return T + V


def forward_kinematics(theta, l):
    # 中文：从角度得到枢轴+三质点坐标
    # English: convert angles to pivot + 3 mass coordinates
    t1, t2, t3 = theta
    x1 = l[0] * np.sin(t1)
    y1 = -l[0] * np.cos(t1)
    x2 = x1 + l[1] * np.sin(t2)
    y2 = y1 - l[1] * np.cos(t2)
    x3 = x2 + l[2] * np.sin(t3)
    y3 = y2 - l[2] * np.cos(t3)
    return np.array([0.0, x1, x2, x3]), np.array([0.0, y1, y2, y3])


def dynamics(state, l, m, g, damping, tail):
    # 中文：一阶系统 RHS: d/dt[q,dq] = [dq, ddq]
    # English: first-order RHS: d/dt[q,dq] = [dq, ddq]
    q = state[:3]
    dq = state[3:]

    M = mass_matrix(q, l, tail)
    cvec = coriolis_vector(q, dq, l, tail)
    gvec = grad_potential(q, l, g, tail)
    dvec = damping * dq

    rhs = -(cvec + gvec + dvec)
    ddq = np.linalg.solve(M, rhs)

    out = np.zeros(6, dtype=float)
    out[:3] = dq
    out[3:] = ddq
    return out


def step_state(state, dt, l, m, g, damping, tail):
    # 中文：积分步，可选 RK4 / 半隐式欧拉
    # English: integration step with RK4 or semi-implicit Euler
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
    # 中文：检测 NaN/Inf，避免静默发散
    # English: detect NaN/Inf to avoid silent divergence
    if STRICT_NUMERIC_CHECK and not np.all(np.isfinite(arr)):
        raise FloatingPointError(f"Non-finite values detected in {name}")


def build_cache_key(theta0_deg):
    # 中文：轨迹缓存键（仅依赖动力学相关参数）
    # English: trajectory cache key from dynamics-related parameters
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
    # 中文：主仿真函数，先算完整轨迹
    # English: main simulation routine, precompute full trajectory
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
    ts = np.linspace(0.0, n_steps * SIM_DT, n_steps + 1)
    states = np.zeros((n_steps + 1, 6), dtype=float)
    states[0] = state

    sub = max(int(SUBSTEPS), 1)
    dt_sub = SIM_DT / sub

    for i in iter_with_progress(range(n_steps), n_steps, "Simulating"):
        for _ in range(sub):
            state = step_state(state, dt_sub, l, m, G, DAMPING, tail)
            check_finite(state, "state")
        states[i + 1] = state

    if USE_TRAJ_CACHE:
        key = build_cache_key(theta0_deg)
        cache_file = os.path.join(TRAJ_CACHE_DIR, f"traj_{key}.npz")
        np.savez_compressed(cache_file, ts=ts, states=states)
        print(f"[OK] Saved trajectory cache: {cache_file}")

    return ts, states, l, m, tail, cache_hit


# -------------------------
# Rendering / Color
# -------------------------
def angle_to_channel(angle, mode):
    # 中文：单角度映射到 [0,1] 通道
    # English: map one angle to [0,1] channel
    if mode == "sin_smooth":
        return 0.5 + 0.5 * np.sin(angle)
    if mode == "cos_phase":
        return 0.5 + 0.5 * np.cos(angle)
    if mode == "atan_soft":
        return 0.5 + np.arctan(angle) / np.pi
    w = wrap_to_pi(angle)
    return (w + np.pi) / (2.0 * np.pi)


def theta_to_rgb(theta, mode, energy_value=None, e_min=None, e_max=None):
    # 中文：多模式颜色映射
    # English: multi-mode color mapping
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
        if energy_value is None or e_min is None or e_max is None or e_max <= e_min:
            u = 0.5
        else:
            u = float(np.clip((energy_value - e_min) / (e_max - e_min), 0.0, 1.0))
        return tuple(cmap(u)[:3])

    return (
        angle_to_channel(t1, "sin_smooth"),
        angle_to_channel(t2, "sin_smooth"),
        angle_to_channel(t3, "sin_smooth"),
    )


def densify_polyline(x, y, factor):
    # 中文：轨迹折线加密，提高视觉丝滑度
    # English: densify trail polyline for smoother appearance
    if factor <= 1 or len(x) < 2:
        return x, y
    n = len(x)
    src = np.arange(n, dtype=float)
    dst = np.linspace(0.0, n - 1, (n - 1) * factor + 1)
    xd = np.interp(dst, src, x)
    yd = np.interp(dst, src, y)
    return xd, yd


def build_background_colors(q_out, energy_out=None):
    # 中文：生成背景色序列并可做时间平滑
    # English: build background color sequence with optional temporal smoothing
    n = len(q_out)
    bg = np.zeros((n, 3), dtype=float)

    e_min = None
    e_max = None
    if energy_out is not None and len(energy_out) > 0:
        e_min = float(np.min(energy_out))
        e_max = float(np.max(energy_out))

    for i in iter_with_progress(range(n), n, "Color mapping"):
        bg[i] = theta_to_rgb(q_out[i], COLOR_MODE, None if energy_out is None else energy_out[i], e_min, e_max)

    if COLOR_TEMPORAL_SMOOTH and n > 1:
        alpha = float(np.clip(COLOR_SMOOTH_ALPHA, 0.0, 1.0))
        for i in range(1, n):
            bg[i] = alpha * bg[i] + (1.0 - alpha) * bg[i - 1]

    if not COLOR_DYNAMIC:
        bg[:] = bg[0]

    return bg


def sample_for_render(ts_sim, states_sim, l, m, tail):
    # 中文：从仿真时间轴采样到显示时间轴
    # English: sample simulation timeline into display timeline
    t_out = np.arange(0.0, ts_sim[-1] + 1e-12, 1.0 / FPS)
    t_lookup = np.clip(t_out * PLAYBACK_SPEED, ts_sim[0], ts_sim[-1])

    q_sim = states_sim[:, :3]
    dq_sim = states_sim[:, 3:]

    q_out = np.zeros((len(t_out), 3), dtype=float)
    dq_out = np.zeros((len(t_out), 3), dtype=float)
    for j in range(3):
        q_out[:, j] = np.interp(t_lookup, ts_sim, q_sim[:, j])
        dq_out[:, j] = np.interp(t_lookup, ts_sim, dq_sim[:, j])

    n = len(t_out)
    xs = np.zeros((n, 4), dtype=float)
    ys = np.zeros((n, 4), dtype=float)

    for i in iter_with_progress(range(n), n, "Frame geometry"):
        xs[i], ys[i] = forward_kinematics(q_out[i], l)

    energy_out = np.zeros(n, dtype=float)
    for i in range(n):
        energy_out[i] = energy_total(q_out[i], dq_out[i], l, m, G, tail)

    bg = build_background_colors(q_out, energy_out)

    if TRAIL_SOURCE == "display":
        mass_x = np.zeros((n, 3), dtype=float)
        mass_y = np.zeros((n, 3), dtype=float)
        for i in iter_with_progress(range(n), n, "Trail source prep"):
            xx, yy = forward_kinematics(q_out[i], l)
            mass_x[i] = xx[1:]
            mass_y[i] = yy[1:]
        ts_trail = t_lookup.copy()
    else:
        n_sim = len(ts_sim)
        mass_x = np.zeros((n_sim, 3), dtype=float)
        mass_y = np.zeros((n_sim, 3), dtype=float)
        for i in iter_with_progress(range(n_sim), n_sim, "Trail source prep"):
            xx, yy = forward_kinematics(q_sim[i], l)
            mass_x[i] = xx[1:]
            mass_y[i] = yy[1:]
        ts_trail = ts_sim.copy()

    return t_out, t_lookup, q_out, xs, ys, bg, ts_trail, mass_x, mass_y


def save_with_progress(ani, path, writer, label, total_frames):
    # 中文：带进度导出动画文件
    # English: export animation file with progress
    cb, progress_obj = make_export_callback(label, total_frames)
    if cb is None:
        ani.save(path, writer=writer, dpi=OUTPUT_DPI)
        return

    ani.save(path, writer=writer, dpi=OUTPUT_DPI, progress_callback=cb)

    if HAS_TQDM:
        if progress_obj.n < total_frames:
            progress_obj.update(total_frames - progress_obj.n)
        progress_obj.close()
    else:
        progress_obj.close()


def build_side_panel_text(input_angles_deg):
    # 中文：只打印物理数学参数
    # English: print only physical and mathematical parameters
    lines = [
        "Physical / Mathematical Parameters",
        "---------------------------------",
        f"L = [{L1:.5g}, {L2:.5g}, {L3:.5g}]",
        f"M = [{M1:.5g}, {M2:.5g}, {M3:.5g}]",
        f"g = {G:.5g}",
        f"damping = {DAMPING:.5g}",
        "",
        "Initial conditions",
        f"theta0(deg) = [{input_angles_deg[0]:.5g}, {input_angles_deg[1]:.5g}, {input_angles_deg[2]:.5g}]",
        f"omega0(rad/s) = [{INITIAL_OMEGA[0]:.5g}, {INITIAL_OMEGA[1]:.5g}, {INITIAL_OMEGA[2]:.5g}]",
        "",
        "Numerical model",
        "ODE: M(q)ddq + C(q,dq) + G(q) + D(dq) = 0",
        f"Integrator = {INTEGRATOR}",
        f"Substeps = {SUBSTEPS}",
        f"T_total = {T_TOTAL:.5g} s",
        f"dt = {SIM_DT:.5g} s",
    ]
    return "\n".join(lines)


def render_and_export(t_out, t_lookup, q_out, xs, ys, bg, ts_trail, mass_x, mass_y, input_angles_deg):
    # 中文：左侧主图运动，右侧参数面板，避免遮挡
    # English: left motion plot + right parameter panel to avoid overlap
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.46], wspace=0.08)

    ax = fig.add_subplot(gs[0, 0])
    ax_panel = fig.add_subplot(gs[0, 1])
    ax_panel.axis("off")

    rmax = L1 + L2 + L3 + PLOT_MARGIN
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Triple Pendulum | NumPy Model")
    ax.grid(SHOW_GRID, alpha=0.25)

    ax_panel.text(
        0.02, 0.98, build_side_panel_text(input_angles_deg),
        ha="left", va="top", family="monospace", fontsize=10.0
    )

    side_dynamic = ax_panel.text(
        0.02, 0.08, "",
        ha="left", va="bottom", family="monospace", fontsize=10.8
    )

    line, = ax.plot([], [], "o-", lw=2.2, ms=6.0, color=ROD_COLOR, antialiased=True)
    trail1, = ax.plot([], [], "-", lw=1.5, color=(1.0, 0.2, 0.2), alpha=TRAIL_ALPHA,
                      antialiased=True, solid_capstyle="round", solid_joinstyle="round")
    trail2, = ax.plot([], [], "-", lw=1.5, color=(0.2, 1.0, 0.2), alpha=TRAIL_ALPHA,
                      antialiased=True, solid_capstyle="round", solid_joinstyle="round")
    trail3, = ax.plot([], [], "-", lw=1.5, color=(0.2, 0.5, 1.0), alpha=TRAIL_ALPHA,
                      antialiased=True, solid_capstyle="round", solid_joinstyle="round")

    def init():
        line.set_data([], [])
        trail1.set_data([], [])
        trail2.set_data([], [])
        trail3.set_data([], [])
        side_dynamic.set_text("")
        return line, trail1, trail2, trail3, side_dynamic

    def get_trail_segment(mass_idx, t_cur, t_span):
        # 中文：按时间窗口切轨迹，再做限点和加密
        # English: slice by time-window, cap points, then densify
        t0 = max(ts_trail[0], t_cur - t_span)
        i0 = np.searchsorted(ts_trail, t0, side="left")
        i1 = np.searchsorted(ts_trail, t_cur, side="right")
        if i1 <= i0:
            i1 = min(i0 + 1, len(ts_trail))

        x = mass_x[i0:i1, mass_idx]
        y = mass_y[i0:i1, mass_idx]

        if len(x) > TRAIL_MAX_POINTS:
            stride = int(np.ceil(len(x) / TRAIL_MAX_POINTS))
            x = x[::stride]
            y = y[::stride]

        x, y = densify_polyline(x, y, TRAIL_SMOOTH_FACTOR)
        return x, y

    def update(i):
        ax.set_facecolor(bg[i])
        line.set_data(xs[i], ys[i])

        if SHOW_TRAIL:
            trail_seconds = (TRAIL_LENGTH / FPS) * max(PLAYBACK_SPEED, 1e-12)
            t_cur = t_lookup[i]

            x1, y1 = get_trail_segment(0, t_cur, trail_seconds)
            x2, y2 = get_trail_segment(1, t_cur, trail_seconds)
            x3, y3 = get_trail_segment(2, t_cur, trail_seconds)

            trail1.set_data(x1, y1)
            trail2.set_data(x2, y2)
            trail3.set_data(x3, y3)
        else:
            trail1.set_data([], [])
            trail2.set_data([], [])
            trail3.set_data([], [])

        if SHOW_INFO:
            deg = np.rad2deg(q_out[i])
            side_dynamic.set_text(
                "Runtime state\n"
                "-------------\n"
                f"t = {t_out[i]:.4f} s\n"
                f"theta1 = {deg[0]:.3f} deg\n"
                f"theta2 = {deg[1]:.3f} deg\n"
                f"theta3 = {deg[2]:.3f} deg\n"
            )
        else:
            side_dynamic.set_text("")

        return line, trail1, trail2, trail3, side_dynamic

    ani = FuncAnimation(
        fig,
        update,
        frames=len(t_out),
        init_func=init,
        interval=1000.0 / FPS,
        blit=False,
    )

    total_frames = len(t_out)

    if EXPORT_GIF:
        save_with_progress(ani, GIF_NAME, PillowWriter(fps=FPS), "Export GIF", total_frames)
        print(f"[OK] GIF exported: {GIF_NAME}")

    if EXPORT_MP4:
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
    # 中文：读取初始角度（度）
    # English: read initial angles (degrees)
    if not USE_CONSOLE_INPUT:
        return DEFAULT_ANGLES_DEG

    s = input("Enter theta1 theta2 theta3 in degrees (example: 10 20 30): ").strip()
    parts = s.split()
    if len(parts) != 3:
        raise ValueError("Please input exactly 3 numbers.")
    return [float(parts[0]), float(parts[1]), float(parts[2])]


def main():
    # 中文：主流程：输入 -> 仿真 -> 渲染预处理 -> 导出
    # English: main flow: input -> simulate -> render prep -> export
    input_angles_deg = read_initial_angles()

    print("[INFO] Precomputing NumPy dynamics...")
    ts_sim, states_sim, l, m, tail, cache_hit = simulate(input_angles_deg)
    print(f"[INFO] Simulation done. Steps = {len(ts_sim)} | Cache hit = {cache_hit}")

    print("[INFO] Preparing render buffers...")
    t_out, t_lookup, q_out, xs, ys, bg, ts_trail, mass_x, mass_y = sample_for_render(ts_sim, states_sim, l, m, tail)
    print(f"[INFO] Render frames prepared. Frames = {len(t_out)} | Speed = {PLAYBACK_SPEED:.2f}x")

    if SAVE_NPZ:
        np.savez(
            NPZ_NAME,
            ts_sim=ts_sim,
            states_sim=states_sim,
            t_out=t_out,
            t_lookup=t_lookup,
            q_out=q_out,
            xs=xs,
            ys=ys,
            bg=bg,
            ts_trail=ts_trail,
            mass_x=mass_x,
            mass_y=mass_y,
        )
        print(f"[OK] Precomputed data saved: {NPZ_NAME}")

    render_and_export(t_out, t_lookup, q_out, xs, ys, bg, ts_trail, mass_x, mass_y, input_angles_deg)


if __name__ == "__main__":
    main()