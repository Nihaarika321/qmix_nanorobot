# ============================================================
# QMIX BIOLOGICALLY INSPIRED VISUALIZATION
# ============================================================
#
# Uses the SAME best QMIX episode saved by evaluate_qmix.py:
#   qmix_best_trajectory.npy
#   qmix_best_target_positions.npy
#   qmix_best_actions.npy
#   qmix_best_treated_history.npy
#   qmix_best_discovered_history.npy
#   qmix_best_energy_history.npy
#   qmix_best_coverage_history.npy
#
# Creates:
#   1. RL grid trajectory PNG
#   2. Biological-inspired trajectory PNG
#   3. Biological-inspired animated GIF
#
# NOTE:
# This is a BIOLOGICALLY INSPIRED visualization, not an
# anatomically accurate model of a real joint.
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Ellipse, Rectangle

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

TRAJ_FILE = os.path.join(RESULTS_DIR, "qmix_best_trajectory.npy")
TARGET_FILE = os.path.join(RESULTS_DIR, "qmix_best_target_positions.npy")
ACTIONS_FILE = os.path.join(RESULTS_DIR, "qmix_best_actions.npy")
TREATED_FILE = os.path.join(RESULTS_DIR, "qmix_best_treated_history.npy")
DISCOVERED_FILE = os.path.join(RESULTS_DIR, "qmix_best_discovered_history.npy")
ENERGY_FILE = os.path.join(RESULTS_DIR, "qmix_best_energy_history.npy")
COVERAGE_FILE = os.path.join(RESULTS_DIR, "qmix_best_coverage_history.npy")

OUT_DIR = os.path.join(RESULTS_DIR, "qmix_biological_visualization")
os.makedirs(OUT_DIR, exist_ok=True)

GRID_SIZE = 20
SENSOR_RADIUS = 4

# Action mapping from joint_env.py
STAY = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
TREAT = 5


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

trajectory = np.load(TRAJ_FILE)
targets = np.load(TARGET_FILE)

actions = np.load(ACTIONS_FILE)
treated_history = np.load(TREATED_FILE)
discovered_history = np.load(DISCOVERED_FILE)
energy_history = np.load(ENERGY_FILE)
coverage_history = np.load(COVERAGE_FILE)

print("=" * 70)
print("QMIX BIOLOGICAL VISUALIZATION")
print("=" * 70)

print("Trajectory shape:", trajectory.shape)
print("Target positions:", targets.tolist())
print("Actions shape:", actions.shape)
print("Treated history:", treated_history.shape)
print("Discovered history:", discovered_history.shape)
print("Energy history:", energy_history.shape)

# trajectory has initial position + one position per environment step
n_frames = len(trajectory)
n_agents = trajectory.shape[1]

# actions normally has one fewer frame than trajectory
n_action_steps = len(actions)


# ------------------------------------------------------------
# TARGET IDENTIFICATION
# ------------------------------------------------------------

target_names = ["T1", "T2", "T3"]


# ------------------------------------------------------------
# HELPER: WHICH TARGETS ARE TREATED?
#
# We only saved the number of treated targets, not the exact
# target ID history. For the visualization we infer the first
# treatment event associated with a nearby target.
# ------------------------------------------------------------

target_treated_frame = np.full(len(targets), -1, dtype=int)

previous_treated = int(treated_history[0])

for t in range(1, len(treated_history)):
    current_treated = int(treated_history[t])

    if current_treated > previous_treated:
        # Find untreated-looking target(s) near any robot at this
        # frame. A treatment is possible at Manhattan distance <= 1.
        candidate_targets = []

        for target_id, (tx, ty) in enumerate(targets):
            if target_treated_frame[target_id] != -1:
                continue

            for agent_id in range(n_agents):
                x, y = trajectory[min(t, n_frames - 1), agent_id]
                distance = abs(int(x) - int(tx)) + abs(int(y) - int(ty))

                if distance <= 1:
                    candidate_targets.append(target_id)
                    break

        for target_id in candidate_targets[: current_treated - previous_treated]:
            target_treated_frame[target_id] = t

    previous_treated = current_treated


# ------------------------------------------------------------
# FIGURE 1: RL GRID TRAJECTORY
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 9))

ax.set_xlim(-0.5, GRID_SIZE - 0.5)
ax.set_ylim(-0.5, GRID_SIZE - 0.5)
ax.set_aspect("equal")

ax.set_xticks(range(GRID_SIZE))
ax.set_yticks(range(GRID_SIZE))
ax.grid(True, alpha=0.25)

# Plot target locations
for i, (tx, ty) in enumerate(targets):
    ax.scatter(
        tx, ty,
        s=220,
        marker="*",
        edgecolors="black",
        linewidths=1.2,
        zorder=6
    )
    ax.text(
        tx + 0.35,
        ty + 0.35,
        target_names[i],
        fontsize=11,
        weight="bold",
        zorder=7
    )

# Plot trajectories
for agent_id in range(n_agents):
    xs = trajectory[:, agent_id, 0]
    ys = trajectory[:, agent_id, 1]

    ax.plot(
        xs,
        ys,
        linewidth=2,
        label=f"Nanorobot R{agent_id + 1}"
    )

    ax.scatter(
        xs[0],
        ys[0],
        s=80,
        marker="o",
        edgecolors="black",
        zorder=5
    )

    ax.scatter(
        xs[-1],
        ys[-1],
        s=100,
        marker="X",
        edgecolors="black",
        zorder=5
    )

# Sensing circles at final positions
for agent_id in range(n_agents):
    x, y = trajectory[-1, agent_id]

    circle = Circle(
        (x, y),
        SENSOR_RADIUS,
        fill=False,
        linestyle="--",
        linewidth=1.2,
        alpha=0.6
    )

    ax.add_patch(circle)

ax.set_title(
    "QMIX Best Episode — Coordinated Nanorobot Search",
    fontsize=15,
    weight="bold"
)

ax.set_xlabel("Grid X")
ax.set_ylabel("Grid Y")
ax.legend(loc="upper right")

grid_path = os.path.join(
    OUT_DIR,
    "qmix_rl_grid_trajectory.png"
)

fig.savefig(
    grid_path,
    dpi=220,
    bbox_inches="tight"
)

plt.close(fig)


# ------------------------------------------------------------
# BIOLOGICAL-INSPIRED STATIC VISUALIZATION
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 9))

ax.set_xlim(0, GRID_SIZE)
ax.set_ylim(0, GRID_SIZE)
ax.set_aspect("equal")
ax.axis("off")

# ------------------------------------------------------------
# Tissue background
# ------------------------------------------------------------

background = Rectangle(
    (0, 0),
    GRID_SIZE,
    GRID_SIZE,
    alpha=0.18
)

ax.add_patch(background)

# ------------------------------------------------------------
# Blood-vessel-like structures
# ------------------------------------------------------------

vessel_paths = [
    np.array([
        [0, 4], [3, 5], [6, 4], [9, 5],
        [12, 7], [15, 6], [20, 8]
    ]),

    np.array([
        [2, 20], [4, 16], [7, 14], [10, 13],
        [13, 14], [17, 12], [20, 13]
    ]),

    np.array([
        [0, 12], [4, 11], [8, 10], [12, 11],
        [16, 10], [20, 11]
    ])
]

for vessel in vessel_paths:
    ax.plot(
        vessel[:, 0],
        vessel[:, 1],
        linewidth=8,
        alpha=0.18,
        solid_capstyle="round"
    )

    ax.plot(
        vessel[:, 0],
        vessel[:, 1],
        linewidth=2.5,
        alpha=0.65,
        solid_capstyle="round"
    )


# ------------------------------------------------------------
# Inflammation target regions
# ------------------------------------------------------------

for i, (tx, ty) in enumerate(targets):

    inflammation = Ellipse(
        (tx + 0.5, ty + 0.5),
        width=2.0,
        height=1.5,
        alpha=0.28
    )

    ax.add_patch(inflammation)

    ax.text(
        tx + 0.5,
        ty + 1.35,
        f"{target_names[i]}: inflammation",
        ha="center",
        fontsize=9,
        weight="bold"
    )


# ------------------------------------------------------------
# Robot trajectories
# ------------------------------------------------------------

for agent_id in range(n_agents):

    xs = trajectory[:, agent_id, 0] + 0.5
    ys = trajectory[:, agent_id, 1] + 0.5

    ax.plot(
        xs,
        ys,
        linewidth=2.4,
        alpha=0.8,
        label=f"Nanorobot R{agent_id + 1}"
    )

    ax.scatter(
        xs[0],
        ys[0],
        s=70,
        marker="o",
        edgecolors="black",
        zorder=7
    )

    ax.scatter(
        xs[-1],
        ys[-1],
        s=100,
        marker="X",
        edgecolors="black",
        zorder=7
    )


# ------------------------------------------------------------
# Final sensing fields
# ------------------------------------------------------------

for agent_id in range(n_agents):

    x, y = trajectory[-1, agent_id]

    sensing = Circle(
        (x + 0.5, y + 0.5),
        SENSOR_RADIUS,
        fill=False,
        linestyle="--",
        linewidth=1.4,
        alpha=0.45
    )

    ax.add_patch(sensing)


# ------------------------------------------------------------
# Targets
# ------------------------------------------------------------

for i, (tx, ty) in enumerate(targets):

    ax.scatter(
        tx + 0.5,
        ty + 0.5,
        s=280,
        marker="*",
        edgecolors="black",
        linewidths=1.2,
        zorder=8
    )

    ax.text(
        tx + 0.5,
        ty + 0.05,
        target_names[i],
        ha="center",
        fontsize=11,
        weight="bold",
        zorder=9
    )


# ------------------------------------------------------------
# Treatment markers
# ------------------------------------------------------------

for target_id, frame in enumerate(target_treated_frame):

    if frame >= 0:

        tx, ty = targets[target_id]

        ax.scatter(
            tx + 0.5,
            ty + 0.5,
            s=450,
            marker="o",
            facecolors="none",
            edgecolors="black",
            linewidths=2.2,
            zorder=10
        )

        ax.text(
            tx + 0.5,
            ty - 0.35,
            "TREATED",
            ha="center",
            fontsize=8,
            weight="bold"
        )


ax.set_title(
    "QMIX Nanorobot Coordination — Biologically Inspired Joint Microenvironment",
    fontsize=15,
    weight="bold",
    pad=15
)

ax.text(
    0.5,
    -0.03,
    "Search → Detect → Approach → Treat",
    transform=ax.transAxes,
    ha="center",
    fontsize=12,
    weight="bold"
)

ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.01, 1.0)
)

biological_path = os.path.join(
    OUT_DIR,
    "qmix_biological_static.png"
)

fig.savefig(
    biological_path,
    dpi=220,
    bbox_inches="tight"
)

plt.close(fig)


# ------------------------------------------------------------
# ANIMATION
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 9))

ax.set_xlim(0, GRID_SIZE)
ax.set_ylim(0, GRID_SIZE)
ax.set_aspect("equal")
ax.axis("off")


# ------------------------------------------------------------
# Draw static biological background
# ------------------------------------------------------------

background = Rectangle(
    (0, 0),
    GRID_SIZE,
    GRID_SIZE,
    alpha=0.18
)

ax.add_patch(background)

for vessel in vessel_paths:
    ax.plot(
        vessel[:, 0],
        vessel[:, 1],
        linewidth=8,
        alpha=0.16,
        solid_capstyle="round"
    )

    ax.plot(
        vessel[:, 0],
        vessel[:, 1],
        linewidth=2.5,
        alpha=0.6,
        solid_capstyle="round"
    )


# ------------------------------------------------------------
# Target inflammation patches
# ------------------------------------------------------------

target_patches = []

for i, (tx, ty) in enumerate(targets):

    patch = Ellipse(
        (tx + 0.5, ty + 0.5),
        width=2.2,
        height=1.7,
        alpha=0.28
    )

    ax.add_patch(patch)
    target_patches.append(patch)

    ax.text(
        tx + 0.5,
        ty + 1.35,
        target_names[i],
        ha="center",
        fontsize=10,
        weight="bold"
    )


# ------------------------------------------------------------
# Robot trails
# ------------------------------------------------------------

trail_lines = []

for agent_id in range(n_agents):

    line, = ax.plot(
        [],
        [],
        linewidth=2.2,
        alpha=0.8,
        label=f"Nanorobot R{agent_id + 1}"
    )

    trail_lines.append(line)


# ------------------------------------------------------------
# Robot markers
# ------------------------------------------------------------

robot_markers = []

for agent_id in range(n_agents):

    marker, = ax.plot(
        [],
        [],
        marker="o",
        markersize=9,
        markeredgecolor="black",
        linestyle=""
    )

    robot_markers.append(marker)


# ------------------------------------------------------------
# Sensing circles
# ------------------------------------------------------------

sensing_circles = []

for agent_id in range(n_agents):

    circle = Circle(
        (0, 0),
        SENSOR_RADIUS,
        fill=False,
        linestyle="--",
        linewidth=1.2,
        alpha=0.30
    )

    ax.add_patch(circle)
    sensing_circles.append(circle)


# ------------------------------------------------------------
# Target markers
# ------------------------------------------------------------

target_markers = []

for i, (tx, ty) in enumerate(targets):

    marker, = ax.plot(
        [tx + 0.5],
        [ty + 0.5],
        marker="*",
        markersize=18,
        markeredgecolor="black",
        linestyle=""
    )

    target_markers.append(marker)


# ------------------------------------------------------------
# Text
# ------------------------------------------------------------

title_text = ax.text(
    0.5,
    1.02,
    "",
    transform=ax.transAxes,
    ha="center",
    fontsize=15,
    weight="bold"
)

status_text = ax.text(
    0.5,
    -0.025,
    "",
    transform=ax.transAxes,
    ha="center",
    fontsize=11,
    weight="bold"
)


# ------------------------------------------------------------
# Animation update
# ------------------------------------------------------------

def update(frame):

    # --------------------------------------------
    # Robot positions
    # --------------------------------------------

    for agent_id in range(n_agents):

        xs = trajectory[:frame + 1, agent_id, 0] + 0.5
        ys = trajectory[:frame + 1, agent_id, 1] + 0.5

        trail_lines[agent_id].set_data(xs, ys)

        x = xs[-1]
        y = ys[-1]

        robot_markers[agent_id].set_data([x], [y])

        sensing_circles[agent_id].center = (x, y)


    # --------------------------------------------
    # Target treatment visualization
    # --------------------------------------------

    current_treated = int(
        treated_history[
            min(frame, len(treated_history) - 1)
        ]
    )

    for target_id in range(len(targets)):

        tx, ty = targets[target_id]

        # If we inferred a treatment frame
        if (
            target_treated_frame[target_id] >= 0
            and frame >= target_treated_frame[target_id]
        ):

            target_patches[target_id].set_alpha(0.08)

            target_markers[target_id].set_markersize(13)

        else:

            target_patches[target_id].set_alpha(0.28)

            target_markers[target_id].set_markersize(18)


    # --------------------------------------------
    # Status
    # --------------------------------------------

    discovered = int(
        discovered_history[
            min(frame, len(discovered_history) - 1)
        ]
    )

    energy = float(
        np.mean(
            energy_history[
                min(frame, len(energy_history) - 1)
            ]
        )
    )

    coverage = float(
        coverage_history[
            min(frame, len(coverage_history) - 1)
        ]
    )

    if current_treated > 0:
        phase = "TREATMENT / TARGET CLEARANCE"
    elif discovered > 0:
        phase = "DETECT → APPROACH"
    else:
        phase = "SEARCH / EXPLORATION"

    title_text.set_text(
        f"QMIX Nanorobot Coordination | Step {frame}/{n_frames - 1}"
    )

    status_text.set_text(
        f"{phase}   |   "
        f"Detected: {discovered}/3   |   "
        f"Treated: {current_treated}/3   |   "
        f"Mean Energy: {energy:.1f}   |   "
        f"Coverage: {coverage:.3f}"
    )

    return (
        trail_lines
        + robot_markers
        + sensing_circles
        + target_markers
        + target_patches
        + [title_text, status_text]
    )


# ------------------------------------------------------------
# Create animation
# ------------------------------------------------------------

animation = FuncAnimation(
    fig,
    update,
    frames=n_frames,
    interval=180,
    blit=False,
    repeat=False
)

gif_path = os.path.join(
    OUT_DIR,
    "qmix_biological_animation.gif"
)

animation.save(
    gif_path,
    writer=PillowWriter(fps=6)
)

plt.close(fig)


# ------------------------------------------------------------
# SAVE A SIMPLE METRICS FIGURE
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))

steps = np.arange(len(treated_history))

ax.plot(
    steps,
    treated_history,
    linewidth=2,
    label="Targets treated"
)

ax.plot(
    steps,
    discovered_history,
    linewidth=2,
    label="Targets discovered"
)

ax.set_xlabel("Environment step")
ax.set_ylabel("Number of targets")
ax.set_ylim(-0.1, 3.2)
ax.grid(True, alpha=0.25)
ax.legend()

ax.set_title(
    "QMIX Target Discovery and Treatment Timeline",
    weight="bold"
)

metrics_path = os.path.join(
    OUT_DIR,
    "qmix_target_treatment_timeline.png"
)

fig.savefig(
    metrics_path,
    dpi=220,
    bbox_inches="tight"
)

plt.close(fig)


# ------------------------------------------------------------
# FINAL OUTPUT
# ------------------------------------------------------------

print()
print("=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

print()
print("RL grid figure:")
print(grid_path)

print()
print("Biological-inspired static figure:")
print(biological_path)

print()
print("Biological animation:")
print(gif_path)

print()
print("Target discovery/treatment timeline:")
print(metrics_path)

print()
print("Actual targets used:")
print(targets.tolist())

print()
print("Best episode:")
print(f"Frames: {n_frames}")
print(f"Agents: {n_agents}")
print(f"Targets: {len(targets)}")

print()
print("=" * 70)