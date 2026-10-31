import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Circle

# ============================================================
# SETTINGS
# ============================================================

RESULTS_DIR = "results"
OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "trajectory_figures"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

GRID_SIZE = 20
SENSING_RADIUS = 4

TARGETS = [
    (13, 6),
    (4, 2),
    (9, 8)
]

ALGORITHMS = {
    "IQL": "iql_best_trajectory.npy",
    "VDN": "vdn_best_trajectory.npy",
    "QMIX": "qmix_best_trajectory.npy"
}


# ============================================================
# LOAD TRAJECTORY
# ============================================================

def load_trajectory(path):

    raw = np.load(
        path,
        allow_pickle=True
    )

    # IQL format:
    # array of dictionaries
    if raw.dtype == object and isinstance(raw[0], dict):

        positions = []

        for item in raw:

            positions.append(
                np.asarray(
                    item["positions"],
                    dtype=float
                )
            )

        return np.asarray(
            positions,
            dtype=float
        )

    # VDN / QMIX:
    # object array containing position arrays
    positions = []

    for item in raw:

        positions.append(
            np.asarray(
                item,
                dtype=float
            )
        )

    return np.asarray(
        positions,
        dtype=float
    )


# ============================================================
# MANHATTAN DISTANCE
# ============================================================

def manhattan_distance(position, target):

    return (
        abs(position[0] - target[0])
        +
        abs(position[1] - target[1])
    )


# ============================================================
# LOAD TRAJECTORIES
# ============================================================

trajectories = {}

for algorithm, filename in ALGORITHMS.items():

    path = os.path.join(
        RESULTS_DIR,
        filename
    )

    if not os.path.exists(path):

        print(
            f"WARNING: {path} not found"
        )

        continue

    trajectory = load_trajectory(path)

    trajectories[algorithm] = trajectory

    print(
        f"{algorithm}: "
        f"{trajectory.shape}"
    )


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(20, 7)
)


robot_markers = [
    "o",
    "s",
    "^"
]

robot_labels = [
    "Robot 1",
    "Robot 2",
    "Robot 3"
]


# ============================================================
# PLOT EACH ALGORITHM
# ============================================================

for ax, (algorithm, trajectory) in zip(
    axes,
    trajectories.items()
):

    steps = trajectory.shape[0]

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.set_xlim(
        -0.5,
        GRID_SIZE - 0.5
    )

    ax.set_ylim(
        -0.5,
        GRID_SIZE - 0.5
    )

    ax.set_xticks(
        range(GRID_SIZE)
    )

    ax.set_yticks(
        range(GRID_SIZE)
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.5,
        alpha=0.35
    )

    # --------------------------------------------------------
    # SENSING RADIUS
    # --------------------------------------------------------

    for i, (row, col) in enumerate(TARGETS):

        circle = Circle(
            (col, row),
            SENSING_RADIUS,
            fill=False,
            linestyle="--",
            linewidth=2,
            alpha=0.7
        )

        ax.add_patch(circle)

        # Target marker
        ax.scatter(
            col,
            row,
            s=220,
            marker="D",
            edgecolors="black",
            linewidths=1.5,
            zorder=10
        )

        # Target label
        ax.text(
            col + 0.25,
            row + 0.3,
            f"T{i + 1}",
            fontsize=11,
            fontweight="bold",
            zorder=12
        )

        # Radius label
        ax.text(
            col + 2.3,
            row + 2.3,
            "R=4",
            fontsize=8,
            zorder=12
        )

    # --------------------------------------------------------
    # ROBOT TRAJECTORIES
    # --------------------------------------------------------

    for robot in range(3):

        rows = trajectory[
            :,
            robot,
            0
        ]

        cols = trajectory[
            :,
            robot,
            1
        ]

        # Path
        ax.plot(
            cols,
            rows,
            linewidth=2,
            marker=robot_markers[robot],
            markersize=3,
            label=robot_labels[robot]
        )

        # Start
        ax.scatter(
            cols[0],
            rows[0],
            s=180,
            marker="*",
            edgecolors="black",
            linewidths=1.2,
            zorder=15
        )

        # End
        ax.scatter(
            cols[-1],
            rows[-1],
            s=120,
            marker="X",
            edgecolors="black",
            linewidths=1.2,
            zorder=15
        )

        # Start label
        ax.text(
            cols[0] + 0.25,
            rows[0] + 0.25,
            f"R{robot + 1} start",
            fontsize=8
        )

    # --------------------------------------------------------
    # TARGET PROXIMITY ANALYSIS
    # --------------------------------------------------------

    proximity_text = []

    for target_id, target in enumerate(TARGETS):

        minimum_distance = float("inf")
        closest_robot = None
        closest_step = None

        for step in range(steps):

            for robot in range(3):

                position = trajectory[
                    step,
                    robot
                ]

                distance = manhattan_distance(
                    position,
                    target
                )

                if distance < minimum_distance:

                    minimum_distance = distance
                    closest_robot = robot + 1
                    closest_step = step

        if minimum_distance <= SENSING_RADIUS:

            status = "SENSED"

        else:

            status = "OUTSIDE"

        proximity_text.append(
            f"T{target_id + 1}: "
            f"{status}, "
            f"d={minimum_distance:.0f}, "
            f"R{closest_robot}"
        )

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    ax.set_title(
        f"{algorithm} Best Episode\n"
        f"{steps} Steps",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Grid Column"
    )

    ax.set_ylabel(
        "Grid Row"
    )

    ax.set_aspect(
        "equal"
    )

    # --------------------------------------------------------
    # PROXIMITY INFORMATION
    # --------------------------------------------------------

    ax.text(
        0.02,
        0.02,
        "\n".join(proximity_text),
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round",
            alpha=0.8
        )
    )


# ============================================================
# COMMON LEGEND
# ============================================================

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, 0.01),
    fontsize=10
)


# ============================================================
# MAIN TITLE
# ============================================================

fig.suptitle(
    "Nanorobot Trajectories with Target Sensing Radius",
    fontsize=18,
    fontweight="bold"
)


# ============================================================
# EXPLANATION
# ============================================================

fig.text(
    0.5,
    0.045,
    "Dashed circles indicate the sensing radius "
    f"(R = {SENSING_RADIUS} grid cells) around each inflammation target",
    ha="center",
    fontsize=10
)


# ============================================================
# SAVE
# ============================================================

plt.tight_layout(
    rect=[
        0,
        0.08,
        1,
        0.93
    ]
)

output_path = os.path.join(
    OUTPUT_DIR,
    "sensing_radius_trajectories.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SENSING-RADIUS TRAJECTORY VISUALIZATION")
print("=" * 70)

print(
    f"\nSensing radius = {SENSING_RADIUS} grid cells"
)

for algorithm, trajectory in trajectories.items():

    print(
        f"\n{algorithm}:"
    )

    for target_id, target in enumerate(TARGETS):

        minimum_distance = float("inf")
        closest_robot = None
        closest_step = None

        for step in range(len(trajectory)):

            for robot in range(3):

                distance = manhattan_distance(
                    trajectory[step, robot],
                    target
                )

                if distance < minimum_distance:

                    minimum_distance = distance
                    closest_robot = robot + 1
                    closest_step = step

        if minimum_distance <= SENSING_RADIUS:

            status = "WITHIN SENSING RANGE"

        else:

            status = "OUTSIDE SENSING RANGE"

        print(
            f"  T{target_id + 1} {target}: "
            f"{status}, "
            f"minimum distance = {minimum_distance:.0f}, "
            f"closest robot = R{closest_robot}, "
            f"step = {closest_step}"
        )

print(
    f"\nSaved to:\n{output_path}"
)

print("=" * 70)