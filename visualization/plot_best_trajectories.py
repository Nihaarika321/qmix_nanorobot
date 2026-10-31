import numpy as np
import matplotlib.pyplot as plt
import os

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

# ============================================================
# ENVIRONMENT TARGETS
# ============================================================

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

    print("\nLoaded:", path)
    print("Raw shape:", raw.shape)
    print("Raw dtype:", raw.dtype)

    # --------------------------------------------------------
    # IQL format:
    # object array of dictionaries
    # --------------------------------------------------------

    if isinstance(raw, np.ndarray) and raw.dtype == object:

        first = raw[0]

        if isinstance(first, dict):

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

        # ----------------------------------------------------
        # VDN/QMIX may also be object arrays containing arrays
        # ----------------------------------------------------

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

    # --------------------------------------------------------
    # Normal numeric array
    # --------------------------------------------------------

    return np.asarray(
        raw,
        dtype=float
    )


# ============================================================
# LOAD ALL THREE
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

    print(
        f"{algorithm} position shape:",
        trajectory.shape
    )

    trajectories[algorithm] = trajectory


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(19, 7)
)


# ============================================================
# ROBOT SETTINGS
# ============================================================

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
# PLOT
# ============================================================

for ax, (algorithm, trajectory) in zip(
    axes,
    trajectories.items()
):

    num_steps = trajectory.shape[0]

    # --------------------------------------------------------
    # Actual starting positions
    # --------------------------------------------------------

    start_positions = trajectory[0]

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
    # Plot each robot
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

        # ----------------------------------------------------
        # Trajectory
        # ----------------------------------------------------

        ax.plot(
            cols,
            rows,
            linewidth=2,
            marker=robot_markers[robot],
            markersize=3,
            label=robot_labels[robot]
        )

        # ----------------------------------------------------
        # ACTUAL start position
        # ----------------------------------------------------

        ax.scatter(
            cols[0],
            rows[0],
            s=180,
            marker="*",
            edgecolors="black",
            linewidths=1.2,
            zorder=10
        )

        # ----------------------------------------------------
        # End position
        # ----------------------------------------------------

        ax.scatter(
            cols[-1],
            rows[-1],
            s=120,
            marker="X",
            edgecolors="black",
            linewidths=1.2,
            zorder=10
        )

        # ----------------------------------------------------
        # Start label
        # ----------------------------------------------------

        ax.text(
            cols[0] + 0.25,
            rows[0] + 0.25,
            f"R{robot + 1} start",
            fontsize=8
        )

    # --------------------------------------------------------
    # Plot targets
    # --------------------------------------------------------

    for i, (row, col) in enumerate(TARGETS):

        ax.scatter(
            col,
            row,
            s=220,
            marker="D",
            edgecolors="black",
            linewidths=1.5,
            zorder=11
        )

        ax.text(
            col + 0.25,
            row + 0.25,
            f"T{i + 1}",
            fontsize=10,
            fontweight="bold"
        )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    ax.set_title(
        f"{algorithm} Best Episode\n"
        f"{num_steps} Steps",
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


# ============================================================
# LEGEND
# ============================================================

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, 0.01)
)


# ============================================================
# TITLE
# ============================================================

fig.suptitle(
    "Best Nanorobot Trajectories: IQL vs VDN vs QMIX",
    fontsize=18,
    fontweight="bold"
)


# ============================================================
# SAVE
# ============================================================

plt.tight_layout(
    rect=[
        0,
        0.07,
        1,
        0.94
    ]
)

output_path = os.path.join(
    OUTPUT_DIR,
    "all_algorithms_best_trajectories.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# PRINT INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TRAJECTORY VISUALIZATION")
print("=" * 70)

for algorithm, trajectory in trajectories.items():

    print(
        f"\n{algorithm}:"
    )

    print(
        f"  Steps: {trajectory.shape[0]}"
    )

    print(
        f"  Robot 1 start: {tuple(trajectory[0, 0].astype(int))}"
    )

    print(
        f"  Robot 2 start: {tuple(trajectory[0, 1].astype(int))}"
    )

    print(
        f"  Robot 3 start: {tuple(trajectory[0, 2].astype(int))}"
    )

print("\nTargets:")

for i, target in enumerate(TARGETS):

    print(
        f"  T{i + 1}: {target}"
    )

print(
    f"\nSaved to:\n{output_path}"
)