import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# SETTINGS
# ============================================================

RESULTS_DIR = "results"
OUTPUT_DIR = "results/trajectory_figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SENSING_RADIUS = 4

# Target locations from the environment
TARGETS = [
    (13, 6),   # T1
    (4, 2),    # T2
    (9, 8)     # T3
]

ALGORITHMS = {
    "IQL": "iql_best_trajectory.npy",
    "VDN": "vdn_best_trajectory.npy",
    "QMIX": "qmix_best_trajectory.npy"
}


# ============================================================
# LOAD TRAJECTORY
# ============================================================

def load_positions(filepath, algorithm):

    raw = np.load(filepath, allow_pickle=True)

    positions = []

    for item in raw:

        # IQL trajectory contains dictionaries
        if algorithm == "IQL":
            pos = np.array(item["positions"])

        # VDN / QMIX directly contain positions
        else:
            pos = np.array(item)

        positions.append(pos)

    return np.array(positions)


# ============================================================
# CHECK TARGET COVERAGE
# ============================================================

def calculate_coverage(positions):

    coverage = np.zeros((len(TARGETS), len(positions)), dtype=bool)

    for t, robot_positions in enumerate(positions):

        for target_id, target in enumerate(TARGETS):

            for robot_pos in robot_positions:

                distance = (
                    abs(robot_pos[0] - target[0])
                    + abs(robot_pos[1] - target[1])
                )

                if distance <= SENSING_RADIUS:
                    coverage[target_id, t] = True
                    break

    return coverage


# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 8),
    sharex=False
)

for ax, (algorithm, filename) in zip(axes, ALGORITHMS.items()):

    filepath = os.path.join(RESULTS_DIR, filename)

    positions = load_positions(filepath, algorithm)

    coverage = calculate_coverage(positions)

    steps = np.arange(len(positions))

    # Plot each target
    for target_id in range(len(TARGETS)):

        y = np.full(len(steps), target_id + 1)

        # Only show points when target is sensed
        sensed_steps = steps[coverage[target_id]]

        ax.scatter(
            sensed_steps,
            y[coverage[target_id]],
            marker="s",
            s=35,
            label=f"T{target_id + 1}"
        )

    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["T1", "T2", "T3"])

    ax.set_ylabel("Target")

    ax.set_title(
        f"{algorithm} – Target Sensing Timeline"
    )

    ax.grid(True, alpha=0.3)

    ax.legend(
        loc="upper right",
        ncol=3
    )

axes[-1].set_xlabel("Timestep")

plt.tight_layout()

output_file = os.path.join(
    OUTPUT_DIR,
    "target_coverage_timeline.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nPlot saved to:")
print(output_file)