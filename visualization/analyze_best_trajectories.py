import numpy as np
import os

RESULTS_DIR = "results"

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

    # IQL: dictionaries
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
            positions
        )

    # VDN / QMIX
    positions = []

    for item in raw:

        positions.append(
            np.asarray(
                item,
                dtype=float
            )
        )

    return np.asarray(
        positions
    )


# ============================================================
# DISTANCE
# ============================================================

def manhattan_distance(p1, p2):

    return (
        abs(p1[0] - p2[0])
        +
        abs(p1[1] - p2[1])
    )


# ============================================================
# ANALYZE ONE ALGORITHM
# ============================================================

def analyze(algorithm, trajectory):

    print("\n")
    print("=" * 70)
    print(algorithm)
    print("=" * 70)

    print(
        "Steps:",
        len(trajectory)
    )

    print(
        "Starting positions:"
    )

    for robot in range(3):

        print(
            f"  Robot {robot + 1}: "
            f"{tuple(trajectory[0, robot].astype(int))}"
        )

    print(
        "\nFinal positions:"
    )

    for robot in range(3):

        print(
            f"  Robot {robot + 1}: "
            f"{tuple(trajectory[-1, robot].astype(int))}"
        )

    # --------------------------------------------------------
    # Target analysis
    # --------------------------------------------------------

    reached_targets = 0

    print("\nTarget analysis:")

    for target_id, target in enumerate(TARGETS):

        target_reached = False
        first_step = None
        reaching_robot = None

        # Search every timestep
        for step in range(len(trajectory)):

            for robot in range(3):

                position = trajectory[
                    step,
                    robot
                ]

                if (
                    int(position[0]) == target[0]
                    and
                    int(position[1]) == target[1]
                ):

                    target_reached = True
                    first_step = step
                    reaching_robot = robot + 1

                    break

            if target_reached:
                break

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        if target_reached:

            reached_targets += 1

            print(
                f"  T{target_id + 1} {target}: "
                f"REACHED by Robot {reaching_robot} "
                f"at step {first_step}"
            )

        else:

            print(
                f"  T{target_id + 1} {target}: "
                f"NOT REACHED"
            )

            # Find closest robot/step
            best_distance = float("inf")
            best_robot = None
            best_step = None

            for step in range(len(trajectory)):

                for robot in range(3):

                    distance = manhattan_distance(
                        trajectory[step, robot],
                        target
                    )

                    if distance < best_distance:

                        best_distance = distance
                        best_robot = robot + 1
                        best_step = step

            print(
                f"       Closest: Robot {best_robot}, "
                f"step {best_step}, "
                f"distance = {best_distance}"
            )

    # --------------------------------------------------------
    # Path statistics
    # --------------------------------------------------------

    print(
        "\nTargets reached:",
        f"{reached_targets}/3"
    )

    # Calculate movement distance
    total_movement = 0

    for robot in range(3):

        for step in range(1, len(trajectory)):

            total_movement += manhattan_distance(
                trajectory[step, robot],
                trajectory[step - 1, robot]
            )

    print(
        "Total robot movement:",
        total_movement,
        "grid cells"
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("BEST EPISODE TRAJECTORY ANALYSIS")
print("=" * 70)

print("\nTargets:")

for i, target in enumerate(TARGETS):

    print(
        f"  T{i + 1} = {target}"
    )


for algorithm, filename in ALGORITHMS.items():

    path = os.path.join(
        RESULTS_DIR,
        filename
    )

    if not os.path.exists(path):

        print(
            f"\nWARNING: {path} not found"
        )

        continue

    trajectory = load_trajectory(path)

    analyze(
        algorithm,
        trajectory
    )


print("\n")
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)