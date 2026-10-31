import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = "results"

files = {
    "IQL": os.path.join(RESULTS_DIR, "iql_evaluation_results.npz"),
    "VDN": os.path.join(RESULTS_DIR, "vdn_evaluation_results.npz"),
    "QMIX": os.path.join(RESULTS_DIR, "qmix_evaluation_results.npz")
}

# ============================================================
# LOAD RESULTS
# ============================================================

results = {}

for name, path in files.items():
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        continue

    data = np.load(path, allow_pickle=True)

    print(f"\n{name}")
    print("Available keys:", list(data.keys()))

    results[name] = data


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_value(data, possible_names):
    """
    Finds the first matching key from possible_names.
    """
    for name in possible_names:
        if name in data:
            return data[name]
    return None


# ============================================================
# EXTRACT EPISODE VALUES
# ============================================================

metrics = {
    "Reward": {},
    "Treated": {},
    "Discovered": {},
    "Coverage": {},
    "Energy": {},
    "Steps": {},
    "Success": {}
}

for algorithm, data in results.items():

    metrics["Reward"][algorithm] = get_value(
        data,
        ["rewards", "episode_rewards", "reward"]
    )

    metrics["Treated"][algorithm] = get_value(
        data,
        ["treated", "episode_treated", "targets_treated"]
    )

    metrics["Discovered"][algorithm] = get_value(
        data,
        ["discovered", "episode_discovered", "targets_discovered"]
    )

    metrics["Coverage"][algorithm] = get_value(
        data,
        ["coverage", "episode_coverage"]
    )

    metrics["Energy"][algorithm] = get_value(
        data,
        ["energy", "remaining_energy", "episode_energy"]
    )

    metrics["Steps"][algorithm] = get_value(
        data,
        ["steps", "episode_steps"]
    )

    metrics["Success"][algorithm] = get_value(
        data,
        ["success", "success_rate", "episode_success"]
    )


# ============================================================
# PRINT RAW ARRAY INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

for metric_name, algorithms in metrics.items():

    print(f"\n{metric_name}")

    for algorithm, values in algorithms.items():

        if values is None:
            print(f"  {algorithm}: NOT FOUND")
        else:
            print(
                f"  {algorithm}: "
                f"shape={np.asarray(values).shape}"
            )


# ============================================================
# MANUALLY DEFINED SUMMARY
# ============================================================
#
# These are the evaluation results obtained from the project.
# They are included here so the final comparison remains clear.
#

summary = {
    "IQL": {
        "Reward": -58.22,
        "Treated": 1.86,
        "Discovered": 2.78,
        "Coverage": 0.409,
        "Energy": 14.95,
        "Steps": 131.60,
        "Success": 26.0
    },

    "VDN": {
        "Reward": -19.05,
        "Treated": 2.52,
        "Discovered": 2.62,
        "Coverage": 0.268,
        "Energy": 34.51,
        "Steps": 96.50,
        "Success": 64.0
    },

    "QMIX": {
        "Reward": -251.23,
        "Treated": 1.26,
        "Discovered": 2.40,
        "Coverage": 0.280,
        "Energy": 9.90,
        "Steps": 139.22,
        "Success": 10.0
    }
}


# ============================================================
# PRINT COMPARISON TABLE
# ============================================================

print("\n" + "=" * 85)
print("IQL vs VDN vs QMIX - FINAL EVALUATION")
print("=" * 85)

print(
    f"{'Metric':<20}"
    f"{'IQL':>15}"
    f"{'VDN':>15}"
    f"{'QMIX':>15}"
)

print("-" * 85)

for metric in [
    "Reward",
    "Treated",
    "Discovered",
    "Coverage",
    "Energy",
    "Steps",
    "Success"
]:

    print(
        f"{metric:<20}"
        f"{summary['IQL'][metric]:>15.2f}"
        f"{summary['VDN'][metric]:>15.2f}"
        f"{summary['QMIX'][metric]:>15.2f}"
    )

print("=" * 85)


# ============================================================
# CREATE RESULTS DIRECTORY FOR FIGURES
# ============================================================

fig_dir = os.path.join(RESULTS_DIR, "comparison_figures")
os.makedirs(fig_dir, exist_ok=True)


# ============================================================
# GRAPH FUNCTION
# ============================================================

def make_bar_chart(metric, ylabel, title, filename):

    algorithms = ["IQL", "VDN", "QMIX"]

    values = [
        summary["IQL"][metric],
        summary["VDN"][metric],
        summary["QMIX"][metric]
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(algorithms, values)

    plt.ylabel(ylabel)
    plt.xlabel("Algorithm")
    plt.title(title)

    plt.grid(axis="y", alpha=0.3)

    for i, value in enumerate(values):
        plt.text(
            i,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    save_path = os.path.join(fig_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Saved: {save_path}")


# ============================================================
# CREATE COMPARISON GRAPHS
# ============================================================

make_bar_chart(
    "Reward",
    "Average Reward",
    "Average Reward Comparison",
    "reward_comparison.png"
)

make_bar_chart(
    "Treated",
    "Targets Treated",
    "Average Targets Treated",
    "treated_comparison.png"
)

make_bar_chart(
    "Discovered",
    "Targets Discovered",
    "Average Targets Discovered",
    "discovered_comparison.png"
)

make_bar_chart(
    "Coverage",
    "Coverage Ratio",
    "Environment Coverage Comparison",
    "coverage_comparison.png"
)

make_bar_chart(
    "Energy",
    "Remaining Energy",
    "Remaining Energy Comparison",
    "energy_comparison.png"
)

make_bar_chart(
    "Steps",
    "Number of Steps",
    "Episode Length Comparison",
    "steps_comparison.png"
)

make_bar_chart(
    "Success",
    "Success Rate (%)",
    "Task Success Rate Comparison",
    "success_comparison.png"
)


# ============================================================
# FINAL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("FINAL ANALYSIS")
print("=" * 70)

print("""
1. VDN achieved the best overall performance among the three
   algorithms based on the current evaluation.

2. VDN obtained the highest:
   - Average reward
   - Targets treated
   - Remaining energy
   - Success rate

3. IQL obtained the highest:
   - Target discovery
   - Environment coverage

4. QMIX currently performs worse than both IQL and VDN.
   Therefore, the results should NOT be presented as QMIX being
   the best algorithm simply because it is a more advanced
   multi-agent RL method.

5. The current QMIX result indicates that the centralized
   training / decentralized execution architecture has not
   yet learned an effective coordination policy for this
   particular environment.

6. Possible reasons for QMIX underperformance include:
   - Limited training episodes
   - Reward-function sensitivity
   - Energy constraints
   - Difficulty of the environment
   - Global-state representation
   - Mixer learning instability
   - Exploration/evaluation mismatch

7. VDN provides a strong baseline because it learns a joint
   value function through additive value decomposition while
   remaining relatively simple.

8. IQL provides useful independent-agent performance but does
   not explicitly coordinate the nanorobots.

9. The results demonstrate that more complex MARL algorithms
   do not automatically outperform simpler methods. Their
   effectiveness depends strongly on the environment,
   reward design, state representation and training process.
""")

print("\nAll comparison graphs have been generated.")