import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = "results"
OUTPUT_DIR = "results/trajectory_figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD RESULTS
# ============================================================

iql = np.load(
    os.path.join(RESULTS_DIR, "iql_evaluation_results.npz"),
    allow_pickle=True
)

vdn = np.load(
    os.path.join(RESULTS_DIR, "vdn_evaluation_results.npz"),
    allow_pickle=True
)

qmix = np.load(
    os.path.join(RESULTS_DIR, "qmix_evaluation_results.npz"),
    allow_pickle=True
)


# ============================================================
# EXTRACT DATA
# ============================================================

# IQL
iql_rewards = iql["rewards"]
iql_treated = iql["treated"]
iql_discovered = iql["discovered"]
iql_coverage = iql["coverage"]
iql_energy = iql["energy"]
iql_steps = iql["steps"]

# VDN
vdn_rewards = vdn["episode_rewards"]
vdn_treated = vdn["episode_treated"]
vdn_discovered = vdn["episode_discovered"]
vdn_coverage = vdn["episode_coverage"]
vdn_energy = vdn["episode_energy"]
vdn_steps = vdn["episode_steps"]

# QMIX
qmix_rewards = qmix["rewards"]
qmix_treated = qmix["treated"]
qmix_discovered = qmix["discovered"]
qmix_coverage = qmix["coverage"]
qmix_energy = qmix["energy"]
qmix_steps = qmix["steps"]


# ============================================================
# SUCCESS RATE
# ============================================================

# IQL: success = all 3 targets treated
iql_success = np.mean(iql_treated >= 3) * 100

# VDN already stores success rate
vdn_success = float(vdn["success_rate"]) * 100 \
    if np.max(vdn["success_rate"]) <= 1 else float(vdn["success_rate"])

# QMIX: success = all 3 targets treated
if "success" in qmix.files:
    qmix_success_data = qmix["success"]
    qmix_success = np.mean(qmix_success_data) * 100
else:
    qmix_success = np.mean(qmix_treated >= 3) * 100


# ============================================================
# CALCULATE MEANS
# ============================================================

algorithms = ["IQL", "VDN", "QMIX"]

avg_reward = [
    np.mean(iql_rewards),
    np.mean(vdn_rewards),
    np.mean(qmix_rewards)
]

avg_treated = [
    np.mean(iql_treated),
    np.mean(vdn_treated),
    np.mean(qmix_treated)
]

avg_discovered = [
    np.mean(iql_discovered),
    np.mean(vdn_discovered),
    np.mean(qmix_discovered)
]

avg_coverage = [
    np.mean(iql_coverage),
    np.mean(vdn_coverage),
    np.mean(qmix_coverage)
]

avg_energy = [
    np.mean(iql_energy),
    np.mean(vdn_energy),
    np.mean(qmix_energy)
]

avg_steps = [
    np.mean(iql_steps),
    np.mean(vdn_steps),
    np.mean(qmix_steps)
]

success_rates = [
    iql_success,
    vdn_success,
    qmix_success
]


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL ALGORITHM COMPARISON")
print("=" * 70)

print(f"{'Metric':<25} {'IQL':>12} {'VDN':>12} {'QMIX':>12}")
print("-" * 70)

print(
    f"{'Average Reward':<25}"
    f"{avg_reward[0]:>12.2f}"
    f"{avg_reward[1]:>12.2f}"
    f"{avg_reward[2]:>12.2f}"
)

print(
    f"{'Targets Treated':<25}"
    f"{avg_treated[0]:>12.2f}"
    f"{avg_treated[1]:>12.2f}"
    f"{avg_treated[2]:>12.2f}"
)

print(
    f"{'Targets Discovered':<25}"
    f"{avg_discovered[0]:>12.2f}"
    f"{avg_discovered[1]:>12.2f}"
    f"{avg_discovered[2]:>12.2f}"
)

print(
    f"{'Coverage':<25}"
    f"{avg_coverage[0]:>12.3f}"
    f"{avg_coverage[1]:>12.3f}"
    f"{avg_coverage[2]:>12.3f}"
)

print(
    f"{'Remaining Energy':<25}"
    f"{avg_energy[0]:>12.2f}"
    f"{avg_energy[1]:>12.2f}"
    f"{avg_energy[2]:>12.2f}"
)

print(
    f"{'Episode Steps':<25}"
    f"{avg_steps[0]:>12.2f}"
    f"{avg_steps[1]:>12.2f}"
    f"{avg_steps[2]:>12.2f}"
)

print(
    f"{'Success Rate (%)':<25}"
    f"{success_rates[0]:>12.2f}"
    f"{success_rates[1]:>12.2f}"
    f"{success_rates[2]:>12.2f}"
)

print("=" * 70)


# ============================================================
# PLOT FUNCTION
# ============================================================

def create_bar_plot(values, title, ylabel, filename, decimals=2):

    plt.figure(figsize=(8, 6))

    bars = plt.bar(algorithms, values)

    plt.title(title)
    plt.ylabel(ylabel)

    # Add values above bars
    for bar, value in zip(bars, values):

        if decimals == 0:
            label = f"{value:.0f}"
        elif decimals == 1:
            label = f"{value:.1f}"
        else:
            label = f"{value:.2f}"

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


# ============================================================
# CREATE GRAPHS
# ============================================================

create_bar_plot(
    avg_reward,
    "Average Episode Reward",
    "Average Reward",
    "comparison_reward.png",
    decimals=2
)

create_bar_plot(
    avg_treated,
    "Average Targets Treated",
    "Targets Treated",
    "comparison_treated.png",
    decimals=2
)

create_bar_plot(
    avg_discovered,
    "Average Targets Discovered",
    "Targets Discovered",
    "comparison_discovered.png",
    decimals=2
)

create_bar_plot(
    avg_coverage,
    "Average Environment Coverage",
    "Coverage",
    "comparison_coverage.png",
    decimals=3
)

create_bar_plot(
    avg_energy,
    "Average Remaining Energy",
    "Remaining Energy",
    "comparison_energy.png",
    decimals=2
)

create_bar_plot(
    avg_steps,
    "Average Episode Steps",
    "Steps",
    "comparison_steps.png",
    decimals=2
)

create_bar_plot(
    success_rates,
    "Episode Success Rate",
    "Success Rate (%)",
    "comparison_success_rate.png",
    decimals=1
)


print("\n" + "=" * 70)
print("ALL COMPARISON PLOTS GENERATED")
print("=" * 70)