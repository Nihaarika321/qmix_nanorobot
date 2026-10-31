"""
IQL MODEL EVALUATION
Independent Q-Learning using the existing DQNAgent implementation.

Project:
QMIX for Coordinated Nanorobot Search in Autoimmune Joint Inflammation

IQL baseline:
Each nanorobot has its own independent DQN agent.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# IMPORTS
# ============================================================

from environment.joint_env import JointInflammationEnv
from agents.dqn_agent import DQNAgent


# ============================================================
# CONFIGURATION
# ============================================================

NUM_AGENTS = 3
NUM_TARGETS = 3

STATE_SIZE = 20
ACTION_SIZE = 6

NUM_EPISODES = 50

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(RESULT_DIR, exist_ok=True)

MODEL_PATHS = [
    os.path.join(MODEL_DIR, "iql_agent_1.pth"),
    os.path.join(MODEL_DIR, "iql_agent_2.pth"),
    os.path.join(MODEL_DIR, "iql_agent_3.pth"),
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("IQL MODEL EVALUATION")
print("=" * 60)
print()
print("Device:", DEVICE)
print()


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

env = JointInflammationEnv(
    n_agents=NUM_AGENTS,
    n_targets=NUM_TARGETS
)


# ============================================================
# LOAD IQL AGENTS
# ============================================================

agents = []

for i in range(NUM_AGENTS):

    print(f"Loading IQL Agent {i + 1}:")
    print(MODEL_PATHS[i])

    # IMPORTANT:
    # DQNAgent uses state_size and action_size
    agent = DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE
    )

    if not os.path.exists(MODEL_PATHS[i]):
        raise FileNotFoundError(
            f"\nModel not found:\n{MODEL_PATHS[i]}"
        )

    agent.load(MODEL_PATHS[i])

    # Evaluation only
    agent.epsilon = 0.0

    agents.append(agent)

    print("Loaded successfully.")
    print()


# ============================================================
# STORAGE
# ============================================================

episode_rewards = []
episode_treated = []
episode_discovered = []
episode_coverage = []
episode_energy = []
episode_steps = []

best_reward = -float("inf")
best_trajectory = None
best_result = None


# ============================================================
# EVALUATION
# ============================================================

for episode in range(NUM_EPISODES):

    observations = env.reset()

    total_reward = 0.0
    trajectory = []

    done = False
    steps = 0

    while not done:

        # ----------------------------------------------------
        # Select independent actions
        # ----------------------------------------------------

        actions = []

        for i in range(NUM_AGENTS):

            obs = observations[i]

            action = agents[i].select_action(
                obs,
                training=False
            )

            actions.append(int(action))

        # ----------------------------------------------------
        # Store current state
        # ----------------------------------------------------

        trajectory.append({
            "step": steps,
            "positions": [
                tuple(pos)
                for pos in env.agent_positions
            ],
            "actions": actions.copy(),
            "energy": env.energy.copy()
        })

        # ----------------------------------------------------
        # Environment step
        # ----------------------------------------------------

        result = env.step(actions)

        # Handle either Gymnasium-style 5 outputs
        # or older 4-output environment
        if len(result) == 5:

            next_observations, reward, terminated, truncated, info = result

            done = terminated or truncated

        else:

            next_observations, reward, done, info = result

        # ----------------------------------------------------
        # Accumulate reward
        # ----------------------------------------------------

        total_reward += float(reward)

        observations = next_observations

        steps += 1

        # Safety limit
        if steps >= 300:
            done = True

    # ========================================================
    # EPISODE METRICS
    # ========================================================

    treated = int(np.sum(env.target_treated))

    discovered = int(np.sum(env.target_discovered))

    # Coverage
    visited_cells = np.sum(env.visited)

    total_cells = env.grid_size * env.grid_size

    coverage = visited_cells / total_cells

    # Average remaining energy
    remaining_energy = float(np.mean(env.energy))

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    episode_rewards.append(total_reward)
    episode_treated.append(treated)
    episode_discovered.append(discovered)
    episode_coverage.append(coverage)
    episode_energy.append(remaining_energy)
    episode_steps.append(steps)

    # ========================================================
    # BEST EPISODE
    # ========================================================

    if total_reward > best_reward:

        best_reward = total_reward

        best_trajectory = trajectory.copy()

        best_result = {
            "reward": total_reward,
            "treated": treated,
            "discovered": discovered,
            "coverage": coverage,
            "energy": remaining_energy,
            "steps": steps
        }

    # ========================================================
    # PRINT
    # ========================================================

    print(
        f"Evaluation Episode {episode + 1:2d}/{NUM_EPISODES} | "
        f"Reward: {total_reward:8.2f} | "
        f"Treated: {treated}/{NUM_TARGETS} | "
        f"Discovered: {discovered}/{NUM_TARGETS} | "
        f"Coverage: {coverage:.3f} | "
        f"Energy: {remaining_energy:.2f}"
    )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

episode_rewards = np.array(episode_rewards)

episode_treated = np.array(episode_treated)

episode_discovered = np.array(
    episode_discovered
)

episode_coverage = np.array(
    episode_coverage
)

episode_energy = np.array(
    episode_energy
)

episode_steps = np.array(
    episode_steps
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("IQL EVALUATION COMPLETE")
print("=" * 60)
print()

print(
    f"Average Reward: "
    f"{np.mean(episode_rewards):.2f}"
)

print(
    f"Average Treated: "
    f"{np.mean(episode_treated):.2f}/{NUM_TARGETS}"
)

print(
    f"Average Discovered: "
    f"{np.mean(episode_discovered):.2f}/{NUM_TARGETS}"
)

print(
    f"Average Coverage: "
    f"{np.mean(episode_coverage):.3f}"
)

print(
    f"Average Remaining Energy: "
    f"{np.mean(episode_energy):.2f}"
)

print(
    f"Average Steps: "
    f"{np.mean(episode_steps):.2f}"
)


# ============================================================
# SUCCESS RATE
# ============================================================

success_rate = (
    np.sum(episode_treated == NUM_TARGETS)
    / NUM_EPISODES
    * 100
)

print(
    f"Success Rate: "
    f"{success_rate:.2f}%"
)


# ============================================================
# BEST EPISODE
# ============================================================

print()
print("=" * 60)
print("BEST IQL EPISODE")
print("=" * 60)

print(
    f"Reward: "
    f"{best_result['reward']:.2f}"
)

print(
    f"Targets Treated: "
    f"{best_result['treated']}/{NUM_TARGETS}"
)

print(
    f"Targets Discovered: "
    f"{best_result['discovered']}/{NUM_TARGETS}"
)

print(
    f"Coverage: "
    f"{best_result['coverage']:.3f}"
)

print(
    f"Remaining Energy: "
    f"{best_result['energy']:.2f}"
)

print(
    f"Steps: "
    f"{best_result['steps']}"
)


# ============================================================
# SAVE NUMERICAL RESULTS
# ============================================================

results_path = os.path.join(
    RESULT_DIR,
    "iql_evaluation_results.npz"
)

np.savez(
    results_path,
    rewards=episode_rewards,
    treated=episode_treated,
    discovered=episode_discovered,
    coverage=episode_coverage,
    energy=episode_energy,
    steps=episode_steps
)


# ============================================================
# SAVE BEST TRAJECTORY
# ============================================================

trajectory_path = os.path.join(
    RESULT_DIR,
    "iql_best_trajectory.npy"
)

np.save(
    trajectory_path,
    np.array(best_trajectory, dtype=object),
    allow_pickle=True
)


# ============================================================
# PLOT 1 — REWARD
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    episode_rewards,
    label="Episode Reward"
)

if NUM_EPISODES >= 20:

    moving_average = np.convolve(
        episode_rewards,
        np.ones(20) / 20,
        mode="valid"
    )

    plt.plot(
        range(19, NUM_EPISODES),
        moving_average,
        label="20-Episode Average"
    )

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("IQL Evaluation Reward")
plt.grid(True)
plt.legend()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "iql_evaluation_reward.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PLOT 2 — TARGET TREATMENT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    episode_treated,
    label="Targets Treated"
)

plt.axhline(
    NUM_TARGETS,
    linestyle="--",
    label="All Targets"
)

plt.xlabel("Episode")
plt.ylabel("Targets Treated")
plt.title("IQL Evaluation Target Treatment")
plt.grid(True)
plt.legend()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "iql_evaluation_treatment.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PLOT 3 — COVERAGE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    episode_coverage,
    label="Coverage"
)

plt.xlabel("Episode")
plt.ylabel("Coverage")
plt.title("IQL Evaluation Environment Coverage")
plt.grid(True)
plt.legend()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "iql_evaluation_coverage.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PLOT 4 — REMAINING ENERGY
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    episode_energy,
    label="Average Remaining Energy"
)

plt.xlabel("Episode")
plt.ylabel("Energy")
plt.title("IQL Evaluation Remaining Energy")
plt.grid(True)
plt.legend()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "iql_evaluation_energy.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 60)
print("IQL EVALUATION FILES SAVED")
print("=" * 60)

print()
print("Evaluation results:")
print(results_path)

print()
print("Best trajectory:")
print(trajectory_path)

print()
print("Plots saved in:")
print(RESULT_DIR)

print()
print("=" * 60)
print("IQL EVALUATION COMPLETE")
print("=" * 60)