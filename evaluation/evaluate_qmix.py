# ============================================================
# EVALUATE QMIX
# ============================================================
#
# Evaluates the trained QMIX model and saves:
#   1. Aggregate evaluation metrics
#   2. Best episode robot trajectory
#   3. ACTUAL target positions of the best episode
#   4. Actions taken in the best episode
#   5. Energy history
#   6. Target discovery/treatment history
#
# This allows accurate visualization of the SAME episode
# used for the best-trajectory biological visualization.
# ============================================================

import os
import numpy as np
import torch

from environment.joint_env import JointInflammationEnv
from qmix.qmix_learner import QMIXLearner


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "results",
    "qmix_nanobot.pth"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# EVALUATION SETTINGS
# ============================================================

NUM_EPISODES = 50

N_AGENTS = 3
OBS_DIM = 20
ACTION_DIM = 6
STATE_DIM = 824

HIDDEN_DIM = 64
MIXING_EMBED_DIM = 32

LEARNING_RATE = 0.001
GAMMA = 0.99

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

env = JointInflammationEnv()


# ============================================================
# CREATE QMIX LEARNER
# ============================================================

learner = QMIXLearner(
    n_agents=N_AGENTS,
    obs_dim=OBS_DIM,
    action_dim=ACTION_DIM,
    state_dim=STATE_DIM,
    hidden_dim=HIDDEN_DIM,
    mixing_embed_dim=MIXING_EMBED_DIM,
    lr=LEARNING_RATE,
    gamma=GAMMA,
    device=DEVICE
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("=" * 70)
print("LOADING QMIX MODEL")
print("=" * 70)

print("Model path:")
print(MODEL_PATH)

learner.load(MODEL_PATH)

print("QMIX model loaded successfully.")
print("Device:", DEVICE)


# ============================================================
# STORAGE FOR EVALUATION RESULTS
# ============================================================

episode_rewards = []
episode_treated = []
episode_discovered = []
episode_coverage = []
episode_energy = []
episode_steps = []
episode_success = []


# ============================================================
# BEST EPISODE STORAGE
# ============================================================

best_reward = -float("inf")

best_trajectory = None
best_target_positions = None
best_actions = None
best_energy_history = None
best_discovered_history = None
best_treated_history = None
best_coverage_history = None

best_episode_info = None


# ============================================================
# EVALUATION LOOP
# ============================================================

print()
print("=" * 70)
print("STARTING QMIX EVALUATION")
print("=" * 70)

for episode in range(NUM_EPISODES):

    # --------------------------------------------------------
    # RESET ENVIRONMENT
    # --------------------------------------------------------

    observations = env.reset()

    # --------------------------------------------------------
    # IMPORTANT:
    # Save the actual targets generated for THIS episode.
    #
    # These targets are randomized by the environment.
    # --------------------------------------------------------

    episode_target_positions = [
        tuple(position)
        for position in env.target_positions
    ]

    # --------------------------------------------------------
    # Initial robot positions
    # --------------------------------------------------------

    trajectory = [
        np.array(env.agent_positions, dtype=np.int32)
    ]

    # --------------------------------------------------------
    # Per-step histories
    # --------------------------------------------------------

    actions_history = []

    energy_history = [
        np.array(env.energy, dtype=np.float32)
    ]

    discovered_history = [
        int(np.sum(env.target_discovered))
    ]

    treated_history = [
        int(np.sum(env.target_treated))
    ]

    coverage_history = [
        float(
            np.sum(env.visited)
            /
            (env.grid_size * env.grid_size)
        )
    ]

    total_reward = 0.0

    done = False
    step_count = 0


    # ========================================================
    # EPISODE LOOP
    # ========================================================

    while not done:

        # ----------------------------------------------------
        # Select deterministic QMIX actions
        # ----------------------------------------------------

        actions = learner.select_actions(
            observations,
            epsilon=0.0
        )

        # Convert to normal Python list
        actions = [int(a) for a in actions]

        actions_history.append(actions.copy())

        # ----------------------------------------------------
        # Environment step
        # ----------------------------------------------------

        result = env.step(actions)

        # ----------------------------------------------------
        # Support both:
        #
        # observations, reward, done, info
        #
        # and
        #
        # observations, reward, terminated,
        # truncated, info
        # ----------------------------------------------------

        if len(result) == 5:

            next_observations, reward, terminated, truncated, info = result

            done = bool(
                terminated or truncated
            )

        else:

            next_observations, reward, done, info = result

            done = bool(done)

        # ----------------------------------------------------
        # Update reward
        # ----------------------------------------------------

        total_reward += float(reward)

        # ----------------------------------------------------
        # Update observations
        # ----------------------------------------------------

        observations = next_observations

        # ----------------------------------------------------
        # Step counter
        # ----------------------------------------------------

        step_count += 1

        # ----------------------------------------------------
        # Save robot positions
        # ----------------------------------------------------

        trajectory.append(
            np.array(
                env.agent_positions,
                dtype=np.int32
            )
        )

        # ----------------------------------------------------
        # Save energy
        # ----------------------------------------------------

        energy_history.append(
            np.array(
                env.energy,
                dtype=np.float32
            )
        )

        # ----------------------------------------------------
        # Save discovery status
        # ----------------------------------------------------

        discovered_history.append(
            int(
                np.sum(
                    env.target_discovered
                )
            )
        )

        # ----------------------------------------------------
        # Save treatment status
        # ----------------------------------------------------

        treated_history.append(
            int(
                np.sum(
                    env.target_treated
                )
            )
        )

        # ----------------------------------------------------
        # Save coverage
        # ----------------------------------------------------

        coverage_history.append(
            float(
                np.sum(env.visited)
                /
                (env.grid_size * env.grid_size)
            )
        )


    # ========================================================
    # EPISODE METRICS
    # ========================================================

    final_treated = int(
        np.sum(env.target_treated)
    )

    final_discovered = int(
        np.sum(env.target_discovered)
    )

    final_coverage = float(
        np.sum(env.visited)
        /
        (env.grid_size * env.grid_size)
    )

    final_energy = float(
        np.mean(env.energy)
    )

    success = (
        final_treated == env.n_targets
    )


    # ========================================================
    # STORE EPISODE RESULTS
    # ========================================================

    episode_rewards.append(
        total_reward
    )

    episode_treated.append(
        final_treated
    )

    episode_discovered.append(
        final_discovered
    )

    episode_coverage.append(
        final_coverage
    )

    episode_energy.append(
        final_energy
    )

    episode_steps.append(
        step_count
    )

    episode_success.append(
        int(success)
    )


    # ========================================================
    # CHECK FOR BEST EPISODE
    # ========================================================

    if total_reward > best_reward:

        best_reward = total_reward

        # ----------------------------------------------------
        # Robot trajectory
        # ----------------------------------------------------

        best_trajectory = np.array(
            trajectory,
            dtype=np.int32
        )

        # ----------------------------------------------------
        # ACTUAL target positions
        # ----------------------------------------------------

        best_target_positions = np.array(
            episode_target_positions,
            dtype=np.int32
        )

        # ----------------------------------------------------
        # Actions
        # ----------------------------------------------------

        best_actions = np.array(
            actions_history,
            dtype=np.int32
        )

        # ----------------------------------------------------
        # Energy history
        # ----------------------------------------------------

        best_energy_history = np.array(
            energy_history,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Discovery history
        # ----------------------------------------------------

        best_discovered_history = np.array(
            discovered_history,
            dtype=np.int32
        )

        # ----------------------------------------------------
        # Treatment history
        # ----------------------------------------------------

        best_treated_history = np.array(
            treated_history,
            dtype=np.int32
        )

        # ----------------------------------------------------
        # Coverage history
        # ----------------------------------------------------

        best_coverage_history = np.array(
            coverage_history,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Best episode summary
        # ----------------------------------------------------

        best_episode_info = {
            "reward": float(total_reward),
            "treated": int(final_treated),
            "discovered": int(final_discovered),
            "coverage": float(final_coverage),
            "energy": float(final_energy),
            "steps": int(step_count),
            "target_positions": best_target_positions.copy()
        }


    # ========================================================
    # PRINT PROGRESS
    # ========================================================

    print(
        f"Episode {episode + 1:02d}/{NUM_EPISODES} | "
        f"Reward: {total_reward:8.2f} | "
        f"Treated: {final_treated}/{env.n_targets} | "
        f"Discovered: {final_discovered}/{env.n_targets} | "
        f"Coverage: {final_coverage:.3f} | "
        f"Energy: {final_energy:6.2f} | "
        f"Steps: {step_count:3d}"
    )


# ============================================================
# CONVERT RESULTS TO NUMPY ARRAYS
# ============================================================

episode_rewards = np.array(
    episode_rewards,
    dtype=np.float64
)

episode_treated = np.array(
    episode_treated,
    dtype=np.int32
)

episode_discovered = np.array(
    episode_discovered,
    dtype=np.int32
)

episode_coverage = np.array(
    episode_coverage,
    dtype=np.float64
)

episode_energy = np.array(
    episode_energy,
    dtype=np.float64
)

episode_steps = np.array(
    episode_steps,
    dtype=np.int32
)

episode_success = np.array(
    episode_success,
    dtype=np.int32
)


# ============================================================
# SAVE AGGREGATE EVALUATION RESULTS
# ============================================================

evaluation_results_path = os.path.join(
    RESULTS_DIR,
    "qmix_evaluation_results.npz"
)

np.savez(
    evaluation_results_path,

    rewards=episode_rewards,

    treated=episode_treated,

    discovered=episode_discovered,

    coverage=episode_coverage,

    energy=episode_energy,

    steps=episode_steps,

    success=episode_success
)


# ============================================================
# SAVE BEST ROBOT TRAJECTORY
# ============================================================

trajectory_path = os.path.join(
    RESULTS_DIR,
    "qmix_best_trajectory.npy"
)

np.save(
    trajectory_path,
    best_trajectory
)


# ============================================================
# SAVE ACTUAL TARGET POSITIONS
# ============================================================

target_path = os.path.join(
    RESULTS_DIR,
    "qmix_best_target_positions.npy"
)

np.save(
    target_path,
    best_target_positions
)


# ============================================================
# SAVE BEST ACTIONS
# ============================================================

actions_path = os.path.join(
    RESULTS_DIR,
    "qmix_best_actions.npy"
)

np.save(
    actions_path,
    best_actions
)


# ============================================================
# SAVE BEST ENERGY HISTORY
# ============================================================

energy_path = os.path.join(
    RESULTS_DIR,
    "qmix_best_energy_history.npy"
)

np.save(
    energy_path,
    best_energy_history
)


# ============================================================
# SAVE DISCOVERY HISTORY
# ============================================================

discovered_path = os.path.join(
    RESULTS_DIR,
    "qmix_best_discovered_history.npy"
)

np.save(
    discovered_path,
    best_discovered_history
)


# ============================================================
# SAVE TREATMENT HISTORY
# ============================================================

treated_path = os.path.join(
    RESULTS_DIR,
    "qmix_best_treated_history.npy"
)

np.save(
    treated_path,
    best_treated_history
)


# ============================================================
# SAVE COVERAGE HISTORY
# ============================================================

coverage_path = os.path.join(
    RESULTS_DIR,
    "qmix_best_coverage_history.npy"
)

np.save(
    coverage_path,
    best_coverage_history
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("QMIX EVALUATION COMPLETE")
print("=" * 70)

print()

print(
    f"Average Reward: "
    f"{np.mean(episode_rewards):.2f}"
)

print(
    f"Average Treated: "
    f"{np.mean(episode_treated):.2f}/"
    f"{env.n_targets}"
)

print(
    f"Average Discovered: "
    f"{np.mean(episode_discovered):.2f}/"
    f"{env.n_targets}"
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

print(
    f"Success Rate: "
    f"{100.0 * np.mean(episode_success):.2f}%"
)


# ============================================================
# BEST EPISODE
# ============================================================

print()
print("-" * 70)
print("BEST QMIX EPISODE")
print("-" * 70)

print(
    f"Reward:     "
    f"{best_episode_info['reward']:.2f}"
)

print(
    f"Treated:    "
    f"{best_episode_info['treated']}/{env.n_targets}"
)

print(
    f"Discovered: "
    f"{best_episode_info['discovered']}/{env.n_targets}"
)

print(
    f"Coverage:   "
    f"{best_episode_info['coverage']:.3f}"
)

print(
    f"Energy:     "
    f"{best_episode_info['energy']:.2f}"
)

print(
    f"Steps:      "
    f"{best_episode_info['steps']}"
)

print(
    "Targets:    "
    f"{best_target_positions.tolist()}"
)


# ============================================================
# FILE LOCATIONS
# ============================================================

print()
print("-" * 70)
print("SAVED FILES")
print("-" * 70)

print(
    "Evaluation results:"
)
print(
    evaluation_results_path
)

print(
    "Best trajectory:"
)
print(
    trajectory_path
)

print(
    "Best target positions:"
)
print(
    target_path
)

print(
    "Best actions:"
)
print(
    actions_path
)

print(
    "Best energy history:"
)
print(
    energy_path
)

print(
    "Best discovery history:"
)
print(
    discovered_path
)

print(
    "Best treatment history:"
)
print(
    treated_path
)

print(
    "Best coverage history:"
)
print(
    coverage_path
)

print()
print("=" * 70)