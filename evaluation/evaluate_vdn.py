# ============================================================
# EVALUATE VDN
# ============================================================

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from environment.joint_env import JointInflammationEnv
from qmix.vdn_agent import VDNAgent


# ============================================================
# CONFIGURATION
# ============================================================

NUM_AGENTS = 3
OBS_DIM = 20
ACTION_DIM = 6

NUM_EPISODES = 50
MAX_STEPS = 200

MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_observations(reset_result):
    """
    Handles different reset() return formats.

    Expected project format:
        observations
        observations, info
        observations, global_state, info
    """

    if isinstance(reset_result, tuple):

        # First element is always the local observations
        observations = reset_result[0]

    else:
        observations = reset_result

    return np.asarray(observations, dtype=np.float32)


def extract_step_result(step_result):
    """
    Handles both Gymnasium-style and older Gym-style step outputs.
    """

    if len(step_result) == 5:

        next_obs, reward, terminated, truncated, info = step_result

        done = bool(terminated or truncated)

        return (
            np.asarray(next_obs, dtype=np.float32),
            float(reward),
            done,
            info
        )

    elif len(step_result) == 4:

        next_obs, reward, done, info = step_result

        return (
            np.asarray(next_obs, dtype=np.float32),
            float(reward),
            bool(done),
            info
        )

    else:

        raise ValueError(
            f"Unexpected env.step() output length: "
            f"{len(step_result)}"
        )


def get_value(source, names, default=None):
    """
    Safely retrieve an attribute/key from either:
        - object
        - dictionary
    """

    if source is None:
        return default

    for name in names:

        # Dictionary
        if isinstance(source, dict):

            if name in source:
                return source[name]

        # Object attribute
        if hasattr(source, name):

            try:
                return getattr(source, name)
            except Exception:
                pass

    return default


def get_metric(env, info, names, default=0.0):
    """
    Try to obtain a metric from:
        1. info
        2. environment
    """

    value = get_value(info, names, None)

    if value is not None:
        return value

    value = get_value(env, names, None)

    if value is not None:
        return value

    return default


# ============================================================
# POSITION EXTRACTION
# ============================================================

def extract_positions(env, info):
    """
    Try multiple possible names used by the environment.

    Returns:
        list of (x, y) positions
    """

    candidate_names = [

        "robot_positions",
        "agent_positions",
        "positions",
        "robot_pos",
        "agent_pos",
        "current_positions",
        "current_position"
    ]

    # --------------------------------------------------------
    # First try info
    # --------------------------------------------------------

    positions = get_value(
        info,
        candidate_names,
        None
    )

    # --------------------------------------------------------
    # Then try environment
    # --------------------------------------------------------

    if positions is None:

        positions = get_value(
            env,
            candidate_names,
            None
        )

    # --------------------------------------------------------
    # Nothing found
    # --------------------------------------------------------

    if positions is None:
        return None

    try:

        positions = np.asarray(
            positions,
            dtype=float
        )

        # Single robot position
        if positions.ndim == 1:

            if len(positions) >= 2:

                return [
                    (
                        int(round(positions[0])),
                        int(round(positions[1]))
                    )
                ]

        # Multiple robot positions
        if positions.ndim >= 2:

            result = []

            for p in positions:

                if len(p) >= 2:

                    result.append(
                        (
                            int(round(p[0])),
                            int(round(p[1]))
                        )
                    )

            if len(result) > 0:
                return result

    except Exception:
        pass

    return None


# ============================================================
# GRID EXTRACTION
# ============================================================

def extract_grid(env, info):
    """
    Try to obtain the environment grid.

    This is mainly for identifying valid cells.
    """

    grid_names = [
        "grid",
        "map",
        "environment_grid",
        "world_grid",
        "occupancy_grid"
    ]

    grid = get_value(
        info,
        grid_names,
        None
    )

    if grid is None:

        grid = get_value(
            env,
            grid_names,
            None
        )

    if grid is not None:

        try:
            return np.asarray(grid)

        except Exception:
            pass

    return None


# ============================================================
# INITIALIZE ENVIRONMENT
# ============================================================

print()
print("=" * 60)
print("VDN MODEL EVALUATION")
print("=" * 60)
print()

env = JointInflammationEnv()

print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
print()

print("Observation shape:")

reset_result = env.reset()

observations = extract_observations(
    reset_result
)

print(observations.shape)

print()
print("Observation dimension:", observations.shape[-1])
print()


# ============================================================
# CREATE VDN AGENT
# ============================================================

agent = VDNAgent(
    n_agents=NUM_AGENTS,
    state_size=OBS_DIM,
    action_size=ACTION_DIM
)

print("=" * 60)
print("LOADING VDN AGENTS")
print("=" * 60)
print()

# ------------------------------------------------------------
# IMPORTANT:
# VDNAgent.load() already loads:
#
# vdn_agent_1.pth
# vdn_agent_2.pth
# vdn_agent_3.pth
# ------------------------------------------------------------

agent.load(
    MODEL_DIR
)

# Evaluation must be greedy
agent.epsilon = 0.0

for network in agent.q_networks:
    network.eval()


print()
print("VDN models loaded successfully.")
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
best_episode_data = None


# ============================================================
# EVALUATION
# ============================================================

for episode in range(1, NUM_EPISODES + 1):

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    reset_result = env.reset()

    observations = extract_observations(
        reset_result
    )

    # --------------------------------------------------------
    # VISITED CELLS
    #
    # This is the important coverage fix.
    #
    # We maintain our own set of visited cells rather than
    # depending on env.robot_positions / env.visited_cells.
    # --------------------------------------------------------

    visited_cells = set()

    trajectory = []

    # Try to obtain initial positions
    positions = extract_positions(
        env,
        None
    )

    if positions is not None:

        for position in positions:

            visited_cells.add(
                (
                    int(position[0]),
                    int(position[1])
                )
            )

    # --------------------------------------------------------
    # Episode variables
    # --------------------------------------------------------

    total_reward = 0.0
    steps = 0

    final_info = None

    # --------------------------------------------------------
    # EPISODE LOOP
    # --------------------------------------------------------

    for step in range(1, MAX_STEPS + 1):

        steps = step

        # ----------------------------------------------------
        # Select greedy VDN actions
        # ----------------------------------------------------

        actions = agent.select_actions(
            observations,
            training=False
        )

        # ----------------------------------------------------
        # Store positions before action
        # ----------------------------------------------------

        old_positions = extract_positions(
            env,
            None
        )

        if old_positions is not None:

            for position in old_positions:

                visited_cells.add(
                    (
                        int(position[0]),
                        int(position[1])
                    )
                )

        # ----------------------------------------------------
        # Environment step
        # ----------------------------------------------------

        step_result = env.step(
            actions
        )

        (
            next_observations,
            reward,
            done,
            info
        ) = extract_step_result(
            step_result
        )

        final_info = info

        total_reward += reward

        # ----------------------------------------------------
        # Get robot positions AFTER action
        # ----------------------------------------------------

        positions = extract_positions(
            env,
            info
        )

        if positions is not None:

            clean_positions = []

            for position in positions:

                x = int(round(position[0]))
                y = int(round(position[1]))

                visited_cells.add(
                    (x, y)
                )

                clean_positions.append(
                    (x, y)
                )

            trajectory.append(
                clean_positions
            )

        else:

            # If environment doesn't expose positions,
            # preserve the timestep in trajectory.
            trajectory.append(None)

        # ----------------------------------------------------
        # Update observation
        # ----------------------------------------------------

        observations = next_observations

        # ----------------------------------------------------
        # Stop
        # ----------------------------------------------------

        if done:
            break

    # ========================================================
    # FINAL METRICS
    # ========================================================

    # --------------------------------------------------------
    # Treated
    # --------------------------------------------------------

    treated = get_metric(
        env,
        final_info,
        [
            "targets_treated",
            "target_treated",
            "treated_targets",
            "num_targets_treated",
            "treated"
        ],
        default=0
    )

    # --------------------------------------------------------
    # Discovered
    # --------------------------------------------------------

    discovered = get_metric(
        env,
        final_info,
        [
            "targets_discovered",
            "target_discovered",
            "discovered_targets",
            "num_targets_discovered",
            "discovered"
        ],
        default=0
    )

    # --------------------------------------------------------
    # Energy
    # --------------------------------------------------------

    energy = get_metric(
        env,
        final_info,
        [
            "remaining_energy",
            "energy",
            "robot_energy",
            "energies"
        ],
        default=0.0
    )

    # --------------------------------------------------------
    # Convert energy array to average remaining energy
    # --------------------------------------------------------

    try:

        energy_array = np.asarray(
            energy,
            dtype=float
        )

        if energy_array.size > 1:

            energy = float(
                np.mean(
                    np.maximum(
                        energy_array,
                        0.0
                    )
                )
            )

        else:

            energy = float(
                max(
                    float(energy_array.reshape(-1)[0]),
                    0.0
                )
            )

    except Exception:

        energy = 0.0

    # --------------------------------------------------------
    # COVERAGE
    #
    # The environment is 20 x 20.
    #
    # Therefore:
    #
    # coverage = unique visited cells / 400
    #
    # This avoids the previous 0.000 issue.
    # --------------------------------------------------------

    GRID_SIZE = 20

    valid_visited = set()

    for cell in visited_cells:

        try:

            x = int(cell[0])
            y = int(cell[1])

            if (
                0 <= x < GRID_SIZE
                and
                0 <= y < GRID_SIZE
            ):

                valid_visited.add(
                    (x, y)
                )

        except Exception:
            pass

    coverage = (
        len(valid_visited)
        /
        float(GRID_SIZE * GRID_SIZE)
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    episode_rewards.append(
        total_reward
    )

    episode_treated.append(
        float(treated)
    )

    episode_discovered.append(
        float(discovered)
    )

    episode_coverage.append(
        coverage
    )

    episode_energy.append(
        energy
    )

    episode_steps.append(
        steps
    )

    # --------------------------------------------------------
    # Best episode
    # --------------------------------------------------------

    if total_reward > best_reward:

        best_reward = total_reward

        best_trajectory = trajectory

        best_episode_data = {
            "reward": total_reward,
            "treated": treated,
            "discovered": discovered,
            "coverage": coverage,
            "energy": energy,
            "steps": steps
        }

    # ========================================================
    # PRINT
    # ========================================================

    print(
        f"Evaluation Episode "
        f"{episode:2d}/{NUM_EPISODES} | "
        f"Reward: {total_reward:8.2f} | "
        f"Treated: {int(treated)}/3 | "
        f"Discovered: {int(discovered)}/3 | "
        f"Coverage: {coverage:.3f} | "
        f"Energy: {energy:.2f}"
    )


# ============================================================
# CONVERT RESULTS TO NUMPY
# ============================================================

episode_rewards = np.asarray(
    episode_rewards,
    dtype=np.float32
)

episode_treated = np.asarray(
    episode_treated,
    dtype=np.float32
)

episode_discovered = np.asarray(
    episode_discovered,
    dtype=np.float32
)

episode_coverage = np.asarray(
    episode_coverage,
    dtype=np.float32
)

episode_energy = np.asarray(
    episode_energy,
    dtype=np.float32
)

episode_steps = np.asarray(
    episode_steps,
    dtype=np.float32
)


# ============================================================
# SUCCESS RATE
# ============================================================

success_rate = (
    np.mean(
        episode_treated >= NUM_AGENTS
    )
    *
    100.0
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()
print("=" * 60)
print("VDN EVALUATION COMPLETE")
print("=" * 60)
print()

print(
    f"Average Reward: "
    f"{np.mean(episode_rewards):.2f}"
)

print(
    f"Average Treated: "
    f"{np.mean(episode_treated):.2f}/3"
)

print(
    f"Average Discovered: "
    f"{np.mean(episode_discovered):.2f}/3"
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
    f"{success_rate:.2f}%"
)


# ============================================================
# BEST EPISODE
# ============================================================

print()
print("=" * 60)
print("BEST VDN EPISODE")
print("=" * 60)
print()

print(
    f"Reward: "
    f"{best_episode_data['reward']:.2f}"
)

print(
    f"Targets Treated: "
    f"{int(best_episode_data['treated'])}/3"
)

print(
    f"Targets Discovered: "
    f"{int(best_episode_data['discovered'])}/3"
)

print(
    f"Coverage: "
    f"{best_episode_data['coverage']:.3f}"
)

print(
    f"Remaining Energy: "
    f"{best_episode_data['energy']:.2f}"
)

print(
    f"Steps: "
    f"{best_episode_data['steps']}"
)


# ============================================================
# SAVE EVALUATION RESULTS
# ============================================================

results_path = os.path.join(
    RESULT_DIR,
    "vdn_evaluation_results.npz"
)

np.savez(
    results_path,

    episode_rewards=episode_rewards,

    episode_treated=episode_treated,

    episode_discovered=episode_discovered,

    episode_coverage=episode_coverage,

    episode_energy=episode_energy,

    episode_steps=episode_steps,

    success_rate=success_rate
)


# ============================================================
# SAVE BEST TRAJECTORY
# ============================================================

trajectory_path = os.path.join(
    RESULT_DIR,
    "vdn_best_trajectory.npy"
)

np.save(
    trajectory_path,
    np.array(
        best_trajectory,
        dtype=object
    ),
    allow_pickle=True
)


# ============================================================
# PLOT 1: REWARD
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    episode_rewards,
    label="Episode Reward",
    alpha=0.45
)

if len(episode_rewards) >= 20:

    moving_average = np.convolve(
        episode_rewards,
        np.ones(20) / 20,
        mode="valid"
    )

    plt.plot(
        range(
            19,
            len(episode_rewards)
        ),
        moving_average,
        label="20-Episode Average"
    )

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("VDN Evaluation Reward")
plt.grid()
plt.legend()

reward_plot = os.path.join(
    RESULT_DIR,
    "vdn_evaluation_reward.png"
)

plt.savefig(
    reward_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PLOT 2: TARGET TREATMENT
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    episode_treated,
    label="Targets Treated"
)

plt.xlabel("Episode")
plt.ylabel("Targets Treated")
plt.title("VDN Target Treatment")
plt.yticks(
    range(0, NUM_AGENTS + 1)
)
plt.grid()
plt.legend()

treatment_plot = os.path.join(
    RESULT_DIR,
    "vdn_evaluation_treatment.png"
)

plt.savefig(
    treatment_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PLOT 3: COVERAGE
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    episode_coverage,
    label="Coverage"
)

plt.xlabel("Episode")
plt.ylabel("Coverage")
plt.title("VDN Environment Coverage")
plt.grid()
plt.legend()

coverage_plot = os.path.join(
    RESULT_DIR,
    "vdn_evaluation_coverage.png"
)

plt.savefig(
    coverage_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PLOT 4: ENERGY
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    episode_energy,
    label="Remaining Energy"
)

plt.xlabel("Episode")
plt.ylabel("Energy")
plt.title("VDN Remaining Energy")
plt.grid()
plt.legend()

energy_plot = os.path.join(
    RESULT_DIR,
    "vdn_evaluation_energy.png"
)

plt.savefig(
    energy_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# FINAL FILE INFORMATION
# ============================================================

print()
print("=" * 60)
print("EVALUATION FILES SAVED")
print("=" * 60)
print()

print("Evaluation results:")
print(
    os.path.abspath(results_path)
)

print()

print("Best trajectory:")
print(
    os.path.abspath(trajectory_path)
)

print()

print("Plots saved in:")
print(
    os.path.abspath(RESULT_DIR)
)

print()
print("=" * 60)
print("VDN EVALUATION COMPLETE")
print("=" * 60)