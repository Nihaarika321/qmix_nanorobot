import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from environment.joint_env import JointInflammationEnv
from agents.dqn_agent import DQNAgent


# =============================================================
# REPRODUCIBILITY
# =============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================
# PARAMETERS
# =============================================================

NUM_EPISODES = 500

NUM_AGENTS = 3

STATE_SIZE = 20

ACTION_SIZE = 6

# =============================================================
# CREATE ENVIRONMENT
# =============================================================

env = JointInflammationEnv(
    grid_size=20,
    n_agents=NUM_AGENTS,
    n_targets=3
)


# =============================================================
# CREATE THREE INDEPENDENT DQN AGENTS
# =============================================================

agents = [

    DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE
    )

    for _ in range(NUM_AGENTS)

]


# =============================================================
# RESULT STORAGE
# =============================================================

episode_rewards = []

episode_treated = []

episode_discovered = []

episode_coverage = []

episode_losses = []


# =============================================================
# TRAINING LOOP
# =============================================================

for episode in range(
    NUM_EPISODES
):

    observations = env.reset()

    total_reward = 0.0

    total_loss = []

    done = False

    # =========================================================
    # EPISODE
    # =========================================================

    while not done:

        actions = []

        # -----------------------------------------------------
        # Each robot independently selects an action
        # -----------------------------------------------------

        for i in range(NUM_AGENTS):

            action = agents[i].select_action(
                observations[i],
                training=True
            )

            actions.append(
                action
            )

        # -----------------------------------------------------
        # Environment transition
        # -----------------------------------------------------

        next_observations, reward, done, info = env.step(
            actions
        )

        # -----------------------------------------------------
        # Store experience for every robot
        # -----------------------------------------------------

        for i in range(NUM_AGENTS):

            agents[i].remember(

                observations[i],

                actions[i],

                reward,

                next_observations[i],

                done

            )

            # -------------------------------------------------
            # Train agent
            # -------------------------------------------------

            loss = agents[i].train_step()

            if loss is not None:

                total_loss.append(
                    loss
                )

        # -----------------------------------------------------
        # Update
        # -----------------------------------------------------

        observations = next_observations

        total_reward += reward

    # =========================================================
    # EPSILON DECAY
    # =========================================================

    for agent in agents:

        agent.decay_epsilon()

    # =========================================================
    # SAVE METRICS
    # =========================================================

    episode_rewards.append(
        total_reward
    )

    episode_treated.append(
        info["targets_treated"]
    )

    episode_discovered.append(
        info["targets_discovered"]
    )

    episode_coverage.append(
        info["coverage"]
    )

    if len(total_loss) > 0:

        episode_losses.append(
            np.mean(total_loss)
        )

    else:

        episode_losses.append(
            0.0
        )

    # =========================================================
    # PRINT PROGRESS
    # =========================================================

    if (
        episode + 1
    ) % 10 == 0:

        print(

            f"Episode {episode + 1:4d}/{NUM_EPISODES} | "

            f"Reward: {total_reward:8.2f} | "

            f"Treated: "
            f"{info['targets_treated']}/3 | "

            f"Discovered: "
            f"{info['targets_discovered']}/3 | "

            f"Coverage: "
            f"{info['coverage']:.3f} | "

            f"Epsilon: "
            f"{agents[0].epsilon:.3f}"

        )


# =============================================================
# CREATE DIRECTORIES
# =============================================================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "results",
    exist_ok=True
)


# =============================================================
# SAVE MODELS
# =============================================================

for i, agent in enumerate(agents):

    agent.save(
        f"models/iql_agent_{i + 1}.pth"
    )


# =============================================================
# SAVE RESULTS
# =============================================================

np.save(
    "results/iql_rewards.npy",
    np.array(episode_rewards)
)

np.save(
    "results/iql_treated.npy",
    np.array(episode_treated)
)

np.save(
    "results/iql_discovered.npy",
    np.array(episode_discovered)
)

np.save(
    "results/iql_coverage.npy",
    np.array(episode_coverage)
)

np.save(
    "results/iql_losses.npy",
    np.array(episode_losses)
)


# =============================================================
# TRAINING REWARD GRAPH
# =============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    episode_rewards,
    alpha=0.4,
    label="Episode Reward"
)

# Moving average
window = 20

if len(episode_rewards) >= window:

    moving_average = np.convolve(
        episode_rewards,
        np.ones(window) / window,
        mode="valid"
    )

    plt.plot(
        range(
            window - 1,
            len(episode_rewards)
        ),
        moving_average,
        label="20-Episode Average"
    )

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Total Reward"
)

plt.title(
    "Independent DQN Training"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    "results/iql_training_reward.png",
    dpi=300
)

plt.show()


# =============================================================
# TARGET TREATMENT GRAPH
# =============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    episode_treated
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Targets Treated"
)

plt.title(
    "Target Treatment During Training"
)

plt.grid()

plt.tight_layout()

plt.savefig(
    "results/iql_target_treatment.png",
    dpi=300
)

plt.show()


# =============================================================
# COVERAGE GRAPH
# =============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    episode_coverage
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Coverage"
)

plt.title(
    "Environment Coverage During Training"
)

plt.grid()

plt.tight_layout()

plt.savefig(
    "results/iql_coverage.png",
    dpi=300
)

plt.show()


print("\n==========================================")
print("IQL TRAINING COMPLETE")
print("==========================================")

print(
    f"Final epsilon: "
    f"{agents[0].epsilon:.4f}"
)

print(
    f"Average reward (last 50): "
    f"{np.mean(episode_rewards[-50:]):.2f}"
)

print(
    f"Average targets treated (last 50): "
    f"{np.mean(episode_treated[-50:]):.2f}"
)

print(
    f"Average coverage (last 50): "
    f"{np.mean(episode_coverage[-50:]):.3f}"
)