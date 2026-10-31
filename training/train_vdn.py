import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

from environment.joint_env import JointInflammationEnv
from qmix.vdn_agent import VDNAgent


# =============================================================
# RANDOM SEED
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

BATCH_SIZE = 64


# =============================================================
# CREATE ENVIRONMENT
# =============================================================

env = JointInflammationEnv(

    grid_size=20,

    n_agents=NUM_AGENTS,

    n_targets=3

)


# =============================================================
# CREATE VDN
# =============================================================

vdn = VDNAgent(

    n_agents=NUM_AGENTS,

    state_size=STATE_SIZE,

    action_size=ACTION_SIZE,

    learning_rate=0.001,

    gamma=0.99,

    epsilon=1.0,

    epsilon_min=0.05,

    epsilon_decay=0.995,

    buffer_size=50000,

    batch_size=BATCH_SIZE,

    target_update=100

)


# =============================================================
# RESULT ARRAYS
# =============================================================

episode_rewards = []

episode_treated = []

episode_discovered = []

episode_coverage = []

episode_losses = []


# =============================================================
# TRAINING
# =============================================================

for episode in range(
    NUM_EPISODES
):

    observations = env.reset()

    done = False

    total_reward = 0.0

    losses = []

    # =========================================================
    # EPISODE LOOP
    # =========================================================

    while not done:

        # -----------------------------------------------------
        # SELECT JOINT ACTION
        # -----------------------------------------------------

        actions = vdn.select_actions(

            observations,

            training=True

        )

        # -----------------------------------------------------
        # ENVIRONMENT STEP
        # -----------------------------------------------------

        (
            next_observations,
            reward,
            done,
            info
        ) = env.step(
            actions
        )

        # -----------------------------------------------------
        # STORE JOINT EXPERIENCE
        # -----------------------------------------------------

        vdn.remember(

            observations,

            actions,

            reward,

            next_observations,

            done

        )

        # -----------------------------------------------------
        # TRAIN VDN
        # -----------------------------------------------------

        loss = vdn.train_step()

        if loss is not None:

            losses.append(
                loss
            )

        # -----------------------------------------------------
        # UPDATE
        # -----------------------------------------------------

        observations = next_observations

        total_reward += reward

    # =========================================================
    # EPSILON DECAY
    # =========================================================

    vdn.decay_epsilon()

    # =========================================================
    # RECORD RESULTS
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

    if len(losses) > 0:

        episode_losses.append(
            np.mean(losses)
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
            f"{vdn.epsilon:.3f}"

        )


# =============================================================
# CREATE RESULT DIRECTORIES
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

vdn.save(
    "models"
)


# =============================================================
# SAVE DATA
# =============================================================

np.save(
    "results/vdn_rewards.npy",
    np.array(
        episode_rewards
    )
)

np.save(
    "results/vdn_treated.npy",
    np.array(
        episode_treated
    )
)

np.save(
    "results/vdn_discovered.npy",
    np.array(
        episode_discovered
    )
)

np.save(
    "results/vdn_coverage.npy",
    np.array(
        episode_coverage
    )
)

np.save(
    "results/vdn_losses.npy",
    np.array(
        episode_losses
    )
)

# =============================================================
# REWARD GRAPH
# =============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(

    episode_rewards,

    alpha=0.4,

    label="Episode Reward"

)

# -------------------------------------------------------------
# Moving average
# -------------------------------------------------------------

window = 20

if len(
    episode_rewards
) >= window:

    moving_average = np.convolve(

        episode_rewards,

        np.ones(window)
        /
        window,

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
    "VDN Training Reward"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(

    "results/vdn_training_reward.png",

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
    "VDN Target Treatment"
)

plt.grid()

plt.tight_layout()

plt.savefig(

    "results/vdn_target_treatment.png",

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
    "VDN Environment Coverage"
)

plt.grid()

plt.tight_layout()

plt.savefig(

    "results/vdn_coverage.png",

    dpi=300

)

plt.show()


# =============================================================
# LOSS GRAPH
# =============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    episode_losses
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Training Loss"
)

plt.title(
    "VDN Training Loss"
)

plt.grid()

plt.tight_layout()

plt.savefig(

    "results/vdn_training_loss.png",

    dpi=300

)

plt.show()


# =============================================================
# FINAL RESULTS
# =============================================================

print(
    "\n=========================================="
)

print(
    "VDN TRAINING COMPLETE"
)

print(
    "=========================================="
)

print(
    f"Final epsilon: "
    f"{vdn.epsilon:.4f}"
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
    f"Average targets discovered (last 50): "
    f"{np.mean(episode_discovered[-50:]):.2f}"
)

print(
    f"Average coverage (last 50): "
    f"{np.mean(episode_coverage[-50:]):.3f}"
)

print(
    f"Average loss (last 50): "
    f"{np.mean(episode_losses[-50:]):.4f}"
)