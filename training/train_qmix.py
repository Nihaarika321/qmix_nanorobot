import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Make project root available
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

from environment.joint_env import JointInflammationEnv

from qmix.qmix_learner import QMIXLearner
from qmix.replay_buffer import QMIXReplayBuffer


# =========================================================
# CONFIGURATION
# =========================================================

NUM_EPISODES = 2000

BATCH_SIZE = 64

BUFFER_SIZE = 50000

MIN_BUFFER_SIZE = 500

TARGET_UPDATE_FREQUENCY = 20

GAMMA = 0.99

LEARNING_RATE = 0.0005

EPSILON_START = 1.0

EPSILON_END = 0.05

EPSILON_DECAY = 0.995

SEED = 42


# =========================================================
# RANDOM SEEDS
# =========================================================

np.random.seed(SEED)


# =========================================================
# ENVIRONMENT
# =========================================================

env = JointInflammationEnv()


# =========================================================
# QMIX LEARNER
# =========================================================

learner = QMIXLearner(
    n_agents=3,
    obs_dim=20,
    action_dim=6,
    state_dim=824,
    hidden_dim=64,
    mixing_embed_dim=32,
    lr=LEARNING_RATE,
    gamma=GAMMA
)


# =========================================================
# REPLAY BUFFER
# =========================================================

replay_buffer = QMIXReplayBuffer(
    capacity=BUFFER_SIZE
)


# =========================================================
# RESULT STORAGE
# =========================================================

episode_rewards = []

episode_losses = []

episode_treated = []

episode_discovered = []

episode_coverage = []

episode_energy = []

episode_steps = []

epsilon_history = []


# =========================================================
# EPSILON
# =========================================================

epsilon = EPSILON_START


# =========================================================
# CREATE RESULTS DIRECTORY
# =========================================================

results_dir = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(
    results_dir,
    exist_ok=True
)


# =========================================================
# TRAINING
# =========================================================

print("=" * 60)
print("QMIX TRAINING")
print("=" * 60)

print()

for episode in range(1, NUM_EPISODES + 1):

    # -----------------------------------------------------
    # Reset environment
    # -----------------------------------------------------

    observations = env.reset()

    # -----------------------------------------------------
    # Obtain global state
    # -----------------------------------------------------

    state = env.get_global_state()

    done = False

    total_reward = 0.0

    losses = []

    step_count = 0

    # -----------------------------------------------------
    # Episode loop
    # -----------------------------------------------------

    while not done:

        step_count += 1

        # -------------------------------------------------
        # Select joint action
        # -------------------------------------------------

        actions = learner.select_actions(
            observations,
            epsilon
        )

        # -------------------------------------------------
        # Environment step
        # -------------------------------------------------

        result = env.step(actions)

        # -------------------------------------------------
        # Handle Gym-style / custom return
        # -------------------------------------------------

        if len(result) == 5:

            next_observations, reward, terminated, truncated, info = result

            done = terminated or truncated

        else:

            next_observations, reward, done, info = result

        # -------------------------------------------------
        # Next global state
        # -------------------------------------------------

        next_state = env.get_global_state()

        # -------------------------------------------------
        # Store experience
        # -------------------------------------------------

        replay_buffer.add(
            observations=observations,
            actions=actions,
            reward=reward,
            next_observations=next_observations,
            state=state,
            next_state=next_state,
            done=done
        )

        # -------------------------------------------------
        # Update state
        # -------------------------------------------------

        observations = next_observations

        state = next_state

        total_reward += reward

        # -------------------------------------------------
        # Train QMIX
        # -------------------------------------------------

        if len(replay_buffer) >= MIN_BUFFER_SIZE:

            batch = replay_buffer.sample(
                BATCH_SIZE
            )

            loss = learner.train_batch(
                batch
            )

            losses.append(loss)

    # =====================================================
    # END OF EPISODE
    # =====================================================

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    if hasattr(env, "targets_treated"):

        treated = int(
            np.sum(env.targets_treated)
        )

    elif hasattr(env, "target_treated"):

        treated = int(
            np.sum(env.target_treated)
        )

    else:

        treated = 0


    if hasattr(env, "targets_discovered"):

        discovered = int(
            np.sum(env.targets_discovered)
        )

    elif hasattr(env, "target_discovered"):

        discovered = int(
            np.sum(env.target_discovered)
        )

    else:

        discovered = 0


    # -----------------------------------------------------
    # Coverage
    # -----------------------------------------------------

    if hasattr(env, "coverage"):

        coverage = float(
            env.coverage
        )

    elif hasattr(env, "visited"):

        coverage = float(
            np.mean(env.visited)
        )

    else:

        coverage = 0.0


    # -----------------------------------------------------
    # Energy
    # -----------------------------------------------------

    if hasattr(env, "robot_energy"):

        energy = float(
            np.mean(env.robot_energy)
        )

    elif hasattr(env, "energy"):

        energy = float(
            np.mean(env.energy)
        )

    else:

        energy = 0.0


    # -----------------------------------------------------
    # Average loss
    # -----------------------------------------------------

    if len(losses) > 0:

        avg_loss = float(
            np.mean(losses)
        )

    else:

        avg_loss = 0.0


    # -----------------------------------------------------
    # Store statistics
    # -----------------------------------------------------

    episode_rewards.append(
        total_reward
    )

    episode_losses.append(
        avg_loss
    )

    episode_treated.append(
        treated
    )

    episode_discovered.append(
        discovered
    )

    episode_coverage.append(
        coverage
    )

    episode_energy.append(
        energy
    )

    episode_steps.append(
        step_count
    )

    epsilon_history.append(
        epsilon
    )


    # =====================================================
    # EPSILON DECAY
    # =====================================================

    epsilon = max(
        EPSILON_END,
        epsilon * EPSILON_DECAY
    )


    # =====================================================
    # TARGET NETWORK UPDATE
    # =====================================================

    if episode % TARGET_UPDATE_FREQUENCY == 0:

        learner.update_target_networks()


    # =====================================================
    # PRINT PROGRESS
    # =====================================================

    if episode % 10 == 0 or episode == 1:

        print(
            f"Episode {episode:3d}/{NUM_EPISODES} | "
            f"Reward: {total_reward:8.2f} | "
            f"Treated: {treated}/3 | "
            f"Discovered: {discovered}/3 | "
            f"Coverage: {coverage:.3f} | "
            f"Epsilon: {epsilon:.3f}"
        )


# =========================================================
# TRAINING COMPLETE
# =========================================================

print()
print("=" * 60)
print("QMIX TRAINING COMPLETE")
print("=" * 60)


# =========================================================
# FINAL STATISTICS
# =========================================================

last_n = min(
    50,
    len(episode_rewards)
)

print(
    f"Final epsilon: {epsilon:.4f}"
)

print(
    f"Average reward (last {last_n}): "
    f"{np.mean(episode_rewards[-last_n:]):.2f}"
)

print(
    f"Average targets treated (last {last_n}): "
    f"{np.mean(episode_treated[-last_n:]):.2f}"
)

print(
    f"Average targets discovered (last {last_n}): "
    f"{np.mean(episode_discovered[-last_n:]):.2f}"
)

print(
    f"Average coverage (last {last_n}): "
    f"{np.mean(episode_coverage[-last_n:]):.3f}"
)

print(
    f"Average energy (last {last_n}): "
    f"{np.mean(episode_energy[-last_n:]):.2f}"
)

print(
    f"Average episode steps (last {last_n}): "
    f"{np.mean(episode_steps[-last_n:]):.2f}"
)

print(
    f"Average loss (last {last_n}): "
    f"{np.mean(episode_losses[-last_n:]):.4f}"
)


# =========================================================
# SAVE MODEL
# =========================================================

model_path = os.path.join(
    results_dir,
    "qmix_nanobot.pth"
)

learner.save(
    model_path
)

print()
print("Model saved to:")
print(model_path)


# =========================================================
# SAVE NUMERICAL RESULTS
# =========================================================

np.savez(
    os.path.join(
        results_dir,
        "qmix_training_results.npz"
    ),

    rewards=np.array(
        episode_rewards
    ),

    losses=np.array(
        episode_losses
    ),

    treated=np.array(
        episode_treated
    ),

    discovered=np.array(
        episode_discovered
    ),

    coverage=np.array(
        episode_coverage
    ),

    energy=np.array(
        episode_energy
    ),

    steps=np.array(
        episode_steps
    ),

    epsilon=np.array(
        epsilon_history
    )
)


# =========================================================
# PLOT 1 — REWARD
# =========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    episode_rewards,
    alpha=0.35,
    label="Episode Reward"
)

if len(episode_rewards) >= 20:

    moving_average = np.convolve(
        episode_rewards,
        np.ones(20) / 20,
        mode="valid"
    )

    plt.plot(
        range(19, len(episode_rewards)),
        moving_average,
        label="20-Episode Average"
    )

plt.xlabel("Episode")

plt.ylabel("Total Reward")

plt.title("QMIX Training Reward")

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        results_dir,
        "qmix_reward.png"
    )
)

plt.show()


# =========================================================
# PLOT 2 — TARGET TREATMENT
# =========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    episode_treated
)

plt.xlabel("Episode")

plt.ylabel("Targets Treated")

plt.title("QMIX Target Treatment")

plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        results_dir,
        "qmix_treatment.png"
    )
)

plt.show()


# =========================================================
# PLOT 3 — TARGET DISCOVERY
# =========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    episode_discovered
)

plt.xlabel("Episode")

plt.ylabel("Targets Discovered")

plt.title("QMIX Target Discovery")

plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        results_dir,
        "qmix_discovery.png"
    )
)

plt.show()


# =========================================================
# PLOT 4 — LOSS
# =========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    episode_losses
)

plt.xlabel("Episode")

plt.ylabel("Training Loss")

plt.title("QMIX Training Loss")

plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        results_dir,
        "qmix_loss.png"
    )
)

plt.show()


# =========================================================
# PLOT 5 — COVERAGE
# =========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    episode_coverage
)

plt.xlabel("Episode")

plt.ylabel("Coverage")

plt.title("QMIX Environment Coverage")

plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        results_dir,
        "qmix_coverage.png"
    )
)

plt.show()


# =========================================================
# PLOT 6 — ENERGY
# =========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    episode_energy
)

plt.xlabel("Episode")

plt.ylabel("Average Remaining Energy")

plt.title("QMIX Robot Energy")

plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        results_dir,
        "qmix_energy.png"
    )
)

plt.show()


print()
print("=" * 60)
print("ALL QMIX RESULTS SAVED")
print("=" * 60)