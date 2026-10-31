import numpy as np
import matplotlib.pyplot as plt

from environment.joint_env import JointInflammationEnv


# =========================================================
# CREATE ENVIRONMENT
# =========================================================

env = JointInflammationEnv(
    grid_size=20,
    n_agents=3,
    n_targets=3
)


# =========================================================
# RESET
# =========================================================

observations = env.reset()


print("=" * 60)
print("NANOROBOT JOINT ENVIRONMENT")
print("=" * 60)

print("\nObservation shape:")
print(observations.shape)

print("\nObservation size per robot:")
print(observations.shape[1])

print("\nGlobal state size:")
print(env.get_global_state().shape)

print("\nRobot positions:")
print(env.agent_positions)

print("\nTarget positions:")
print(env.target_positions)

print("\nTarget discovered:")
print(env.target_discovered)

print("\nTarget treated:")
print(env.target_treated)

print("\nRobot energy:")
print(env.energy)


# =========================================================
# SHOW INITIAL OBSERVATIONS
# =========================================================

print("\n" + "=" * 60)
print("INITIAL LOCAL OBSERVATIONS")
print("=" * 60)

for i in range(env.n_agents):

    print(
        f"\nRobot {i + 1}:"
    )

    print(
        observations[i]
    )


# =========================================================
# RUN RANDOM EPISODE
# =========================================================

done = False

total_reward = 0.0

while not done:

    actions = np.random.randint(
        0,
        env.n_actions,
        size=env.n_agents
    )

    observations, reward, done, info = env.step(
        actions
    )

    total_reward += reward


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 60)
print("RANDOM EPISODE RESULTS")
print("=" * 60)

print(
    "\nTotal reward:",
    total_reward
)

print(
    "Steps:",
    info["step"]
)

print(
    "Targets discovered:",
    info["targets_discovered"]
)

print(
    "Targets treated:",
    info["targets_treated"]
)

print(
    "Coverage:",
    info["coverage"]
)

print(
    "Final energy:",
    info["energy"]
)


# =========================================================
# VISUALIZATION
# =========================================================

grid = env.render()

plt.figure(
    figsize=(8, 8)
)

plt.imshow(
    grid,
    interpolation="nearest"
)

plt.title(
    "Nanorobot Joint Environment"
)

plt.xlabel("X")

plt.ylabel("Y")

plt.xticks(
    range(env.grid_size)
)

plt.yticks(
    range(env.grid_size)
)

plt.grid()

plt.show()