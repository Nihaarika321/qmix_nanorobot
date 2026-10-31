import numpy as np
import torch

from environment.joint_env import JointInflammationEnv
from qmix.qmix_learner import QMIXLearner


print("=" * 60)
print("QMIX COMPONENT TEST")
print("=" * 60)


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

env = JointInflammationEnv()

observations = env.reset()

print("\nObservation shape:")
print(np.array(observations).shape)


# ---------------------------------------------------------
# QMIX
# ---------------------------------------------------------

learner = QMIXLearner(
    n_agents=3,
    obs_dim=20,
    action_dim=6,
    state_dim=824
)


# ---------------------------------------------------------
# Select actions
# ---------------------------------------------------------

actions = learner.select_actions(
    observations,
    epsilon=0.0
)

print("\nSelected actions:")
print(actions)


# ---------------------------------------------------------
# Test agent network
# ---------------------------------------------------------

obs_tensor = torch.tensor(
    observations,
    dtype=torch.float32
)

with torch.no_grad():

    q_values = learner.agent_network(
        obs_tensor
    )

print("\nAgent Q-value shape:")
print(q_values.shape)


# ---------------------------------------------------------
# Test mixer
# ---------------------------------------------------------

agent_qs = torch.randn(
    1,
    3
)

state = torch.randn(
    1,
    824
)

with torch.no_grad():

    q_total = learner.mixer(
        agent_qs,
        state
    )

print("\nQMIX total Q shape:")
print(q_total.shape)

print("\nQMIX total Q:")
print(q_total)


print("\n" + "=" * 60)
print("QMIX COMPONENT TEST COMPLETE")
print("=" * 60)