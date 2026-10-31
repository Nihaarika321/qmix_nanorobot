import numpy as np


class JointInflammationEnv:

    def __init__(
        self,
        grid_size=20,
        n_agents=3,
        n_targets=3
    ):

        self.grid_size = grid_size
        self.n_agents = n_agents
        self.n_targets = n_targets

        # =====================================================
        # ACTIONS
        # =====================================================

        self.STAY = 0
        self.UP = 1
        self.DOWN = 2
        self.LEFT = 3
        self.RIGHT = 4
        self.TREAT = 5

        self.n_actions = 6

        # =====================================================
        # ENVIRONMENT PARAMETERS
        # =====================================================

        self.max_steps = 200
        self.max_energy = 100.0

        # Limited sensing radius
        self.sensor_radius = 4

        # =====================================================
        # REWARDS
        # =====================================================

        self.reward_target_found = 10.0
        self.reward_treatment = 30.0
        self.reward_all_treated = 100.0
        self.reward_exploration = 0.5

        self.penalty_collision = -5.0
        self.penalty_energy = -0.1
        self.penalty_redundant = -0.5
        self.penalty_invalid_treatment = -3.0
        self.penalty_time = -0.05

        # Create initial environment
        self.reset()

    # =========================================================
    # RESET ENVIRONMENT
    # =========================================================

    def reset(self):

        self.step_count = 0

        # -----------------------------------------------------
        # Grid
        # 0 = free
        # 1 = obstacle
        # -----------------------------------------------------

        self.grid = np.zeros(
            (self.grid_size, self.grid_size),
            dtype=np.int32
        )

        # -----------------------------------------------------
        # Visited cells
        # -----------------------------------------------------

        self.visited = np.zeros(
            (self.grid_size, self.grid_size),
            dtype=np.int32
        )

        # -----------------------------------------------------
        # Robot positions
        # -----------------------------------------------------

        self.agent_positions = []

        # -----------------------------------------------------
        # Target positions
        # -----------------------------------------------------

        self.target_positions = []

        # -----------------------------------------------------
        # Target treatment status
        #
        # 0 = untreated
        # 1 = treated
        # -----------------------------------------------------

        self.target_treated = np.zeros(
            self.n_targets,
            dtype=np.int32
        )

        # -----------------------------------------------------
        # Target discovery status
        #
        # 0 = not discovered
        # 1 = discovered
        # -----------------------------------------------------

        self.target_discovered = np.zeros(
            self.n_targets,
            dtype=np.int32
        )

        # -----------------------------------------------------
        # Inflammation intensity
        # -----------------------------------------------------

        self.target_intensity = np.ones(
            self.n_targets,
            dtype=np.float32
        )

        # -----------------------------------------------------
        # Robot energy
        # -----------------------------------------------------

        self.energy = np.ones(
            self.n_agents,
            dtype=np.float32
        ) * self.max_energy

        # =====================================================
        # GENERATE ENVIRONMENT
        # =====================================================

        self._generate_obstacles()

        self._place_targets()

        self._place_agents()

        # =====================================================
        # MARK INITIAL ROBOT CELLS AS VISITED
        # =====================================================

        for x, y in self.agent_positions:

            self.visited[y, x] = 1

        # =====================================================
        # RETURN INITIAL OBSERVATIONS
        # =====================================================

        return self.get_observations()

    # =========================================================
    # GENERATE OBSTACLES
    # =========================================================

    def _generate_obstacles(self):

        obstacle_count = 30

        placed = 0

        while placed < obstacle_count:

            x = np.random.randint(
                1,
                self.grid_size - 1
            )

            y = np.random.randint(
                1,
                self.grid_size - 1
            )

            if self.grid[y, x] == 0:

                self.grid[y, x] = 1

                placed += 1

    # =========================================================
    # PLACE INFLAMMATION TARGETS
    # =========================================================

    def _place_targets(self):

        while len(self.target_positions) < self.n_targets:

            x = np.random.randint(
                1,
                self.grid_size - 1
            )

            y = np.random.randint(
                1,
                self.grid_size - 1
            )

            # Cannot place target on obstacle
            if self.grid[y, x] != 0:
                continue

            # Cannot overlap existing target
            if (x, y) in self.target_positions:
                continue

            # Keep targets separated
            too_close = False

            for tx, ty in self.target_positions:

                distance = abs(x - tx) + abs(y - ty)

                if distance < 4:

                    too_close = True

                    break

            if not too_close:

                self.target_positions.append(
                    (x, y)
                )

    # =========================================================
    # PLACE NANOROBOTS
    # =========================================================

    def _place_agents(self):

        while len(self.agent_positions) < self.n_agents:

            x = np.random.randint(
                0,
                self.grid_size
            )

            y = np.random.randint(
                0,
                self.grid_size
            )

            # Cannot place robot on obstacle
            if self.grid[y, x] != 0:
                continue

            # Cannot place robot on target
            if (x, y) in self.target_positions:
                continue

            # Cannot overlap another robot
            if (x, y) in self.agent_positions:
                continue

            self.agent_positions.append(
                (x, y)
            )

    # =========================================================
    # ENVIRONMENT STEP
    # =========================================================

    def step(self, actions):

        self.step_count += 1

        total_reward = 0.0

        old_positions = self.agent_positions.copy()

        new_positions = []

        # =====================================================
        # PROCESS EACH ROBOT ACTION
        # =====================================================

        for i, action in enumerate(actions):

            x, y = self.agent_positions[i]

            new_x = x
            new_y = y

            # -------------------------------------------------
            # UP
            # -------------------------------------------------

            if action == self.UP:

                new_y -= 1

            # -------------------------------------------------
            # DOWN
            # -------------------------------------------------

            elif action == self.DOWN:

                new_y += 1

            # -------------------------------------------------
            # LEFT
            # -------------------------------------------------

            elif action == self.LEFT:

                new_x -= 1

            # -------------------------------------------------
            # RIGHT
            # -------------------------------------------------

            elif action == self.RIGHT:

                new_x += 1

            # -------------------------------------------------
            # TREAT
            # -------------------------------------------------

            elif action == self.TREAT:

                reward = self._treat_target(i)

                total_reward += reward

                new_positions.append(
                    (x, y)
                )

                continue

            # -------------------------------------------------
            # STAY
            # -------------------------------------------------

            elif action == self.STAY:

                new_positions.append(
                    (x, y)
                )

                continue

            # -------------------------------------------------
            # INVALID ACTION
            # -------------------------------------------------

            else:

                new_positions.append(
                    (x, y)
                )

                total_reward += self.penalty_collision

                continue

            # =================================================
            # CHECK GRID BOUNDARY
            # =================================================

            if (
                new_x < 0
                or new_x >= self.grid_size
                or new_y < 0
                or new_y >= self.grid_size
            ):

                total_reward += self.penalty_collision

                new_positions.append(
                    (x, y)
                )

                continue

            # =================================================
            # CHECK OBSTACLE COLLISION
            # =================================================

            if self.grid[new_y, new_x] == 1:

                total_reward += self.penalty_collision

                new_positions.append(
                    (x, y)
                )

                continue

            # =================================================
            # VALID MOVEMENT
            # =================================================

            new_positions.append(
                (new_x, new_y)
            )

        # =====================================================
        # ROBOT-ROBOT COLLISION
        # =====================================================

        for i in range(self.n_agents):

            for j in range(i + 1, self.n_agents):

                if new_positions[i] == new_positions[j]:

                    total_reward += self.penalty_collision

                    new_positions[i] = old_positions[i]

                    new_positions[j] = old_positions[j]

        self.agent_positions = new_positions

        # =====================================================
        # TARGET DISCOVERY
        # =====================================================

        for target_id, (tx, ty) in enumerate(
            self.target_positions
        ):

            # Already treated targets don't need discovery
            if self.target_treated[target_id] == 1:

                continue

            for x, y in self.agent_positions:

                distance = abs(x - tx) + abs(y - ty)

                # Target is inside sensing radius
                if distance <= self.sensor_radius:

                    # Give discovery reward only once
                    if self.target_discovered[target_id] == 0:

                        self.target_discovered[target_id] = 1

                        total_reward += (
                            self.reward_target_found
                        )

                    break

        # =====================================================
        # EXPLORATION REWARD
        # =====================================================

        for x, y in self.agent_positions:

            # New cell
            if self.visited[y, x] == 0:

                total_reward += (
                    self.reward_exploration
                )

                self.visited[y, x] = 1

            # Already visited cell
            else:

                total_reward += (
                    self.penalty_redundant
                )

        # =====================================================
        # ENERGY CONSUMPTION
        # =====================================================

        for i, action in enumerate(actions):

            # -------------------------------------------------
            # Movement costs 1 energy
            # -------------------------------------------------

            if action in [
                self.UP,
                self.DOWN,
                self.LEFT,
                self.RIGHT
            ]:

                self.energy[i] -= 1.0

                total_reward += (
                    self.penalty_energy
                )

            # -------------------------------------------------
            # Treatment costs 5 energy
            # -------------------------------------------------

            elif action == self.TREAT:

                self.energy[i] -= 5.0

                total_reward += (
                    5 * self.penalty_energy
                )

            # -------------------------------------------------
            # Staying costs 0.2 energy
            # -------------------------------------------------

            else:

                self.energy[i] -= 0.2

                total_reward += (
                    0.2 * self.penalty_energy
                )

        # =====================================================
        # IMPORTANT:
        # PREVENT ENERGY FROM BECOMING NEGATIVE
        # =====================================================

        self.energy = np.maximum(
            self.energy,
            0.0
        )

        # =====================================================
        # TIME PENALTY
        # =====================================================

        total_reward += self.penalty_time

        # =====================================================
        # TERMINATION CONDITIONS
        # =====================================================

        # All inflammation sites treated
        all_treated = np.all(
            self.target_treated == 1
        )

        # All robots have no energy
        out_of_energy = np.all(
            self.energy <= 0
        )

        # Maximum episode length reached
        timeout = (
            self.step_count >= self.max_steps
        )

        done = (
            all_treated
            or out_of_energy
            or timeout
        )

        # =====================================================
        # COMPLETION BONUS
        # =====================================================

        if all_treated:

            total_reward += (
                self.reward_all_treated
            )

        # =====================================================
        # GET NEXT OBSERVATIONS
        # =====================================================

        observations = self.get_observations()

        # =====================================================
        # INFORMATION FOR EVALUATION
        # =====================================================

        info = {

            "all_treated":
                bool(all_treated),

            "targets_discovered":
                int(
                    np.sum(
                        self.target_discovered
                    )
                ),

            "targets_treated":
                int(
                    np.sum(
                        self.target_treated
                    )
                ),

            "step":
                self.step_count,

            "energy":
                self.energy.copy(),

            "coverage":
                (
                    np.sum(self.visited)
                    /
                    (self.grid_size * self.grid_size)
                )
        }

        return (
            observations,
            total_reward,
            done,
            info
        )

    # =========================================================
    # TREAT TARGET
    # =========================================================

    def _treat_target(self, agent_id):

        x, y = self.agent_positions[agent_id]

        reward = 0.0

        for i, (tx, ty) in enumerate(
            self.target_positions
        ):

            distance = (
                abs(x - tx)
                +
                abs(y - ty)
            )

            # Robot must be on or next to target
            if distance <= 1:

                # Target has not been treated
                if self.target_treated[i] == 0:

                    self.target_treated[i] = 1

                    self.target_intensity[i] = 0.0

                    reward += (
                        self.reward_treatment
                    )

                # Target already treated
                else:

                    reward += (
                        self.penalty_invalid_treatment
                    )

                return reward

        # No target close enough
        reward += (
            self.penalty_invalid_treatment
        )

        return reward

    # =========================================================
    # GET LOCAL OBSERVATIONS
    # =========================================================

    def get_observations(self):

        observations = []

        for agent_id in range(
            self.n_agents
        ):

            x, y = self.agent_positions[
                agent_id
            ]

            # =================================================
            # FIND NEAREST VISIBLE TARGET
            # =================================================

            target_visible = 0

            target_dx = 0.0

            target_dy = 0.0

            target_distance = 0.0

            target_intensity = 0.0

            nearest_distance = (
                self.sensor_radius + 1
            )

            for i, (tx, ty) in enumerate(
                self.target_positions
            ):

                # Ignore treated targets
                if self.target_treated[i] == 1:

                    continue

                distance = (
                    abs(x - tx)
                    +
                    abs(y - ty)
                )

                # Target is visible
                if distance <= self.sensor_radius:

                    if distance < nearest_distance:

                        nearest_distance = distance

                        target_visible = 1

                        target_dx = (
                            tx - x
                        )

                        target_dy = (
                            ty - y
                        )

                        target_distance = (
                            distance
                            /
                            self.grid_size
                        )

                        target_intensity = (
                            self.target_intensity[i]
                        )

            # =================================================
            # LOCAL OBSTACLES
            # =================================================

            obstacle_up = self._cell_blocked(
                x,
                y - 1
            )

            obstacle_down = self._cell_blocked(
                x,
                y + 1
            )

            obstacle_left = self._cell_blocked(
                x - 1,
                y
            )

            obstacle_right = self._cell_blocked(
                x + 1,
                y
            )

            # =================================================
            # NEARBY ROBOTS
            # =================================================

            robot_up = 0

            robot_down = 0

            robot_left = 0

            robot_right = 0

            for j in range(
                self.n_agents
            ):

                if j == agent_id:

                    continue

                rx, ry = self.agent_positions[j]

                if (
                    rx == x
                    and
                    ry == y - 1
                ):

                    robot_up = 1

                if (
                    rx == x
                    and
                    ry == y + 1
                ):

                    robot_down = 1

                if (
                    rx == x - 1
                    and
                    ry == y
                ):

                    robot_left = 1

                if (
                    rx == x + 1
                    and
                    ry == y
                ):

                    robot_right = 1

            # =================================================
            # VISITED INFORMATION
            # =================================================

            visited_up = self._cell_visited(
                x,
                y - 1
            )

            visited_down = self._cell_visited(
                x,
                y + 1
            )

            visited_left = self._cell_visited(
                x - 1,
                y
            )

            visited_right = self._cell_visited(
                x + 1,
                y
            )

            # =================================================
            # CREATE OBSERVATION VECTOR
            # =================================================

            obs = np.array([

                # -----------------------------
                # Robot position
                # -----------------------------

                x / (self.grid_size - 1),

                y / (self.grid_size - 1),

                # -----------------------------
                # Energy
                # -----------------------------

                self.energy[agent_id]
                /
                self.max_energy,

                # -----------------------------
                # Target sensing
                # -----------------------------

                target_visible,

                target_dx
                /
                self.grid_size,

                target_dy
                /
                self.grid_size,

                target_distance,

                target_intensity,

                # -----------------------------
                # Obstacles
                # -----------------------------

                obstacle_up,

                obstacle_down,

                obstacle_left,

                obstacle_right,

                # -----------------------------
                # Nearby robots
                # -----------------------------

                robot_up,

                robot_down,

                robot_left,

                robot_right,

                # -----------------------------
                # Visited cells
                # -----------------------------

                visited_up,

                visited_down,

                visited_left,

                visited_right

            ], dtype=np.float32)

            observations.append(
                obs
            )

        return np.array(
            observations,
            dtype=np.float32
        )

    # =========================================================
    # GLOBAL STATE
    #
    # Used by QMIX mixer during training.
    #
    # Individual robots DO NOT receive this.
    # =========================================================

    def get_global_state(self):

        state = []

        # =====================================================
        # ROBOT POSITIONS
        # =====================================================

        for x, y in self.agent_positions:

            state.append(
                x / (self.grid_size - 1)
            )

            state.append(
                y / (self.grid_size - 1)
            )

        # =====================================================
        # ROBOT ENERGY
        # =====================================================

        for e in self.energy:

            state.append(
                e / self.max_energy
            )

        # =====================================================
        # TARGET INFORMATION
        # =====================================================

        for i, (x, y) in enumerate(
            self.target_positions
        ):

            state.append(
                x / (self.grid_size - 1)
            )

            state.append(
                y / (self.grid_size - 1)
            )

            state.append(
                self.target_treated[i]
            )

            state.append(
                self.target_discovered[i]
            )

            state.append(
                self.target_intensity[i]
            )

        # =====================================================
        # GLOBAL OBSTACLE MAP
        # =====================================================

        state.extend(
            self.grid.flatten()
        )

        # =====================================================
        # GLOBAL VISITED MAP
        # =====================================================

        state.extend(
            self.visited.flatten()
        )

        return np.array(
            state,
            dtype=np.float32
        )

    # =========================================================
    # CHECK WHETHER CELL IS BLOCKED
    # =========================================================

    def _cell_blocked(self, x, y):

        # Outside grid = blocked
        if (
            x < 0
            or x >= self.grid_size
            or y < 0
            or y >= self.grid_size
        ):

            return 1

        return int(
            self.grid[y, x] == 1
        )

    # =========================================================
    # CHECK WHETHER CELL WAS VISITED
    # =========================================================

    def _cell_visited(self, x, y):

        # Outside grid considered visited
        if (
            x < 0
            or x >= self.grid_size
            or y < 0
            or y >= self.grid_size
        ):

            return 1

        return int(
            self.visited[y, x]
        )

    # =========================================================
    # RENDER ENVIRONMENT
    #
    # Values:
    # 0 = free space
    # 1 = obstacle
    # 2 = untreated inflammation
    # 3 = treated inflammation
    # 4+ = nanorobots
    # =========================================================

    def render(self):

        display = np.zeros(
            (
                self.grid_size,
                self.grid_size
            ),
            dtype=np.int32
        )

        # -----------------------------------------------------
        # Obstacles
        # -----------------------------------------------------

        display[
            self.grid == 1
        ] = 1

        # -----------------------------------------------------
        # Targets
        # -----------------------------------------------------

        for i, (x, y) in enumerate(
            self.target_positions
        ):

            if self.target_treated[i] == 0:

                display[y, x] = 2

            else:

                display[y, x] = 3

        # -----------------------------------------------------
        # Nanorobots
        # -----------------------------------------------------

        for i, (x, y) in enumerate(
            self.agent_positions
        ):

            display[y, x] = 4 + i

        return display