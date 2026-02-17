import numpy as np

from evorob.world.robot.controllers.base import Controller


def RK45(state, A, dt):
    """Runge-Kutta 4th/5th order integration."""
    A1 = np.matmul(A, state)
    A2 = np.matmul(A, (state + dt / 2 * A1))
    A3 = np.matmul(A, (state + dt / 2 * A2))
    A4 = np.matmul(A, (state + dt * A3))
    return state + dt / 6 * (A1 + 2 * (A2 + A3) + A4)


class SO2Controller(Controller):

    def __init__(self, input_size: int,  output_size: int, hidden_size: int):
        """
        SO2 oscillator Controller. [https://www.nature.com/articles/s41467-024-50131-4]
        - Uses internal oscillators coupled via a weight matrix.
        - Ignores external observation state (open-loop), relying on internal time-integration.
        """
        num_dofs = output_size
        dt = 0.05
        inter_con_density = 0.5
        self.controller_type = "SO2"
        self.dt = dt
        self.num_dofs = num_dofs

        # Initialize network structure (consistent randomness)
        weight_matrix, weight_map, weights = self.initalise_network(num_dofs, inter_con_density)

        self.A = weight_matrix
        self.weight_map = weight_map
        self.weights = weights

        # Dimensions
        self.n_weights = len(self.weights)
        self.n_params = len(self.weights) + num_dofs * 2

        # For compatibility with Controller inspection
        self.n_input = num_dofs * 2  # Assuming observation space, though unused by CPG logic
        self.n_output = num_dofs  # Actions

        # Internal state shape: (2*N, 1) or (2*N, Batch)
        self.state_shape = (num_dofs * 2, 1)

        # Default initial state
        self.template_initial_state = np.ones(self.state_shape) * np.sqrt(2) / 2

        # Current running state (will be set in reset_controller)
        self.y = None

    def initalise_network(self, num_dofs, inter_con_density: float = 0.5):
        """
        Initializes the network topology.
        Fixed seed ensures structure is random but consistent across runs.
        """
        rng = np.random.default_rng(42)

        # 1. Intrinsic Oscillator Weights (2*pi frequency)
        oscillator_weights = np.ones(num_dofs) * 2 * np.pi

        n_states = num_dofs * 2
        weight_matrix = np.zeros((n_states, n_states))

        # Place intrinsic weights on super-diagonal
        rows_osc = np.arange(0, n_states, 2)
        cols_osc = np.arange(1, n_states, 2)
        weight_matrix[rows_osc, cols_osc] = oscillator_weights

        # 2. Random Inter-connections
        n_possible = (num_dofs * (num_dofs - 1)) // 2
        n_active = int(np.round(n_possible * inter_con_density))

        if n_active > 0:
            triu_rows, triu_cols = np.triu_indices(num_dofs, k=1)
            perm_indices = rng.permutation(len(triu_rows))[:n_active]

            selected_rows = triu_rows[perm_indices]
            selected_cols = triu_cols[perm_indices]

            inter_weights = rng.random(n_active)

            # Map DOF indices to State indices (Position-to-Position coupling)
            weight_matrix[selected_rows * 2, selected_cols * 2] = inter_weights

        # 3. Extract Genotype & Apply Anti-Symmetry
        weight_map = np.argwhere(weight_matrix > 0)
        genotype = weight_matrix[weight_map[:, 0], weight_map[:, 1]]
        weight_matrix -= weight_matrix.T

        return weight_matrix, weight_map, genotype

    def geno2pheno(self, genotype: np.ndarray) -> None:
        """
        Map the flat genotype vector to the weight matrix A and initial states.
        """
        self.genotype = genotype[:self.n_weights]

        # Update initial state template
        self.template_initial_state = genotype[self.n_weights:].reshape(self.state_shape)

        # Update weight matrix (Upper Triangle)
        self.A[self.weight_map[:, 0], self.weight_map[:, 1]] = self.genotype
        # Update weight matrix (Lower Triangle) - Enforce Anti-Symmetry
        self.A[self.weight_map[:, 1], self.weight_map[:, 0]] = -self.genotype

    def reset_controller(self, batch_size=1) -> None:
        """
        Resets internal states for a batch of environments.
        """
        # Tile the initial state to match the batch size: (2*N, Batch)
        self.y = np.tile(self.template_initial_state, (1, batch_size))

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Computes the next action based on internal oscillator state.

        :param state: Observations (Batch, Input_Dim). Ignored by this CPG logic
                      (unless feedback is added), but required for interface compatibility.
        :return: Actions (Batch, Output_Dim)
        """
        next_y = RK45(self.y, self.A, self.dt)
        self.y = next_y
        actions_transposed = np.tanh(next_y[1::2,:])
        return actions_transposed.T