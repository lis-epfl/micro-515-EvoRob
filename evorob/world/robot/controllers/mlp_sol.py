import numpy as np
from evorob.world.robot.controllers.base import Controller


class NeuralNetworkController(Controller):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 8,
    ):
        self.n_input = input_size
        self.n_output = output_size
        self.n_hidden = hidden_size

        # Number of parameters
        self.n_con1 = input_size * hidden_size
        self.n_con2 = hidden_size * output_size
        self.n_bias1 = hidden_size
        self.n_bias2 = output_size

        # Weights
        self.lin = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.output = np.random.uniform(-1, 1, (output_size, hidden_size))

        # Biases
        self.b1 = np.random.uniform(-1, 1, hidden_size)
        self.b2 = np.random.uniform(-1, 1, output_size)

        self.n_params = self.get_num_params()

    def get_action(self, state):
        assert state.shape[-1] == self.n_input, (
            "State does not correspond with expected input size"
        )

        hid_l = np.tanh(state @ self.lin.T + self.b1)
        output_l = np.tanh(hid_l @ self.output.T + self.b2)

        return np.clip(output_l, -1.0, 1.0)

    def set_weights(self, weights):
        """
        Set weights of NN (including biases).
        """
        expected = self.n_con1 + self.n_con2 + self.n_bias1 + self.n_bias2
        assert len(weights) == expected, (
            f"Got {len(weights)} but expected {expected}"
        )

        idx = 0

        # Layer 1 weights
        self.lin = weights[idx : idx + self.n_con1].reshape(self.lin.shape)
        idx += self.n_con1

        # Layer 2 weights
        self.output = weights[idx : idx + self.n_con2].reshape(self.output.shape)
        idx += self.n_con2

        # Biases
        self.b1 = weights[idx : idx + self.n_bias1]
        idx += self.n_bias1

        self.b2 = weights[idx : idx + self.n_bias2]

    def geno2pheno(self, genotype):
        self.set_weights(genotype)

    def get_num_params(self):
        return (
            self.n_con1
            + self.n_con2
            + self.n_bias1
            + self.n_bias2
        )

    def reset_controller(self, batch_size=1) -> None:
        pass