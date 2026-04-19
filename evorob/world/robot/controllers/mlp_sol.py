import numpy as np

from evorob.world.robot.controllers.base import Controller


class NeuralNetworkController(Controller):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 16,
    ):
        """
        A minimalistic Neural Network, using numpy.
        - One hidden layer with tanh activation
        - Output layer with tanh activation

        :param int input_size: Size of input vector
        :param int hidden_size: Size of hidden layer
        :param int output_size: Size of output vector
        """
        self.n_input = input_size
        self.n_output = output_size
        self.n_hidden = hidden_size
        self.n_con1 = input_size * hidden_size
        self.n_con2 = hidden_size * output_size
        self.lin = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.output = np.random.uniform(-1, 1, (output_size, hidden_size))
        self.n_params = self.get_num_params()

    def get_action(self, state):
        assert state.shape[-1] == self.n_input, (
            "State does not correspond with expected input size"
        )

        hid_l = np.tanh(state @ self.lin.T)
        output_l = np.tanh(hid_l @ self.output.T)
        return np.clip(output_l, -1.0, 1.0)

    def set_weights(self, weights):
        """
        Set weights of NN.

        :param np.ndarray weights: Vector of weights
        """
        assert len(weights) == self.n_con1 + self.n_con2, (
            f"Got {len(weights)} but expected {self.n_con1 + self.n_con2}"
        )
        weight_matrix1 = weights[: self.n_con1].reshape(self.lin.shape)
        weight_matrix2 = weights[-self.n_con2 :].reshape(self.output.shape)
        self.lin = weight_matrix1
        self.output = weight_matrix2

    def geno2pheno(self, genotype):
        """Alias for set_weights (genotype to phenotype mapping)."""
        self.set_weights(genotype)

    def get_num_params(self):
        """Return the total number of parameters in the network."""
        return self.n_con1 + self.n_con2

    def reset_controller(self, batch_size=1) -> None:
        pass
