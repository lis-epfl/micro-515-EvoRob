import numpy as np

from evorob.world.robot.controllers.base import Controller


class NeuralNetworkController(Controller):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 16,
    ):
        """Initialize a simple feedforward neural network.

        Network structure: input -> hidden -> output
        Activation: tanh on both layers

        Args:
            input_size: Dimension of input (observation size)
            output_size: Dimension of output (action size)
            hidden_size: Number of hidden neurons
        """
        # Here we randomly initialize our neural network layers,
        # as well as our input and output size.
        self.n_input = input_size
        self.n_output = output_size
        self.n_hidden = hidden_size

        # TODO: Initialize weight matrices with uniform random values in [-1, 1]
        # - self.input_to_hidden: shape (hidden_size, input_size)
        # - self.hidden_to_output: shape (output_size, hidden_size)
        # Hint: Use np.random.uniform(-1, 1, (rows, cols))
        self.input_to_hidden = np.random.uniform(-1, 1, (self.n_hidden, self.n_input)) 

        self.hidden_to_output = np.random.uniform(-1, 1, (self.n_output, self.n_hidden))

        # TODO: Compute number of parameters in each layer
        self.n_params_i2h = self.n_hidden * self.n_input  # input_to_hidden parameters
        self.n_params_h2o = self.n_output * self.n_hidden  # hidden_to_output parameters

        self.n_params = self.get_num_params()


    def get_action(self, state):
        """Forward pass through the network.

        Args:
            state: Observation array, shape (input_size,) or (batch_size, input_size)

        Returns:
            action: Output array, shape (output_size,) or (batch_size, output_size)
        """
        # TODO: Perform forward pass computation
        # 1. Hidden layer: hidden = ...
        # 2. Output layer: output = ...
        # 3. Clip output to [-1, 1] using np.clip()
        #
        # Hint: Use @ operator or np.matmul for matrix multiplication
        # Hint: .T transposes a matrix
        # Hint: np.tanh() applies tanh element-wise
        hidden= np.tanh(self.input_to_hidden @ state.T)  # shape (hidden_size, batch_size)
        output = np.tanh(self.hidden_to_output @ hidden)  # shape (output_size, batch_size)
        return output.T  # shape (batch_size, output_size)

    def set_weights(self, encoding):
        """Set network weights from a flat parameter vector.

        Args:
            encoding: Flat array of size (n_params,) containing all weights
        """
        # TODO: Map the flat encoding to weight matrices
        # 1. Split encoding into two parts:
        #    - First n_params_i2h values for input_to_hidden
        #    - Remaining n_params_h2o values for hidden_to_output
        # 2. Reshape each part to match the weight matrix shapes
        #
        # Hint: Use array slicing: encoding[:n] and encoding[n:]
        # Hint: Use np.reshape(array, (rows, cols)) or array.reshape((rows, cols))
        self.input_to_hidden = encoding[:self.n_params_i2h].reshape((self.n_hidden, self.n_input))
        self.hidden_to_output = encoding[self.n_params_i2h:].reshape((self.n_output, self.n_hidden))

    def geno2pheno(self, genotype):
        """Alias for set_weights (genotype to phenotype mapping)."""
        self.set_weights(genotype)

    def get_num_params(self):
        # To provide a genetic encoding for our neural network controller,
        # we compute and store the number of parameters in our NN class.
        # TODO: Return the total number of parameters in both layers!
        return self.n_params_i2h + self.n_params_h2o

    def reset_controller(self, batch_size=1) -> None:
        pass
