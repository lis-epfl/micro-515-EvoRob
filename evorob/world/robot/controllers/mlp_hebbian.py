import numpy as np

from evorob.world.robot.controllers.base import Controller


class HebbianNumpyNetwork:

    def __init__(self, n_input: int, n_hidden: int, n_output: int):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.size_l1 = n_input * n_hidden
        self.size_l2 = n_hidden * n_output
        self.total_weights = self.size_l1 + self.size_l2

        self.lr = 0.1

        self.A = np.zeros(self.total_weights)
        self.B = np.zeros(self.total_weights)
        self.C = np.zeros(self.total_weights)
        self.D = np.zeros(self.total_weights)

        rng = np.random.default_rng(42)
        self.rescale_weights = 1
        self.lin1_init = rng.uniform(-self.rescale_weights,  self.rescale_weights, (n_hidden, n_input))
        self.output_init = rng.uniform(-self.rescale_weights,  self.rescale_weights, (n_output, n_hidden))

        self.lin1 = None
        self.output = None

    def set_hebbian_rules(self, abcd: np.array) -> None:
        abcd = np.array(abcd).reshape(4, self.total_weights)
        self.A = abcd[0, :]
        self.B = abcd[1, :]
        self.C = abcd[2, :]
        self.D = abcd[3, :]

        self.A1 = self.A[:self.size_l1].reshape(self.n_hidden, self.n_input)
        self.B1 = self.B[:self.size_l1].reshape(self.n_hidden, self.n_input)
        self.C1 = self.C[:self.size_l1].reshape(self.n_hidden, self.n_input)
        self.D1 = self.D[:self.size_l1].reshape(self.n_hidden, self.n_input)

        self.A2 = self.A[self.size_l1:].reshape(self.n_output, self.n_hidden)
        self.B2 = self.B[self.size_l1:].reshape(self.n_output, self.n_hidden)
        self.C2 = self.C[self.size_l1:].reshape(self.n_output, self.n_hidden)
        self.D2 = self.D[self.size_l1:].reshape(self.n_output, self.n_hidden)

    def reset_weights(self, batch_size=1):
        self.lin1 = np.tile(self.lin1_init, (batch_size, 1, 1))
        self.output = np.tile(self.output_init, (batch_size, 1, 1))

    def forward(self, state: np.ndarray):
        if state.ndim == 1:
            state = state.reshape(1, -1)

        pre_act1 = np.einsum('bhi,bi->bh', self.lin1, state)
        hid_l = np.tanh(pre_act1)

        pre_act2 = np.einsum('boh,bh->bo', self.output, hid_l)
        output_l = np.tanh(pre_act2)

        outer_1 = np.einsum('bh,bi->bhi', hid_l, state)
        delta_1 = self.lr * (
            self.A1 * outer_1 +
            self.B1 * state[:, None, :] +
            self.C1 * hid_l[:, :, None] +
            self.D1
        )
        self.lin1 += delta_1

        outer_2 = np.einsum('bo,bh->boh', output_l, hid_l)
        delta_2 = self.lr * (
            self.A2 * outer_2 +
            self.B2 * hid_l[:, None, :] +
            self.C2 * output_l[:, :, None] +
            self.D2
        )
        self.output += delta_2

        return output_l


class HebbianController(Controller):

    def __init__(self, input_size, output_size, hidden_size,):
        self.controller_type = "Hebbian"
        self.n_input = input_size
        self.n_hidden = hidden_size
        self.n_output = output_size
        self.model = HebbianNumpyNetwork(input_size, hidden_size, output_size)
        self.n_params = (self.model.size_l1 + self.model.size_l2) * 4

    def geno2pheno(self, genotype: np.ndarray) -> None:
        self.model.set_hebbian_rules(genotype)

    def reset_controller(self, batch_size=1) -> None:
        self.model.reset_weights(batch_size)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        action = self.model.forward(state)
        return action