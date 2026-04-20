import numpy as np

from evorob.world.robot.controllers.base import Controller


class NeuralNetworkController(Controller):
    def __init__(
        self,
        input_size: 27,
        output_size: 16,
        hidden_size=[256, 256],
    ):
        """
        Compatible controller for both:
        - EA flat encodings
        - RL/PPO transferred weights

        hidden_size can be:
        - int: one hidden layer, e.g. 16
        - tuple/list: multiple hidden layers, e.g. (64, 64)
        """
        self.n_input = input_size
        self.n_output = output_size

        if isinstance(hidden_size, int):
            self.hidden_sizes = [hidden_size]
        else:
            self.hidden_sizes = list(hidden_size)

        # Full architecture: [input, h1, h2, ..., output]
        self.layer_sizes = [self.n_input] + self.hidden_sizes + [self.n_output]

        self.weights = []
        self.biases = []

        for in_dim, out_dim in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.weights.append(
                np.random.uniform(-1, 1, (out_dim, in_dim)).astype(np.float32)
            )
            self.biases.append(
                np.random.uniform(-1, 1, (out_dim,)).astype(np.float32)
            )

        self.n_params = self.get_num_params()

    def get_action(self, state):
        """
        Forward pass.
        Accepts shape:
        - (input_size,)
        - (batch_size, input_size)

        Hidden layers use tanh.
        Final layer stays linear to better match SB3 PPO's action_net.
        """
        x = np.asarray(state, dtype=np.float32)

        single_input = (x.ndim == 1)
        if single_input:
            x = x[None, :]  # -> (1, input_size)

        n_layers = len(self.weights)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ W.T + b
            if i < n_layers - 1:
                x = np.tanh(x)

        # keep action range compatible with Ant action space
        x = np.clip(x, -1.0, 1.0)

        if single_input:
            return x[0]
        return x

    def set_weights(self, encoding):
        """
        Supports:
        1) flat EA vector (np.ndarray)
        2) dict with RL weights (already converted to numpy)
        """
        if isinstance(encoding, dict):
            self.weights = [np.array(w, dtype=np.float32) for w in encoding["weights"]]
            self.biases = [np.array(b, dtype=np.float32) for b in encoding["biases"]]
            self.n_params = self.get_num_params()
            return

        encoding = np.asarray(encoding, dtype=np.float32).ravel()
        idx = 0
        new_weights = []
        new_biases = []

        for in_dim, out_dim in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            w_size = out_dim * in_dim
            b_size = out_dim

            W = encoding[idx:idx + w_size].reshape(out_dim, in_dim)
            idx += w_size

            b = encoding[idx:idx + b_size]
            idx += b_size

            new_weights.append(W.astype(np.float32))
            new_biases.append(b.astype(np.float32))

        if idx != len(encoding):
            raise ValueError(
                f"Encoding size mismatch: used {idx} params but got {len(encoding)}"
            )

        self.weights = new_weights
        self.biases = new_biases
        self.n_params = self.get_num_params()

    def geno2pheno(self, genotype):
        self.set_weights(genotype)

    def get_num_params(self):
        total = 0
        for in_dim, out_dim in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            total += out_dim * in_dim   # weights
            total += out_dim            # biases
        return total

    def reset_controller(self, batch_size=1) -> None:
        pass

    def load_from_ppo_model(self, model):
        """
        Copy PPO policy network weights into this controller.
        Assumes continuous-control PPO from stable-baselines3.
        """
        policy_net = model.policy.mlp_extractor.policy_net
        action_net = model.policy.action_net

        weights = []
        biases = []

        for layer in policy_net:
            if hasattr(layer, "weight") and hasattr(layer, "bias"):
                weights.append(layer.weight.detach().cpu().numpy().astype(np.float32))
                biases.append(layer.bias.detach().cpu().numpy().astype(np.float32))

        weights.append(action_net.weight.detach().cpu().numpy().astype(np.float32))
        biases.append(action_net.bias.detach().cpu().numpy().astype(np.float32))

        self.set_weights({
            "weights": weights,
            "biases": biases,
        })