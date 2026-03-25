import numpy as np

from evorob.world.robot.controllers.base import Controller


class FrozenPPORNNResidualController(Controller):
    """
    Frozen PPO MLP backbone + trainable recurrent residual head.

    Final action:
        action = clip(ppo_action + residual_scale * residual_action, -1, 1)

    Only the recurrent residual parameters are trainable/evolved.
    The PPO backbone controller is frozen.
    """

    def __init__(
        self,
        ppo_controller,
        input_size: int,
        output_size: int,
        hidden_size: int = 32,
        residual_scale: float = 0.2,
    ):
        self.ppo_controller = ppo_controller
        self.n_input = input_size
        self.n_output = output_size
        self.hidden_size = int(hidden_size)
        self.residual_scale = float(residual_scale)

        # Trainable recurrent residual parameters
        self.Wxh = np.random.uniform(
            -0.1, 0.1, (self.hidden_size, self.n_input)
        ).astype(np.float32)
        self.Whh = np.random.uniform(
            -0.1, 0.1, (self.hidden_size, self.hidden_size)
        ).astype(np.float32)
        self.bh = np.zeros(self.hidden_size, dtype=np.float32)

        self.Why = np.random.uniform(
            -0.1, 0.1, (self.n_output, self.hidden_size)
        ).astype(np.float32)
        self.by = np.zeros(self.n_output, dtype=np.float32)

        self.hidden = None
        self.n_params = self.get_num_params()

    def reset_controller(self, batch_size=1):
        if hasattr(self.ppo_controller, "reset_controller"):
            self.ppo_controller.reset_controller(batch_size=batch_size)

        if batch_size == 1:
            self.hidden = np.zeros(self.hidden_size, dtype=np.float32)
        else:
            self.hidden = np.zeros((batch_size, self.hidden_size), dtype=np.float32)

    def get_action(self, state):
        x = np.asarray(state, dtype=np.float32)

        single_input = (x.ndim == 1)
        if single_input:
            x = x[None, :]  # (1, input_size)

        batch_size = x.shape[0]

        if self.hidden is None:
            self.reset_controller(batch_size=batch_size)

        h_prev = self.hidden[None, :] if self.hidden.ndim == 1 else self.hidden

        # Frozen PPO backbone action
        ppo_action = self.ppo_controller.get_action(x)

        # Recurrent residual update
        h = np.tanh(x @ self.Wxh.T + h_prev @ self.Whh.T + self.bh)
        residual = h @ self.Why.T + self.by

        action = np.clip(
            ppo_action + self.residual_scale * residual,
            -1.0,
            1.0,
        )

        self.hidden = h[0] if single_input else h

        if single_input:
            return action[0]
        return action

    def get_num_params(self):
        return (
            self.hidden_size * self.n_input
            + self.hidden_size * self.hidden_size
            + self.hidden_size
            + self.n_output * self.hidden_size
            + self.n_output
        )

    def get_weights(self):
        return np.concatenate([
            self.Wxh.ravel(),
            self.Whh.ravel(),
            self.bh.ravel(),
            self.Why.ravel(),
            self.by.ravel(),
        ]).astype(np.float32)

    def set_weights(self, encoding):
        encoding = np.asarray(encoding, dtype=np.float32).ravel()
        expected = self.get_num_params()

        if len(encoding) != expected:
            raise ValueError(f"Expected {expected} params, got {len(encoding)}")

        idx = 0

        s = self.hidden_size * self.n_input
        self.Wxh = encoding[idx:idx + s].reshape(self.hidden_size, self.n_input)
        idx += s

        s = self.hidden_size * self.hidden_size
        self.Whh = encoding[idx:idx + s].reshape(self.hidden_size, self.hidden_size)
        idx += s

        s = self.hidden_size
        self.bh = encoding[idx:idx + s]
        idx += s

        s = self.n_output * self.hidden_size
        self.Why = encoding[idx:idx + s].reshape(self.n_output, self.hidden_size)
        idx += s

        s = self.n_output
        self.by = encoding[idx:idx + s]
        idx += s

        self.n_params = self.get_num_params()

    def geno2pheno(self, genotype):
        self.set_weights(genotype)