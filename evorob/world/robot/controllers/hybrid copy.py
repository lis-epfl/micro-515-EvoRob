import numpy as np
from evorob.world.robot.controllers.base import Controller


class PhaseHybridResidualController(Controller):
    """
    Frozen PPO + trainable residual MLP + phase features.

    Final action:
        action = ppo_action + residual_scale * residual_action

    Trainable params:
        [residual_mlp_params | oscillator_params]
    """

    def __init__(
        self,
        ppo_controller,
        residual_controller,
        phase_controller,
        residual_scale: float = 0.05,
        action_dim: int = 8,
    ):
        self.ppo_controller = ppo_controller
        self.residual_controller = residual_controller
        self.phase_controller = phase_controller
        self.residual_scale = float(residual_scale)
        self.action_dim = int(action_dim)

    def reset_controller(self, batch_size=1):
        if hasattr(self.ppo_controller, "reset_controller"):
            self.ppo_controller.reset_controller(batch_size=batch_size)

        if hasattr(self.residual_controller, "reset_controller"):
            self.residual_controller.reset_controller(batch_size=batch_size)

        if hasattr(self.phase_controller, "reset_controller"):
            self.phase_controller.reset_controller(batch_size=batch_size)

    def _augment_state(self, state):
        state = np.asarray(state, dtype=np.float32)
        phase_feat = self.phase_controller.get_phase_features(state)

        if state.ndim == 1:
            return np.concatenate([state, phase_feat], axis=0)
        elif state.ndim == 2:
            return np.concatenate([state, phase_feat], axis=1)
        else:
            raise ValueError(f"Unsupported state shape: {state.shape}")

    def get_action(self, state):
        state = np.asarray(state, dtype=np.float32)

        ppo_action = self.ppo_controller.get_action(state)
        aug_state = self._augment_state(state)
        residual_action = self.residual_controller.get_action(aug_state)

        action = ppo_action + self.residual_scale * residual_action
        action = np.clip(action, -1.0, 1.0)

        self.phase_controller.step_time()
        return action

    def set_weights(self, weights):
        weights = np.asarray(weights, dtype=np.float32).ravel()

        n_res = self.residual_controller.get_num_params()
        n_phase = self.phase_controller.get_num_params()
        expected = n_res + n_phase

        if len(weights) != expected:
            raise ValueError(f"Expected {expected} params, got {len(weights)}")

        self.residual_controller.set_weights(weights[:n_res])
        self.phase_controller.set_weights(weights[n_res:n_res + n_phase])

    def get_num_params(self):
        return (
            self.residual_controller.get_num_params()
            + self.phase_controller.get_num_params()
        )

    def geno2pheno(self, genotype):
        self.set_weights(genotype)

    def get_phase_info(self):
        return {
            "frequency": float(self.phase_controller.frequency),
            "phase": float(self.phase_controller.phase),
        }