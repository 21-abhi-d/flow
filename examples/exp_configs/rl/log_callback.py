from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if self.locals.get("rewards"):
            self.episode_rewards.extend(self.locals["rewards"])
            # Optional: log reward step avg if needed
            self.logger.record("custom/reward_step_avg", np.mean(self.locals["rewards"]))
        return True

    def _on_rollout_end(self):
        if self.episode_rewards:
            avg_reward = np.mean(self.episode_rewards)
            self.logger.record("custom/ep_rew_mean", avg_reward)
            self.episode_rewards = []