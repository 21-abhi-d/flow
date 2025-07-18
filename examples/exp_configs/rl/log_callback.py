from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Log rewards per step
        if self.locals.get("rewards"):
            self.episode_rewards.extend(self.locals["rewards"])
            self.logger.record("custom/reward_step_avg", np.mean(self.locals["rewards"]))
        return True

    def _on_rollout_end(self):
        # Log average episode reward
        if self.episode_rewards:
            avg_reward = np.mean(self.episode_rewards)
            self.logger.record("custom/ep_rew_mean", avg_reward)
            self.episode_rewards = []

        # ✅ Log cumulative avg_wait_time over all completed requests this rollout
        env = self.training_env.envs[0]  # assumes single env
        if hasattr(env, "completed_requests") and env.completed_requests:
            wait_times = [
                r["wait_time"]
                for r in env.completed_requests
                if r.get("wait_time") is not None
            ]
            if wait_times:
                avg_wait_time = sum(wait_times) / len(wait_times)
                self.logger.record("custom/avg_wait_time", avg_wait_time)
