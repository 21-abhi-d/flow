from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

def read_tensorboard_metrics(log_dir="./ppo_tensorboard/"):
    runs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    if not runs:
        print("No TensorBoard logs found.")
        return

    run_dir = max(runs, key=os.path.getmtime)
    print(f"[INFO] Reading logs from: {run_dir}")

    event_acc = EventAccumulator(run_dir)
    event_acc.Reload()

    tags = event_acc.Tags()
    print("[INFO] Available tags:", tags)

    rewards = event_acc.Scalars('rollout/ep_rew_mean')
    losses = event_acc.Scalars('train/value_loss')

    reward_data = [(e.step, e.value) for e in rewards]
    loss_data = [(e.step, e.value) for e in losses]

    return {
        "reward": reward_data,
        "loss": loss_data,
    }

if __name__ == "__main__":
    metrics = read_tensorboard_metrics()
    if metrics:
        print("\nLast 5 rewards:")
        for step, val in metrics["reward"][-5:]:
            print(f"Step {step}: Reward {val}")
