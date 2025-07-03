import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def read_tensorboard_metrics(log_dir="./ppo_tensorboard/"):
    runs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    if not runs:
        raise FileNotFoundError("No TensorBoard logs found.")
    
    run_dir = max(runs, key=os.path.getmtime)
    print(f"[INFO] Reading logs from: {run_dir}")

    event_acc = EventAccumulator(run_dir)
    event_acc.Reload()

    scalars = {}
    for tag in event_acc.Tags().get("scalars", []):
        events = event_acc.Scalars(tag)
        scalars[tag] = {
            "steps": [e.step for e in events],
            "values": [e.value for e in events]
        }
    return scalars

def plot_all_metrics(metrics, output_dir="plots"):
    # Create the output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for tag, data in metrics.items():
        plt.figure()
        plt.plot(data["steps"], data["values"], marker='o', linestyle='-')
        plt.title(tag)
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.grid(True)
        plt.tight_layout()
        filename = f"{tag.replace('/', '_')}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"[INFO] Saved plot: {save_path}")

        # Optional: Print last 5 values
        # print(f"\n[METRIC] {tag}")
        # print("-" * 40)
        # for step, value in zip(data["steps"][-5:], data["values"][-5:]):
        #     print(f"Step {step:<8}: {value:.4f}")
        # print()

if __name__ == "__main__":
    metrics = read_tensorboard_metrics()
    plot_all_metrics(metrics)
