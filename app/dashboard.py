import numpy as np
import matplotlib.pyplot as plt
import os


def create_dashboard(camera_data, output_path):

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cameras = sorted(camera_data.keys())
    frames_pushed = [camera_data[c]["frames_pushed"] for c in cameras]
    frames_dropped = [camera_data[c]["frames_dropped"] for c in cameras]
    drop_rates = [camera_data[c]["drop_rate"] * 100 for c in cameras]
    max_fullness = [max(camera_data[c]["queue_fullness"]) if camera_data[c]["queue_fullness"] else 0 for c in cameras]

    avg_lag = [sum(camera_data[c]["processing_lags"]) / len(camera_data[c]["processing_lags"])
        if camera_data[c]["processing_lags"] else 0 for c in cameras]

    detections = [camera_data[c]["detections"] for c in cameras]
    health_values = []

    for i, camera_id in enumerate(cameras):

        if (camera_data[camera_id]["drop_rate"] >= 0.50 or max_fullness[i] >= 0.95):
            health_values.append(3)

        elif (camera_data[camera_id]["drop_rate"] >= 0.20 or max_fullness[i] >= 0.80):
            health_values.append(2)

        else:
            health_values.append(1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    camera_labels = [str(c) for c in cameras]

    axes[0, 0].bar(camera_labels, health_values)
    axes[0, 0].set_title("Pipeline Health by Camera")
    axes[0, 0].set_xlabel("Camera")
    axes[0, 0].set_ylabel("Health Level")
    axes[0, 0].set_yticks([1, 2, 3])
    axes[0, 0].set_yticklabels(["Healthy", "Warning", "Critical"])

    axes[0, 1].bar(camera_labels, drop_rates)
    axes[0, 1].set_title("Frame Drop Rate")
    axes[0, 1].set_xlabel("Camera")
    axes[0, 1].set_ylabel("Drop Rate (%)")

    axes[0, 2].bar(camera_labels, [x * 100 for x in max_fullness])
    axes[0, 2].set_title("Maximum Queue Fullness")
    axes[0, 2].set_xlabel("Camera")
    axes[0, 2].set_ylabel("Queue Fullness (%)")
    axes[0, 2].set_ylim(0, 105)

    axes[1, 0].bar(camera_labels, avg_lag)
    axes[1, 0].set_title("Average Processing Lag")
    axes[1, 0].set_xlabel("Camera")
    axes[1, 0].set_ylabel("Lag (seconds)")

    x = np.arange(len(cameras))
    width = 0.35

    axes[1, 1].bar(x - width / 2, frames_pushed, width, label="Pushed")
    axes[1, 1].bar(x + width / 2, frames_dropped, width, label="Dropped")
    axes[1, 1].set_title("Frames Pushed vs Dropped")
    axes[1, 1].set_xlabel("Camera")
    axes[1, 1].set_ylabel("Frames")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(camera_labels)
    axes[1, 1].legend()

    axes[1, 2].bar(camera_labels, detections)
    axes[1, 2].set_title("Total Detections")
    axes[1, 2].set_xlabel("Camera")
    axes[1, 2].set_ylabel("Detections")

    fig.suptitle("Pipeline Monitoring Dashboard", fontsize=18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    
    print(f"Dashboard saved to: {output_path}")