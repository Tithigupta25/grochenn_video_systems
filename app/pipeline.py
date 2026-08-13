import os
import queue
import threading
import json
from collections import defaultdict

from app.camera import camera_producer
from app.detector import detection_worker
from app.dashboard import create_dashboard


def build_camera_data(log_file):

    camera_data = defaultdict(lambda: {
        "frames_pushed": 0,
        "frames_dropped": 0,
        "drop_rate": 0,
        "queue_fullness": [],
        "processing_lags": [],
        "detections": 0,
    })

    with open(log_file, "r", encoding="utf-8") as f:

        for line in f:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            camera_id = event.get("camera_id")

            if camera_id is None:
                continue

            if "queue_fullness" in event:
                camera_data[camera_id]["queue_fullness"].append(
                    event["queue_fullness"]
                )

            if "processing_lag_sec" in event:
                camera_data[camera_id]["processing_lags"].append(
                    event["processing_lag_sec"]
                )

            if "frames_pushed" in event:
                camera_data[camera_id]["frames_pushed"] = event["frames_pushed"]

            if "frames_dropped" in event:
                camera_data[camera_id]["frames_dropped"] = event["frames_dropped"]

            if "frame_drop_rate" in event:
                camera_data[camera_id]["drop_rate"] = event["frame_drop_rate"]

            if event.get("level") == "EVENT":
                camera_data[camera_id]["detections"] += len(
                    event.get("detections", [])
                )

    return dict(camera_data)


def run_pipeline(video_sources, config, object_classes):
    output_dir = config["output"]["detections_dir"]
    log_file = config["output"]["log_file"]
    queue_maxsize = config["pipeline"]["queue_maxsize"]
    num_workers = config["pipeline"]["num_workers"]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    if os.path.exists(log_file):
        os.remove(log_file)

    frame_queue = queue.Queue(maxsize=queue_maxsize)
    producers_done = threading.Event()
    camera_metrics = {}

    producer_threads = [threading.Thread(target=camera_producer,
            args=(source, camera_id, frame_queue, config, camera_metrics, log_file),
            name=f"producer-{camera_id}",) for camera_id, source in enumerate(video_sources, start=1)]

    worker_threads = [threading.Thread(target=detection_worker,
                args=(worker_id, frame_queue, producers_done,
                os.path.join("models", config["model"]["name"],),
                object_classes, config["detection"]["confidence_threshold"],
                queue_maxsize, output_dir, log_file,), name=f"worker-{worker_id}")
                for worker_id in range(1, num_workers + 1)]

    for thread in worker_threads:
        thread.start()

    for thread in producer_threads:
        thread.start()

    for thread in producer_threads:
        thread.join()

    producers_done.set()

    for thread in worker_threads:
        thread.join()

    print(
        f"\n[INFO] Pipeline complete."
        f"\nDetections -> '{output_dir}/'"
        f"\nEvents log -> '{log_file}'"
    )
    
    camera_data = build_camera_data(log_file)
    dashboard_path = os.path.join("results", "pipeline_dashboard.png")
    create_dashboard(camera_data, dashboard_path)