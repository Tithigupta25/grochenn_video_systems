import os
import queue
import threading

from app.camera import camera_producer
from app.detector import detection_worker


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