import os
import time
import queue

import cv2
from ultralytics import YOLO

from core.logger import log_event
from core.camera import Frame


def detection_worker(worker_id, frame_queue, producers_done_event, model_path, object_classes, confidence_threshold, queue_maxsize, output_dir, log_file,):
    
    model = YOLO(model_path)
    # model.set_classes(object_classes)
    os.makedirs(output_dir, exist_ok=True)
    processed = 0
    total_detections = 0
    start = time.time()

    while True:
        try:
            item: Frame = frame_queue.get(timeout=2)

        except queue.Empty:
            if producers_done_event.is_set() and frame_queue.empty():
                break
            continue

        try:
            queue_size = frame_queue.qsize()
            queue_capacity = queue_maxsize
            queue_fullness = (queue_size / queue_capacity if queue_capacity > 0 else 0)

            log_event(log_file,{
                    "level": "METRIC",
                    "worker": worker_id,
                    "camera_id": item.camera_id,
                    "frame": item.frame_number,
                    "queue_size": queue_size,
                    "queue_capacity": queue_capacity,
                    "queue_fullness": round(queue_fullness, 3)
                })

            processing_lag = time.time() - item.timestamp

            log_event(log_file,{
                    "level": "METRIC",
                    "camera_id": item.camera_id,
                    "frame": item.frame_number,
                    "worker": worker_id,
                    "processing_lag_sec": round(processing_lag, 3)
                })

            results = model(item.image, verbose=False)[0]
            detections = []

            for box in results.boxes:
                conf = float(box.conf[0])

                if conf < confidence_threshold:
                    continue

                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                        "label": label,
                        "confidence": round(conf, 2),
                        "bbox": [x1, y1, x2, y2]
                    })

                cv2.rectangle(item.image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(item.image, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            total_detections += len(detections)

            if detections:
                out_path = os.path.join(output_dir, f"cam{item.camera_id}_" f"frame{item.frame_number:05d}.jpg")

                cv2.imwrite(out_path, item.image)

                log_event(log_file,{
                        "level": "EVENT",
                        "camera_id": item.camera_id,
                        "frame": item.frame_number,
                        "detections": detections,
                        "worker": worker_id,
                    })

                print(f"[Worker {worker_id}] "
                    f"Cam{item.camera_id} "
                    f"Frame{item.frame_number}: "
                    f"{[(d['label'], d['confidence']) for d in detections]}")

            else:
                print(f"[Worker {worker_id}] "
                    f"Cam{item.camera_id} "
                    f"Frame{item.frame_number}: "
                    f"No detection")

            processed += 1

        except Exception as e:
            log_event(log_file,{
                    "level": "ERROR",
                    "worker": worker_id,
                    "camera_id": item.camera_id,
                    "frame": item.frame_number,
                    "msg": str(e),
                })

            print(f"[Worker {worker_id}] ERROR "
                f"Cam{item.camera_id} "
                f"Frame{item.frame_number}: {e}")

        finally:
            frame_queue.task_done()

    elapsed = time.time() - start
    fps = processed / elapsed if elapsed > 0 else 0

    print()
    print(f"[Worker {worker_id}] Done.")
    print(f"Processed: {processed}")
    print(f"Detections: {total_detections}")
    print(f"Speed: {fps:.2f} FPS")