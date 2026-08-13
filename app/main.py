import os

from app.config import load_config, load_classes
from app.pipeline import run_pipeline
import cv2


def get_video_sources(config):
    input_config = config["input"]

    mode = input_config.get("mode", "auto")
    video_dir = input_config.get("video_dir", "videos")

    if mode == "camera":
        return input_config.get("camera_indices", [0])

    video_sources = []

    if os.path.exists(video_dir):
        video_sources = [os.path.join(video_dir, filename) for filename in os.listdir(video_dir)
                    if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

    if video_sources:
        print(f"[INFO] Found {len(video_sources)} video source(s).")
        return video_sources

    detected_cameras = []

    for index in range(5):
        cap = cv2.VideoCapture(index)

        if cap.isOpened():
            detected_cameras.append(index)

        cap.release()

    if detected_cameras:
        preferred_cameras = input_config.get("camera_indices", [])
    
        if preferred_cameras:
            selected_cameras = [
                index for index in preferred_cameras
                if index in detected_cameras
            ]
    
            if selected_cameras:
                print(
                    f"[INFO] No video sources found. "
                    f"Detected cameras: {detected_cameras}"
                )
                print(f"[INFO] Using configured cameras: {selected_cameras}")
                return selected_cameras
    
        print(
            f"[INFO] No video sources found. "
            f"Detected cameras: {detected_cameras}"
        )
        return detected_cameras
    print("[ERROR] No cameras or video sources found.")
    return []

def main():
    config = load_config()
    object_classes = load_classes()
    video_sources = get_video_sources(config)
    run_pipeline(video_sources, config, object_classes,)


if __name__ == "__main__":
    main()