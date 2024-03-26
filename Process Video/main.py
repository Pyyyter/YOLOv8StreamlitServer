import supervision as sv
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "C:/Users/pyyyt/iCloudDrive/Codes/yolov8-streamlit-detection-tracking/videos/video_3.mp4"

model = YOLO("yolov8s.pt")

video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)

def process_frame(frame: np.ndarray, _) -> np.ndarray:
    results = model(frame, imgsz=1280)[0]
    
    detections = sv.Detections.from_yolov8(results)

    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

    labels = [
    f"{model.model.names[class_id]} {confidence:0.2f}"
    for _, confidence, class_id, _
    in detections
        ]
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

    return frame

sv.process_video(source_path=VIDEO_PATH, target_path=f"result.mp4", callback=process_frame)
