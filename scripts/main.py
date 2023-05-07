import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'models/mobilenetv2_ssd_256_fp32.tflite'
# video_path = 'videos/crowd.mp4'

BaseOptions = mp.tasks.BaseOptions
DetectionResult = mp.tasks.components.containers.DetectionResult
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def print_result(result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
    global bboxes, scores
    for detection in result.detections:
        # bounding box
        bbox = detection.bounding_box.origin_x, \
            detection.bounding_box.origin_y, \
            detection.bounding_box.width, \
            detection.bounding_box.height
        score = detection.categories[0].score

        bboxes.append(bbox)
        scores.append(score)


options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    max_results=5,
    score_threshold=0.5,
    category_allowlist=['person'],
    result_callback=print_result)


with ObjectDetector.create_from_options(options) as detector:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    frame_idx = 0
    while cap.isOpened():
        bboxes = []
        scores = []
        success, frame = cap.read()

        if not success:
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Perform object detection on the video frame.
        detector.detect_async(mp_image, frame_idx)

        if bboxes:
            for bbox, score in zip(bboxes, scores):
                cv2.rectangle(
                    img=frame,
                    pt1=(bbox[0], bbox[1]),
                    pt2=(bbox[0] + bbox[2], bbox[1] + bbox[3]),
                    color=(0, 0, 255),
                    thickness=2
                )
                cv2.putText(
                    img=frame,
                    text=f'{score}',
                    org=(bbox[0], bbox[1]),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 255),
                    thickness=2
                )
        cv2.imshow('Preview', frame)
        frame_idx += 1
        if cv2.waitKey(5) & 0xff == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
