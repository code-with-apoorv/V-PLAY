from ultralytics import YOLO
import cv2
import pickle

class RacketTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        racket_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                racket_detections = pickle.load(f)
            return racket_detections

        for frame in frames:
            racket_dict = self.detect_frame(frame)
            racket_detections.append(racket_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(racket_detections, f)

        return racket_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]
        id_name_dict = results.names

        racket_dict = {}
        next_id = 1  # Give unique IDs per frame, or integrate with tracking if needed

        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            object_cls_id = int(box.cls.tolist()[0])
            object_cls_name = id_name_dict.get(object_cls_id, "unknown")

            if object_cls_name in ["racket", "racquet"]:  # Adjust class name based on your model's training
                racket_dict[next_id] = result
                next_id += 1

        return racket_dict

    def draw_bboxes(self, video_frames, racket_detections):
        output_video_frames = []
        for frame, racket_dict in zip(video_frames, racket_detections):
            for track_id, bbox in racket_dict.items():
                x1, y1, x2, y2 = map(int, bbox)
                cv2.putText(frame, f"Racket Id:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            output_video_frames.append(frame)
        return output_video_frames
