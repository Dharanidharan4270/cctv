import cv2
from ultralytics import YOLO
import json


def side_of_line(p, a, b):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


class PeopleCounter:
    def __init__(self, video_source):
        self.model = YOLO("yolov8m.pt")
        self.cap = cv2.VideoCapture(video_source)

        with open("line.json") as f:
            data = json.load(f)

        self.line = (
            (data["x1"], data["y1"]),
            (data["x2"], data["y2"])
        )

        self.people_count = 0
        self.counted_ids = set()
        self.track_history = {}

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.resize(frame, (1280, 720))

        results = self.model.track(
            frame,
            persist=True,
            conf=0.5,
            iou=0.7,
            classes=[0]  
        )

        cv2.line(frame, self.line[0], self.line[1], (0, 0, 255), 2)

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, center, 4, (0, 0, 255), -1)

                prev = self.track_history.get(track_id)

                if prev and track_id not in self.counted_ids:
                    if side_of_line(prev, *self.line) < 0 and side_of_line(center, *self.line) >= 0:
                        self.people_count += 1
                        self.counted_ids.add(track_id)

                self.track_history[track_id] = center

        cv2.putText(
            frame,
            f"People Count: {self.people_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        return frame

    def get_count(self):
        return self.people_count