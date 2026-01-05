import cv2
import face_recognition
import pickle
import os
import json
from datetime import datetime
from pathlib import Path


class AttendanceTracker:
    def __init__(self, video_source, encodings_file="face_encodings.pkl", attendance_file="attendance.json"):
        self.cap = cv2.VideoCapture(video_source)
        self.encodings_file = encodings_file
        self.attendance_file = attendance_file

        # Load known face encodings
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_encodings()

        # Track attendance
        self.attendance_today = {}
        self.marked_staff = set()
        self.load_attendance()

        # Face detection parameters
        self.face_detection_interval = 5  # Process every 5th frame for performance
        self.frame_count = 0

    def load_encodings(self):
        """Load face encodings from pickle file"""
        if os.path.exists(self.encodings_file):
            with open(self.encodings_file, "rb") as f:
                data = pickle.load(f)
                self.known_face_encodings = data["encodings"]
                self.known_face_names = data["names"]
            print(f"✅ Loaded {len(self.known_face_names)} staff members")
        else:
            print("⚠️ No face encodings found. Please run enroll_staff.py first")

    def load_attendance(self):
        """Load today's attendance records"""
        if os.path.exists(self.attendance_file):
            with open(self.attendance_file, "r") as f:
                all_attendance = json.load(f)
                today = datetime.now().strftime("%Y-%m-%d")
                self.attendance_today = all_attendance.get(today, {})
                self.marked_staff = set(self.attendance_today.keys())
        else:
            self.attendance_today = {}
            self.marked_staff = set()

    def save_attendance(self, name):
        """Save attendance record"""
        today = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")

        # Load all attendance records
        if os.path.exists(self.attendance_file):
            with open(self.attendance_file, "r") as f:
                all_attendance = json.load(f)
        else:
            all_attendance = {}

        # Add today's entry
        if today not in all_attendance:
            all_attendance[today] = {}

        all_attendance[today][name] = {
            "time": current_time,
            "status": "Present"
        }

        # Save back to file
        with open(self.attendance_file, "w") as f:
            json.dump(all_attendance, f, indent=2)

        self.attendance_today[name] = {"time": current_time, "status": "Present"}
        self.marked_staff.add(name)
        print(f"✅ Attendance marked for {name} at {current_time}")

    def get_frame(self):
        """Process frame and detect faces"""
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.resize(frame, (1280, 720))
        self.frame_count += 1

        # Process face recognition every N frames for performance
        if self.frame_count % self.face_detection_interval == 0:
            # Convert BGR to RGB for face_recognition library
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find faces in the frame
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Process each detected face
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compare with known faces
                matches = face_recognition.compare_faces(
                    self.known_face_encodings,
                    face_encoding,
                    tolerance=0.6
                )
                name = "Unknown"
                confidence = 0

                # Find best match
                if True in matches:
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings,
                        face_encoding
                    )
                    best_match_index = face_distances.argmin()

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = (1 - face_distances[best_match_index]) * 100

                        # Mark attendance if not already marked
                        if name not in self.marked_staff:
                            self.save_attendance(name)

                # Draw rectangle around face
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                # Draw label with name and confidence
                label = f"{name}"
                if name != "Unknown":
                    label += f" ({confidence:.1f}%)"
                    status = "✓" if name in self.marked_staff else "○"
                    label = f"{status} {label}"

                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(
                    frame,
                    label,
                    (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

        # Display attendance summary
        summary_text = f"Staff Present: {len(self.marked_staff)}/{len(self.known_face_names)}"
        cv2.putText(
            frame,
            summary_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        # Display current date/time
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame,
            current_datetime,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        return frame

    def get_attendance_summary(self):
        """Get today's attendance summary"""
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_staff": len(self.known_face_names),
            "present": len(self.marked_staff),
            "absent": len(self.known_face_names) - len(self.marked_staff),
            "attendance_records": self.attendance_today
        }

    def get_all_attendance(self):
        """Get all attendance records"""
        if os.path.exists(self.attendance_file):
            with open(self.attendance_file, "r") as f:
                return json.load(f)
        return {}
