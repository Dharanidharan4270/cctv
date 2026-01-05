import cv2
import json

VIDEO_SOURCE = "video.mp4"  # webcam, or RTSP / video file
LINE_FILE = "line.json"

drawing = False
start_point = None
end_point = None


def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, end_point

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)


cap = cv2.VideoCapture(VIDEO_SOURCE)
cv2.namedWindow("Draw Line")
cv2.setMouseCallback("Draw Line", mouse_callback)

print("🖱 Click & drag to draw the line")
print("💾 Press S to save")
print("❌ Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if start_point and end_point:
        cv2.line(frame, start_point, end_point, (0, 255, 255), 2)

    cv2.imshow("Draw Line", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s") and start_point and end_point:
        with open(LINE_FILE, "w") as f:
            json.dump({
                "x1": start_point[0],
                "y1": start_point[1],
                "x2": end_point[0],
                "y2": end_point[1]
            }, f)
        print("✅ Line saved to line.json")
        break

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()