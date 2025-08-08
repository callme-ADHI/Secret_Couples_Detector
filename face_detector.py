import cv2
import mediapipe as mp

# Use raw string to avoid Unicode escape errors in Windows paths
video_path = r"C:\Users\DELL\Downloads\Classroom_Secret_Glances_Video.mp4"

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(
    model_selection=1,                  # Better for mid/far range and tilted faces
    min_detection_confidence=0.3        # Lowered for better sensitivity
)

# Open video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[ERROR] Unable to open video at path: {video_path}")
    exit()

print("[INFO] Video opened successfully.")

frame_count = 0
unique_face_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] End of video or error reading frame.")
        break

    frame_count += 1

    # Convert BGR (OpenCV) to RGB (MediaPipe requirement)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection_results = face_detection.process(rgb_frame)

    # Visual feedback for debugging
    cv2.imshow("Current Frame", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        print("[INFO] Manually interrupted.")
        break

    # Process detections
    if detection_results.detections:
        print(f"[Frame {frame_count}] Faces detected: {len(detection_results.detections)}")
        for idx, detection in enumerate(detection_results.detections):
            # For simplicity, use bounding box center as a "face id"
            bbox = detection.location_data.relative_bounding_box
            x_center = round(bbox.xmin + bbox.width / 2, 2)
            y_center = round(bbox.ymin + bbox.height / 2, 2)
            unique_face_ids.add((x_center, y_center))

    # Stop after 100 frames for now
    if frame_count >= 100:
        print("[INFO] Reached frame limit.")
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nUnique face counts seen in first {frame_count} frames: {len(unique_face_ids)}")
