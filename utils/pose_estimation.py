# utils/pose_estimation.py - IMPROVED Head Pose Estimation with MediaPipe
import cv2
import numpy as np
import mediapipe as mp
import math


class HeadPoseEstimator:
    def __init__(self):
        """Initialize improved head pose estimator using MediaPipe Face Mesh"""
        print("🧠 Initializing MediaPipe Head Pose Estimator...")
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Face mesh configuration - static mode for better accuracy
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,  # Allow multiple faces
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        
        # Key facial landmark indices for pose estimation
        self.FACE_LANDMARKS = {
            'nose_tip': 1,
            'chin': 175,
            'left_eye_left': 33,
            'left_eye_right': 133,
            'right_eye_left': 362,
            'right_eye_right': 263,
            'left_ear': 234,
            'right_ear': 454,
            'forehead': 9,
            'left_cheek': 116,
            'right_cheek': 345
        }
        
        print("   ✅ MediaPipe Face Mesh initialized")
        print("   ✅ Key landmarks configured for pose estimation")
    
    def get_face_region(self, frame, bbox):
        """Extract face region from bounding box with padding"""
        x1, y1, x2, y2 = bbox
        height, width = frame.shape[:2]
        
        # Add padding around the face region
        padding = 20
        x1_padded = max(0, x1 - padding)
        y1_padded = max(0, y1 - padding)
        x2_padded = min(width, x2 + padding)
        y2_padded = min(height, y2 + padding)
        
        face_region = frame[y1_padded:y2_padded, x1_padded:x2_padded]
        
        return face_region, (x1_padded, y1_padded)
    
    def estimate_yaw_from_landmarks(self, landmarks, image_width, image_height):
        """
        Estimate yaw angle from facial landmarks with improved accuracy
        
        Returns:
            float: Yaw angle in degrees (-90 to +90)
                  Negative = looking left
                  Positive = looking right  
                  0 = looking straight
        """
        if not landmarks:
            return 0
        
        # Convert normalized landmarks to pixel coordinates
        def get_landmark_point(idx):
            landmark = landmarks.landmark[idx]
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            return np.array([x, y])
        
        try:
            # Get key points for yaw estimation
            nose_tip = get_landmark_point(self.FACE_LANDMARKS['nose_tip'])
            left_eye_outer = get_landmark_point(self.FACE_LANDMARKS['left_eye_left'])
            right_eye_outer = get_landmark_point(self.FACE_LANDMARKS['right_eye_right'])
            chin = get_landmark_point(self.FACE_LANDMARKS['chin'])
            
            # Calculate face center
            face_center_x = (left_eye_outer[0] + right_eye_outer[0]) / 2
            face_width = abs(right_eye_outer[0] - left_eye_outer[0])
            
            # Method 1: Nose displacement from face center
            nose_displacement = nose_tip[0] - face_center_x
            nose_ratio = nose_displacement / (face_width / 2) if face_width > 0 else 0
            yaw_from_nose = np.clip(nose_ratio * 45, -90, 90)  # Scale to degrees
            
            # Method 2: Eye asymmetry analysis
            left_eye_center = get_landmark_point(self.FACE_LANDMARKS['left_eye_left'])
            right_eye_center = get_landmark_point(self.FACE_LANDMARKS['right_eye_right'])
            
            # Calculate expected vs actual eye positions
            expected_eye_distance = face_width
            actual_left_distance = abs(nose_tip[0] - left_eye_center[0])
            actual_right_distance = abs(nose_tip[0] - right_eye_center[0])
            
            eye_asymmetry = (actual_right_distance - actual_left_distance) / expected_eye_distance if expected_eye_distance > 0 else 0
            yaw_from_asymmetry = np.clip(eye_asymmetry * 60, -90, 90)
            
            # Method 3: 3D pose estimation using nose-chin vector
            if chin[1] > nose_tip[1]:  # Ensure chin is below nose
                face_vertical_center = (nose_tip + chin) / 2
                nose_offset_x = nose_tip[0] - face_vertical_center[0]
                yaw_from_3d = np.clip(nose_offset_x / (face_width / 4) * 30, -90, 90)
            else:
                yaw_from_3d = 0
            
            # Combine methods with weights
            final_yaw = (
                0.5 * yaw_from_nose +
                0.3 * yaw_from_asymmetry + 
                0.2 * yaw_from_3d
            )
            
            return float(np.clip(final_yaw, -90, 90))
            
        except (IndexError, KeyError) as e:
            # If landmark extraction fails, return 0
            return 0
    
    def estimate_yaw(self, frame, bbox):
        """
        Main function to estimate head yaw angle from a person's bounding box
        
        Args:
            frame: Input video frame
            bbox: Person bounding box (x1, y1, x2, y2)
            
        Returns:
            float: Yaw angle in degrees
        """
        try:
            # Extract face region
            face_region, offset = self.get_face_region(frame, bbox)
            
            if face_region.size == 0:
                return 0
            
            # Convert to RGB for MediaPipe
            face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.face_mesh.process(face_rgb)
            
            if results.multi_face_landmarks:
                # Use the first (largest) face detected
                face_landmarks = results.multi_face_landmarks[0]
                
                # Get face region dimensions
                face_height, face_width = face_region.shape[:2]
                
                # Estimate yaw angle
                yaw_angle = self.estimate_yaw_from_landmarks(
                    face_landmarks, face_width, face_height
                )
                
                return yaw_angle
            else:
                # No face landmarks detected
                return 0
                
        except Exception as e:
            # Handle any processing errors gracefully
            print(f"   ⚠️ Head pose estimation error: {e}")
            return 0
    
    def draw_pose_visualization(self, frame, bbox, yaw_angle):
        """
        Draw head pose visualization on frame for debugging
        
        Args:
            frame: Input frame
            bbox: Person bounding box
            yaw_angle: Calculated yaw angle
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Draw head direction arrow
        arrow_length = 60
        arrow_end_x = center_x + int(arrow_length * math.sin(math.radians(yaw_angle)))
        arrow_end_y = center_y - 20  # Slightly above center
        
        # Color based on direction
        if yaw_angle > 10:
            color = (0, 255, 0)  # Green for right
            direction_text = f"RIGHT {yaw_angle:.1f}°"
        elif yaw_angle < -10:
            color = (255, 0, 0)  # Blue for left  
            direction_text = f"LEFT {abs(yaw_angle):.1f}°"
        else:
            color = (0, 255, 255)  # Yellow for center
            direction_text = f"CENTER {yaw_angle:.1f}°"
        
        # Draw direction arrow
        cv2.arrowedLine(frame, (center_x, center_y), (arrow_end_x, arrow_end_y), color, 3, tipLength=0.3)
        
        # Draw angle text
        cv2.putText(frame, direction_text, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame


def test_head_pose_estimation():
    """Test the improved head pose estimation with webcam"""
    print("🧪 Testing Improved Head Pose Estimation")
    print("=" * 50)
    print("Instructions:")
    print("- Look left and right to test yaw detection")
    print("- Press 'q' to quit")
    print("- Press 's' to save current frame")
    
    # Initialize components
    estimator = HeadPoseEstimator()
    
    # Try to open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    # Initialize YOLO for person detection
    try:
        from ultralytics import YOLO
        yolo_model = YOLO('yolov8n.pt')
        print("✅ YOLO model loaded")
    except ImportError:
        print("❌ Could not load YOLO model")
        return
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Detect people
            results = yolo_model(frame, classes=[0], conf=0.5, verbose=False)
            
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Focus on upper body/head region
                    height = y2 - y1
                    head_y2 = y1 + int(height * 0.4)
                    head_bbox = (x1, y1, x2, head_y2)
                    
                    # Estimate yaw angle
                    yaw_angle = estimator.estimate_yaw(frame, head_bbox)
                    
                    # Draw visualization
                    frame = estimator.draw_pose_visualization(frame, head_bbox, yaw_angle)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, head_y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {i+1}", (x1, y1-30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add instructions
            cv2.putText(frame, "Move your head left/right to test", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Head Pose Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"head_pose_test_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Saved: {filename}")
                
    except KeyboardInterrupt:
        print("Test interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Head pose test completed")


if __name__ == "__main__":
    test_head_pose_estimation()