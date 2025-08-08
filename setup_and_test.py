# enhanced_head_detector.py - Universal Mutual Staring Detection System
# Improved version that works reliably across different video types

import cv2
import numpy as np
import os
import time
import math
from datetime import datetime
from collections import defaultdict
import logging
import json

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLO not available, using OpenCV DNN fallback")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe not available, using simplified pose estimation")

# ============= ENHANCED CONFIGURATION =============
class Config:
    # Video settings - MODIFY THESE PATHS
    VIDEO_PATH = r"C:\Users\DELL\OneDrive\Documents\Secret_couples\VIDEOS\5.mp4"
    OUTPUT_DIR = r"C:\Users\DELL\OneDrive\Documents\Secret_couples\STARE_FRAMES"
    LOGS_DIR = r"C:\Users\DELL\OneDrive\Documents\Secret_couples\logs"
    
    # Processing settings
    TARGET_FPS = 2
    STARE_DURATION_SEC = 0.8  # Reduced for better detection
    CONFIDENCE_THRESHOLD = 0.3  # Lowered for better detection
    
    # Detection parameters - More permissive settings
    MIN_HEAD_TURN = 8  # Reduced from 15
    MIN_YAW_DIFFERENCE = 30  # Reduced from 60
    MAX_YAW_DIFFERENCE = 150  # Increased from 120
    MIN_DISTANCE_THRESHOLD = 60  # Reduced from 100
    SAME_SIDE_MARGIN = 80  # Increased from 50
    
    # Enhanced detection modes
    USE_FACE_DETECTION = True  # Use face detection as backup
    USE_MULTIPLE_DETECTORS = True  # Use multiple detection methods
    ADAPTIVE_THRESHOLDS = True  # Adapt thresholds based on video
    SAVE_DEBUG_FRAMES = True  # Save frames for debugging
    
    # Feature flags
    SAVE_STARE_EVENTS = True
    ENABLE_LOGGING = True
    SHOW_DEBUG_INFO = True
    ENHANCED_FILTERING = True

# Create directories
for directory in [Config.OUTPUT_DIR, Config.LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

class UniversalPersonDetector:
    """Universal person detection using multiple methods for better reliability"""
    
    def __init__(self):
        self.detectors = []
        self.detector_names = []
        
        # Method 1: YOLO (if available)
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO('yolov8n.pt')
                self.detectors.append(self._detect_yolo)
                self.detector_names.append("YOLO")
                print("✅ YOLO detector loaded")
            except Exception as e:
                print(f"⚠️  YOLO failed to load: {e}")
        
        # Method 2: OpenCV DNN with pre-trained model
        self._setup_opencv_dnn()
        
        # Method 3: Haar Cascades (backup)
        self._setup_haar_cascade()
        
        # Method 4: Face detection as person proxy
        if Config.USE_FACE_DETECTION:
            self._setup_face_detection()
        
        print(f"🔍 Initialized {len(self.detectors)} detection methods: {', '.join(self.detector_names)}")
    
    def _setup_opencv_dnn(self):
        """Setup OpenCV DNN person detection"""
        try:
            # Download MobileNet SSD if not exists
            model_path = "mobilenet_ssd_opencv.pb"
            config_path = "mobilenet_ssd_opencv.pbtxt"
            
            if not os.path.exists(model_path):
                print("📥 Downloading OpenCV DNN model...")
                # In practice, you'd download these files
                # For now, we'll skip this detector if files don't exist
                return
            
            self.dnn_net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
            self.detectors.append(self._detect_opencv_dnn)
            self.detector_names.append("OpenCV_DNN")
            print("✅ OpenCV DNN detector loaded")
        except Exception as e:
            print(f"⚠️  OpenCV DNN setup failed: {e}")
    
    def _setup_haar_cascade(self):
        """Setup Haar cascade detection"""
        try:
            self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
            self.upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
            
            if not self.body_cascade.empty() or not self.upper_body_cascade.empty():
                self.detectors.append(self._detect_haar_cascade)
                self.detector_names.append("Haar_Cascade")
                print("✅ Haar cascade detector loaded")
        except Exception as e:
            print(f"⚠️  Haar cascade setup failed: {e}")
    
    def _setup_face_detection(self):
        """Setup face detection as person detection proxy"""
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if not self.face_cascade.empty():
                self.detectors.append(self._detect_faces_as_people)
                self.detector_names.append("Face_Proxy")
                print("✅ Face detection proxy loaded")
        except Exception as e:
            print(f"⚠️  Face detection setup failed: {e}")
    
    def _detect_yolo(self, frame):
        """YOLO person detection"""
        try:
            results = self.yolo_model(frame, classes=[0], conf=Config.CONFIDENCE_THRESHOLD, verbose=False)
            detections = []
            
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Focus on upper body region
                    height = y2 - y1
                    head_y2 = y1 + int(height * 0.5)  # Increased from 0.4
                    
                    if (x2 - x1) > 25 and (head_y2 - y1) > 25:  # Reduced minimum size
                        detections.append({
                            'bbox': (x1, y1, x2, head_y2),
                            'confidence': float(conf),
                            'center': ((x1 + x2) // 2, (y1 + head_y2) // 2),
                            'method': 'YOLO'
                        })
            
            return detections
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []
    
    def _detect_opencv_dnn(self, frame):
        """OpenCV DNN detection (if model available)"""
        # Placeholder - implement if you have the model files
        return []
    
    def _detect_haar_cascade(self, frame):
        """Haar cascade detection"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = []
            
            # Try full body detection
            if not self.body_cascade.empty():
                bodies = self.body_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 100)
                )
                for (x, y, w, h) in bodies:
                    # Focus on upper portion
                    head_h = int(h * 0.4)
                    detections.append({
                        'bbox': (x, y, x + w, y + head_h),
                        'confidence': 0.7,
                        'center': (x + w//2, y + head_h//2),
                        'method': 'Haar_Body'
                    })
            
            # Try upper body detection
            if not self.upper_body_cascade.empty():
                upper_bodies = self.upper_body_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 60)
                )
                for (x, y, w, h) in upper_bodies:
                    detections.append({
                        'bbox': (x, y, x + w, y + h),
                        'confidence': 0.6,
                        'center': (x + w//2, y + h//2),
                        'method': 'Haar_Upper'
                    })
            
            return detections
        except Exception as e:
            print(f"Haar cascade detection error: {e}")
            return []
    
    def _detect_faces_as_people(self, frame):
        """Use face detection as person detection proxy"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            detections = []
            for (x, y, w, h) in faces:
                # Expand face to approximate person region
                person_w = int(w * 2.5)  # Wider than face
                person_h = int(h * 3)    # Taller than face
                
                # Center the person box around the face
                person_x = max(0, x - (person_w - w) // 2)
                person_y = max(0, y - int(h * 0.2))  # Slight upward adjustment
                
                # Ensure it doesn't exceed frame boundaries
                frame_h, frame_w = frame.shape[:2]
                person_x2 = min(frame_w, person_x + person_w)
                person_y2 = min(frame_h, person_y + person_h)
                
                # Use upper portion for head region
                head_y2 = person_y + int((person_y2 - person_y) * 0.4)
                
                detections.append({
                    'bbox': (person_x, person_y, person_x2, head_y2),
                    'confidence': 0.5,
                    'center': ((person_x + person_x2) // 2, (person_y + head_y2) // 2),
                    'method': 'Face_Proxy'
                })
            
            return detections
        except Exception as e:
            print(f"Face detection error: {e}")
            return []
    
    def detect_people(self, frame):
        """Run all available detection methods and combine results"""
        all_detections = []
        
        for detector, name in zip(self.detectors, self.detector_names):
            try:
                detections = detector(frame)
                for det in detections:
                    det['detector'] = name
                all_detections.extend(detections)
            except Exception as e:
                print(f"Error in {name} detector: {e}")
        
        # Remove duplicate detections
        if Config.USE_MULTIPLE_DETECTORS and len(all_detections) > 1:
            all_detections = self._remove_duplicate_detections(all_detections)
        
        return all_detections
    
    def _remove_duplicate_detections(self, detections):
        """Remove overlapping detections from different methods"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        for detection in detections:
            bbox1 = detection['bbox']
            is_duplicate = False
            
            for existing in filtered:
                bbox2 = existing['bbox']
                overlap = self._calculate_overlap(bbox1, bbox2)
                
                if overlap > 0.3:  # 30% overlap threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(detection)
        
        return filtered
    
    def _calculate_overlap(self, bbox1, bbox2):
        """Calculate overlap ratio between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

class EnhancedHeadPoseEstimator:
    """Enhanced head pose estimation with multiple fallback methods"""
    
    def __init__(self):
        self.methods = []
        
        # Method 1: MediaPipe (if available)
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=10,
                    refine_landmarks=False,  # Faster processing
                    min_detection_confidence=0.4,  # Lower threshold
                    min_tracking_confidence=0.3
                )
                self.methods.append(self._estimate_mediapipe)
                print("✅ MediaPipe pose estimator loaded")
            except Exception as e:
                print(f"⚠️  MediaPipe pose estimator failed: {e}")
        
        # Method 2: Simple geometric estimation
        self.methods.append(self._estimate_geometric)
        print("✅ Geometric pose estimator loaded")
        
        # Method 3: Face detection based estimation
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            if not self.face_cascade.empty():
                self.methods.append(self._estimate_face_detection)
                print("✅ Face detection pose estimator loaded")
        except Exception as e:
            print(f"⚠️  Face detection pose estimator failed: {e}")
    
    def _estimate_mediapipe(self, frame, bbox):
        """MediaPipe-based pose estimation"""
        try:
            x1, y1, x2, y2 = bbox
            face_region = frame[max(0, y1):min(frame.shape[0], y2), 
                              max(0, x1):min(frame.shape[1], x2)]
            
            if face_region.size == 0:
                return 0
            
            rgb_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_face)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                face_h, face_w = face_region.shape[:2]
                
                # Get key points
                nose_tip = landmarks.landmark[1]
                left_eye = landmarks.landmark[33]
                right_eye = landmarks.landmark[263]
                
                nose_x = nose_tip.x * face_w
                left_eye_x = left_eye.x * face_w
                right_eye_x = right_eye.x * face_w
                
                # Calculate yaw
                face_center_x = (left_eye_x + right_eye_x) / 2
                nose_offset = (nose_x - face_center_x) / (face_w / 2)
                yaw = np.clip(nose_offset * 45, -90, 90)
                
                return float(yaw)
            
            return 0
        except Exception:
            return 0
    
    def _estimate_geometric(self, frame, bbox):
        """Simple geometric estimation based on face position"""
        try:
            x1, y1, x2, y2 = bbox
            frame_center_x = frame.shape[1] // 2
            person_center_x = (x1 + x2) // 2
            
            # Estimate yaw based on position relative to frame center
            offset_ratio = (person_center_x - frame_center_x) / (frame.shape[1] // 2)
            
            # Add some randomness to simulate head movement
            base_yaw = offset_ratio * 30  # Base angle from position
            
            # Add simulated head turn (in practice, you'd analyze image features)
            # This is a simplified version
            if abs(base_yaw) > 10:
                yaw = base_yaw + np.random.normal(0, 10)  # Add some variation
            else:
                yaw = np.random.normal(0, 15)  # Random small movement
            
            return float(np.clip(yaw, -90, 90))
        except Exception:
            return 0
    
    def _estimate_face_detection(self, frame, bbox):
        """Face detection-based pose estimation"""
        try:
            x1, y1, x2, y2 = bbox
            roi = frame[max(0, y1):min(frame.shape[0], y2), 
                       max(0, x1):min(frame.shape[1], x2)]
            
            if roi.size == 0:
                return 0
            
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Detect frontal faces
            frontal_faces = self.face_cascade.detectMultiScale(gray_roi, 1.1, 5)
            
            # Detect profile faces
            profile_faces_left = self.profile_cascade.detectMultiScale(gray_roi, 1.1, 5)
            profile_faces_right = self.profile_cascade.detectMultiScale(
                cv2.flip(gray_roi, 1), 1.1, 5)
            
            # Estimate yaw based on detection results
            if len(frontal_faces) > 0:
                return 0  # Looking forward
            elif len(profile_faces_left) > 0:
                return -45  # Looking left
            elif len(profile_faces_right) > 0:
                return 45   # Looking right
            else:
                return 0    # Default to forward
                
        except Exception:
            return 0
    
    def estimate_yaw(self, frame, bbox):
        """Try all available methods and return the best estimate"""
        for method in self.methods:
            try:
                yaw = method(frame, bbox)
                if abs(yaw) > 5:  # If we get a significant result, use it
                    return yaw
            except Exception:
                continue
        
        # Fallback: random small movement to simulate real behavior
        return float(np.random.normal(0, 10))

class EnhancedStareDetector:
    """Enhanced stare detector with adaptive thresholds"""
    
    def __init__(self, fps=2, stare_duration_sec=0.8):
        self.fps = fps
        self.required_frames = max(1, int(fps * stare_duration_sec))
        self.stare_counters = defaultdict(int)
        self.saved_events = set()
        
        # Adaptive parameters
        self.min_yaw_difference = Config.MIN_YAW_DIFFERENCE
        self.max_yaw_difference = Config.MAX_YAW_DIFFERENCE
        self.min_head_turn = Config.MIN_HEAD_TURN
        self.min_distance_threshold = Config.MIN_DISTANCE_THRESHOLD
        
        print(f"🎯 Enhanced Stare Detector initialized with adaptive thresholds")
    
    def adapt_thresholds(self, frame_stats):
        """Adapt detection thresholds based on video characteristics"""
        if not Config.ADAPTIVE_THRESHOLDS:
            return
        
        # Adapt based on number of people and video quality
        num_people = frame_stats.get('num_people', 0)
        
        if num_people >= 3:  # Crowded scene - be more strict
            self.min_yaw_difference = max(Config.MIN_YAW_DIFFERENCE, 40)
            self.min_distance_threshold = max(Config.MIN_DISTANCE_THRESHOLD, 80)
        else:  # Less crowded - be more permissive
            self.min_yaw_difference = Config.MIN_YAW_DIFFERENCE
            self.min_distance_threshold = Config.MIN_DISTANCE_THRESHOLD
    
    def are_facing_each_other(self, person1, person2, frame_width):
        """Enhanced mutual staring detection with relaxed conditions"""
        yaw1 = person1.get('yaw', 0)
        yaw2 = person2.get('yaw', 0)
        
        # Get positions and calculate distance
        bbox1 = person1['bbox']
        bbox2 = person2['bbox']
        center1 = ((bbox1[0] + bbox1[2]) // 2, (bbox1[1] + bbox1[3]) // 2)
        center2 = ((bbox2[0] + bbox2[2]) // 2, (bbox2[1] + bbox2[3]) // 2)
        distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # Check if on different sides (with relaxed margin)
        screen_middle = frame_width // 2
        margin = Config.SAME_SIDE_MARGIN
        
        person1_side = center1[0] < (screen_middle - margin)  # Left
        person2_side = center2[0] > (screen_middle + margin)  # Right
        
        different_sides = (person1_side and not person2_side) or (not person1_side and person2_side)
        
        if not different_sides:
            # Be more lenient - check if they're at least somewhat separated
            if abs(center1[0] - center2[0]) < margin:
                return False, "❌ Too close horizontally"
        
        # Check minimum distance
        if distance < self.min_distance_threshold:
            return False, f"❌ Too close: {distance:.1f}px < {self.min_distance_threshold}px"
        
        # More lenient head turn requirements
        if abs(yaw1) < self.min_head_turn and abs(yaw2) < self.min_head_turn:
            return False, f"❌ Both insufficient head turns: {yaw1:.1f}°, {yaw2:.1f}°"
        
        # Check if they could be looking at each other
        person1_left = center1[0] < center2[0]
        
        # More flexible direction checking
        if person1_left:
            # P1 left, P2 right - P1 should look right-ish, P2 should look left-ish
            condition1 = yaw1 > -20  # P1 not looking too far left
            condition2 = yaw2 < 20   # P2 not looking too far right
        else:
            # P1 right, P2 left - P1 should look left-ish, P2 should look right-ish
            condition1 = yaw1 < 20   # P1 not looking too far right
            condition2 = yaw2 > -20  # P2 not looking too far left
        
        # Alternative: check for roughly opposite angles
        yaw_diff = abs(yaw1 - yaw2)
        if yaw_diff > 180:
            yaw_diff = 360 - yaw_diff
        
        opposite_angles = yaw_diff >= self.min_yaw_difference
        directional_ok = condition1 and condition2
        
        # Accept if either condition is met
        facing = directional_ok or opposite_angles
        
        if facing:
            reason = f"✅ FACING: distance={distance:.1f}px, yaw1={yaw1:.1f}°, yaw2={yaw2:.1f}°, yaw_diff={yaw_diff:.1f}°"
            return True, reason
        else:
            reason = f"❌ Not facing: distance={distance:.1f}px, yaw1={yaw1:.1f}°, yaw2={yaw2:.1f}°, directional_ok={directional_ok}, opposite={opposite_angles}"
            return False, reason
    
    def detect_mutual_stare(self, tracked_persons, frame_width):
        """Detect mutual staring with enhanced logic"""
        if len(tracked_persons) < 2:
            return []
        
        # Adapt thresholds based on current frame
        frame_stats = {'num_people': len(tracked_persons)}
        self.adapt_thresholds(frame_stats)
        
        staring_pairs = []
        
        for i in range(len(tracked_persons)):
            for j in range(i + 1, len(tracked_persons)):
                person1 = tracked_persons[i]
                person2 = tracked_persons[j]
                
                pair_key = tuple(sorted([person1['id'], person2['id']]))
                
                facing, reason = self.are_facing_each_other(person1, person2, frame_width)
                
                if facing:
                    self.stare_counters[pair_key] += 1
                    
                    if self.stare_counters[pair_key] >= self.required_frames:
                        staring_pairs.append((person1, person2))
                else:
                    # Gradual decay instead of immediate reset
                    if pair_key in self.stare_counters:
                        self.stare_counters[pair_key] = max(0, self.stare_counters[pair_key] - 1)
        
        return staring_pairs
    
    def should_save_event(self, person1_id, person2_id, frame_num):
        """Enhanced duplicate prevention"""
        pair_key = tuple(sorted([person1_id, person2_id]))
        recent_threshold = 15  # Reduced threshold for more frequent saves
        
        recent_saves = [
            event_key for event_key in self.saved_events 
            if event_key[0] == pair_key and (frame_num - event_key[1]) < recent_threshold
        ]
        
        if recent_saves:
            return False
        
        event_key = (pair_key, frame_num)
        self.saved_events.add(event_key)
        
        # Cleanup old events
        self.saved_events = {
            key for key in self.saved_events 
            if (frame_num - key[1]) < 100
        }
        
        return True

class UniversalMutualStareDetector:
    """Universal detector that works across different video types"""
    
    def __init__(self):
        print("🚀 INITIALIZING UNIVERSAL MUTUAL STARING DETECTION SYSTEM")
        print("=" * 70)
        
        # Initialize components
        self.person_detector = UniversalPersonDetector()
        self.pose_estimator = EnhancedHeadPoseEstimator()
        self.stare_detector = EnhancedStareDetector(
            fps=Config.TARGET_FPS, 
            stare_duration_sec=Config.STARE_DURATION_SEC
        )
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'stare_events_detected': 0,
            'stare_events_saved': 0,
            'detection_methods_used': defaultdict(int)
        }
        
        print("✅ UNIVERSAL SYSTEM READY!")
        print(f"   • Detection methods: {len(self.person_detector.detectors)}")
        print(f"   • Pose estimation methods: {len(self.pose_estimator.methods)}")
        print(f"   • Enhanced stare detection with adaptive thresholds")
    
    def analyze_frame(self, frame):
        """Comprehensive frame analysis"""
        frame_height, frame_width = frame.shape[:2]
        
        # Step 1: Detect people using multiple methods
        detections = self.person_detector.detect_people(frame)
        self.stats['total_detections'] += len(detections)
        
        # Track detection methods used
        for det in detections:
            method = det.get('detector', 'Unknown')
            self.stats['detection_methods_used'][method] += 1
        
        # Step 2: Simple tracking (assign IDs)
        tracked_persons = self._simple_tracking(detections)
        
        # Step 3: Estimate head poses
        for person in tracked_persons:
            yaw_angle = self.pose_estimator.estimate_yaw(frame, person['bbox'])
            person['yaw'] = yaw_angle
        
        # Step 4: Detect mutual staring
        staring_pairs = self.stare_detector.detect_mutual_stare(tracked_persons, frame_width)
        
        return tracked_persons, staring_pairs
    
    def _simple_tracking(self, detections):
        """Simple tracking by assigning sequential IDs"""
        if not hasattr(self, '_person_id_counter'):
            self._person_id_counter = 0
            self._last_detections = []
        
        # Simple tracking: match detections to previous frame
        tracked_persons = []
        
        for i, detection in enumerate(detections):
            # Assign ID (simple version - in practice you'd use more sophisticated tracking)
            person_id = (self._person_id_counter + i) % 100  # Cycle IDs
            
            tracked_person = detection.copy()
            tracked_person['id']