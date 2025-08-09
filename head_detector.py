# head_detector.py - Complete Enhanced Mutual Staring Detection System
# Usage: python head_detector.py

import cv2
import numpy as np
import os
import subprocess
import sys
import time
import math
from datetime import datetime
from ultralytics import YOLO
import logging
from collections import defaultdict

# Import custom modules
from staring import StareDetector, save_stare_event, log_stare_event
from utils.pose_estimation import HeadPoseEstimator
from utils.tracker import PersonTracker

# ============= ENHANCED CONFIGURATION =============
VIDEO_PATH = r"C:\Users\DELL\OneDrive\Documents\Secret_couples\VIDEOS\2.mp4"
OUTPUT_DIR = r"C:\Users\DELL\OneDrive\Documents\Secret_couples\STARE_FRAMES"
LOGS_DIR = r"C:\Users\DELL\OneDrive\Documents\Secret_couples\logs"

# Processing settings
TARGET_FPS = 2  # Process at 2 FPS for stability
STARE_DURATION_SEC = 1.0  # Minimum 1 second for staring event
CONFIDENCE_THRESHOLD = 0.5  # YOLO detection confidence

# Enhanced feature flags
SAVE_STARE_EVENTS = True  # Only save stare_event_frame images (no individual faces)
ENABLE_LOGGING = True
SHOW_DEBUG_INFO = True
ENHANCED_FILTERING = True  # Enable same-side filtering and duplicate prevention

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Setup enhanced logging
if ENABLE_LOGGING:
    log_filename = f"mutual_stares_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOGS_DIR, log_filename)),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("🚀 Enhanced Mutual Staring Detection System - Session Started")
else:
    logger = None


class EnhancedMutualStareDetector:
    def __init__(self):
        """Initialize the enhanced mutual stare detection system"""
        print("🚀 INITIALIZING ENHANCED MUTUAL STARING DETECTION SYSTEM")
        print("=" * 65)
        
        # Load YOLOv8 model
        print("📦 Loading YOLOv8 model...")
        self.yolo_model = YOLO('yolov8n.pt')  # Will download if not present
        print("   ✅ YOLOv8n model loaded")
        
        # Initialize enhanced components
        print("🔧 Initializing system components...")
        self.tracker = PersonTracker(max_disappeared=15, max_distance=120)
        self.head_pose_estimator = HeadPoseEstimator()
        self.stare_detector = StareDetector(
            fps=TARGET_FPS, 
            stare_duration_sec=STARE_DURATION_SEC
        )
        
        # Enhanced statistics tracking
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'total_people_tracked': 0,
            'stare_events_detected': 0,
            'stare_events_saved': 0,
            'same_side_filtered': 0,
            'duplicate_events_prevented': 0
        }
        
        print("   ✅ Person tracker initialized")
        print("   ✅ Head pose estimator initialized")  
        print("   ✅ Enhanced stare detector initialized")
        print("=" * 65)
        print("🎯 ENHANCED FEATURES:")
        print("   • Same-side filtering: Rejects people on same screen side")
        print("   • Duplicate prevention: Avoids saving multiple images of same event") 
        print("   • Single image type: Only saves stare_event_frame images")
        print("   • Distance filtering: People must be reasonably apart")
        print("   • Enhanced visualization: Red connecting lines and detailed labels")
        print("=" * 65)
        
        print("✅ SYSTEM READY!")
    
    def detect_people(self, frame):
        """Enhanced people detection using YOLOv8"""
        results = self.yolo_model(frame, classes=[0], conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = box.astype(int)
                
                # Focus on upper body/head region (top 40% of person detection)
                height = y2 - y1
                head_y2 = y1 + int(height * 0.4)
                
                # Ensure minimum detection size
                if (x2 - x1) > 30 and (head_y2 - y1) > 30:
                    detections.append({
                        'bbox': (x1, y1, x2, head_y2),
                        'confidence': conf,
                        'center': ((x1 + x2) // 2, (y1 + head_y2) // 2)
                    })
        
        return detections
    
    def analyze_frame(self, frame):
        """Comprehensive frame analysis with enhanced filtering"""
        frame_height, frame_width = frame.shape[:2]
        
        # Step 1: Detect people
        detections = self.detect_people(frame)
        self.stats['total_detections'] += len(detections)
        
        # Step 2: Update tracking
        tracked_persons = self.tracker.update(detections)
        if tracked_persons:
            self.stats['total_people_tracked'] = max(
                self.stats['total_people_tracked'], 
                max([p['id'] for p in tracked_persons]) + 1
            )
        
        # Step 3: Estimate head poses
        for person in tracked_persons:
            yaw_angle = self.head_pose_estimator.estimate_yaw(frame, person['bbox'])
            person['yaw'] = yaw_angle
        
        # Step 4: Enhanced mutual staring detection
        staring_pairs = self.stare_detector.detect_mutual_stare(tracked_persons, frame_width)
        
        return tracked_persons, staring_pairs
    
    def process_video(self, video_path):
        """Main enhanced video processing loop"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ ERROR: Could not open video file: {video_path}")
            print("   Please check the file path and ensure the video file exists.")
            return False
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / original_fps if original_fps > 0 else 0
        
        print(f"\n📹 VIDEO ANALYSIS:")
        print(f"   Resolution: {width}x{height}")
        print(f"   Original FPS: {original_fps:.1f}")
        print(f"   Total frames: {total_frames:,}")
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Processing FPS: {TARGET_FPS} (every {int(original_fps/TARGET_FPS) if original_fps > 0 else 1} frames)")
        
        if ENHANCED_FILTERING:
            print(f"   Screen regions: LEFT (0-{width//2}), RIGHT ({width//2}-{width})")
            print("   Enhanced filtering: ENABLED")
        
        # Calculate frame skip for target FPS
        frame_skip = max(1, int(original_fps / TARGET_FPS)) if original_fps > 0 else 1
        
        # Processing statistics
        processing_start_time = time.time()
        frame_idx = 0
        processed_frames = 0
        last_stats_update = 0
        
        print(f"\n🎬 STARTING VIDEO PROCESSING...")
        print("   Controls: 'q'=quit, 'p'=pause, 'd'=debug, 's'=save current frame")
        print("-" * 65)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("   📹 End of video reached")
                    break
                
                frame_idx += 1
                
                # Skip frames to achieve target FPS
                if frame_idx % frame_skip != 0:
                    continue
                
                processed_frames += 1
                self.stats['frames_processed'] = processed_frames
                frame_start_time = time.time()
                
                # Analyze frame with enhanced detection
                tracked_persons, staring_pairs = self.analyze_frame(frame)
                
                # Handle staring events with duplicate prevention
                current_staring_ids = set()
                new_events = 0
                
                for pair in staring_pairs:
                    person1, person2 = pair
                    current_staring_ids.add(person1['id'])
                    current_staring_ids.add(person2['id'])
                    
                    # Check if we should save this event (enhanced duplicate prevention)
                    if (SAVE_STARE_EVENTS and 
                        self.stare_detector.should_save_event(person1['id'], person2['id'], processed_frames)):
                        
                        self.handle_stare_event(frame, person1, person2, processed_frames)
                        self.stats['stare_events_saved'] += 1
                        new_events += 1
                    
                    self.stats['stare_events_detected'] += 1
                
                # Create enhanced visualization
                annotated_frame = self.create_enhanced_visualization(
                    frame, tracked_persons, current_staring_ids, processed_frames
                )
                
                # Display frame
                cv2.imshow('Enhanced Mutual Staring Detection System', annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("   🛑 User requested quit")
                    break
                elif key == ord('p'):
                    print("   ⏸️ PAUSED - Press any key to continue...")
                    cv2.waitKey(0)
                    print("   ▶️ RESUMED")
                elif key == ord('d') and SHOW_DEBUG_INFO:
                    self.print_detailed_debug_info(tracked_persons, staring_pairs, processed_frames, width)
                elif key == ord('s'):
                    self.save_current_frame(annotated_frame, processed_frames)
                elif key == ord('r'):  # Reset statistics
                    self.reset_statistics()
                    print("   🔄 Statistics reset")
                
                # Periodic statistics update
                if processed_frames - last_stats_update >= 30:  # Every 15 seconds at 2 FPS
                    self.print_progress_stats(processed_frames, total_frames // frame_skip, processing_start_time)
                    last_stats_update = processed_frames
                
                # Control processing speed
                processing_time = time.time() - frame_start_time
                target_time = 1.0 / TARGET_FPS
                if processing_time < target_time:
                    time.sleep(target_time - processing_time)
        
        except KeyboardInterrupt:
            print("   ⚠️ Processing interrupted by user")
        except Exception as e:
            print(f"   ❌ Error during processing: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.print_final_comprehensive_stats(processing_start_time)
        
        return True
    
    def handle_stare_event(self, frame, person1, person2, frame_num):
        """Handle detected staring event with enhanced logging"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        event_msg = (f"👀 [{timestamp}] Frame {frame_num}: "
                    f"Person_{person1['id']} ↔ Person_{person2['id']} - MUTUAL STARING DETECTED!")
        
        print(event_msg)
        
        # Enhanced logging
        if logger and ENABLE_LOGGING:
            log_stare_event(logger, person1, person2, frame_num)
        
        # Save enhanced stare event image
        if SAVE_STARE_EVENTS:
            save_stare_event(frame, person1, person2, frame_num, OUTPUT_DIR)
    
    def save_current_frame(self, frame, frame_num):
        """Save current frame manually (for debugging)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"manual_save_frame{frame_num}_{timestamp}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        success = cv2.imwrite(filepath, frame)
        if success:
            print(f"   💾 Manual save: {filename}")
        else:
            print(f"   ❌ Failed to save: {filename}")
    
    def create_enhanced_visualization(self, frame, tracked_persons, staring_ids, frame_num):
        """Create enhanced frame visualization with detailed annotations"""
        annotated_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        # Draw screen division line for debugging
        if SHOW_DEBUG_INFO:
            cv2.line(annotated_frame, (frame_width//2, 0), (frame_width//2, frame_height), (100, 100, 100), 1)
            cv2.putText(annotated_frame, "L", (frame_width//4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            cv2.putText(annotated_frame, "R", (3*frame_width//4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        
        # Draw person annotations
        for person in tracked_persons:
            x1, y1, x2, y2 = person['bbox']
            person_id = person['id']
            yaw = person.get('yaw', 0)
            center_x = (x1 + x2) // 2
            
            # Determine side and color
            is_staring = person_id in staring_ids
            color = (0, 0, 255) if is_staring else (0, 255, 0)  # Red if staring, green otherwise
            thickness = 4 if is_staring else 2
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Create enhanced label
            side = "L" if center_x < frame_width//2 else "R"
            direction_symbol = "→" if yaw > 5 else "←" if yaw < -5 else "↑"
            
            label_parts = [f"P{person_id}({side})"]
            if abs(yaw) > 5:
                label_parts.append(f"{direction_symbol}{abs(yaw):.0f}°")
            if is_staring:
                label_parts.append("[STARING]")
            
            label = " ".join(label_parts)
            
            # Draw label with background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            label_bg_height = 30
            
            cv2.rectangle(annotated_frame, (x1, y1 - label_bg_height), 
                         (x1 + label_size[0] + 10, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1 + 5, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw direction arrow for significant head turns
            if abs(yaw) > 15:
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                arrow_length = 50
                end_x = center[0] + int(arrow_length * (1 if yaw > 0 else -1))
                cv2.arrowedLine(annotated_frame, center, (end_x, center[1]), color, 3)
        
        # Add comprehensive info overlay
        self.add_comprehensive_info_overlay(annotated_frame, frame_num, tracked_persons, staring_ids)
        
        return annotated_frame
    
    def add_comprehensive_info_overlay(self, frame, frame_num, tracked_persons, staring_ids):
        """Add comprehensive information overlay to frame"""
        frame_height, frame_width = frame.shape[:2]
        num_staring = len(staring_ids)
        num_staring_pairs = num_staring // 2
        
        # Top status bar
        status_text = f"Frame: {frame_num} | People: {len(tracked_persons)} | Staring: {num_staring_pairs} pairs | FPS: {TARGET_FPS}"
        cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Enhanced features indicator
        if ENHANCED_FILTERING:
            cv2.putText(frame, "Enhanced: Same-side filtering + Duplicate prevention", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Staring notification with animation effect
        if num_staring_pairs > 0:
            # Flashing red background for staring alert
            alert_alpha = 0.3 + 0.2 * abs(math.sin(time.time() * 4))  # Pulsing effect
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame_height - 80), (frame_width, frame_height - 40), (0, 0, 255), -1)
            cv2.addWeighted(overlay, alert_alpha, frame, 1 - alert_alpha, 0, frame)
            
            alert_text = f"🔴 MUTUAL STARING DETECTED: {num_staring_pairs} PAIR(S) 🔴"
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 3)[0]
            text_x = (frame_width - text_size[0]) // 2
            cv2.putText(frame, alert_text, (text_x, frame_height - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        
        # Statistics summary
        stats_y = frame_height - 120
        stats_text = f"Saved: {self.stats['stare_events_saved']} | Total tracked: {self.stats['total_people_tracked']}"
        cv2.putText(frame, stats_text, (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Controls help
        controls_text = "Controls: 'q'=quit | 'p'=pause | 'd'=debug | 's'=save | 'r'=reset stats"
        cv2.putText(frame, controls_text, (10, frame_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    
    def print_detailed_debug_info(self, tracked_persons, staring_pairs, frame_num, frame_width):
        """Print comprehensive debug information"""
        print(f"\n{'='*60}")
        print(f"🐛 DETAILED DEBUG INFO - Frame {frame_num}")
        print(f"{'='*60}")
        
        print(f"📊 CURRENT FRAME ANALYSIS:")
        print(f"   Tracked persons: {len(tracked_persons)}")
        print(f"   Active staring pairs: {len(staring_pairs)}")
        print(f"   Frame width: {frame_width}px (Middle: {frame_width//2}px)")
        
        if tracked_persons:
            print(f"\n👥 TRACKED PERSONS:")
            for person in tracked_persons:
                bbox = person['bbox']
                center_x = (bbox[0] + bbox[2]) // 2
                yaw = person.get('yaw', 0)
                
                side = "LEFT" if center_x < frame_width//2 else "RIGHT"
                direction = "RIGHT" if yaw > 5 else "LEFT" if yaw < -5 else "FORWARD"
                
                print(f"   Person_{person['id']}: {side} side (x={center_x}), looking {direction} ({yaw:.1f}°)")
        
        if staring_pairs:
            print(f"\n👀 STARING PAIRS:")
            for i, (p1, p2) in enumerate(staring_pairs, 1):
                print(f"   Pair {i}: Person_{p1['id']} ↔ Person_{p2['id']}")
        
        # Show stare detector statistics
        stare_stats = self.stare_detector.get_stare_statistics()
        print(f"\n📈 STARE DETECTION STATISTICS:")
        for key, value in stare_stats.items():
            if key != 'stare_counters':  # Skip the detailed counter dict
                print(f"   {key}: {value}")
        
        print(f"{'='*60}\n")
    
    def print_progress_stats(self, current_frame, total_estimated_frames, start_time):
        """Print processing progress statistics"""
        elapsed = time.time() - start_time
        progress_percent = (current_frame / total_estimated_frames * 100) if total_estimated_frames > 0 else 0
        fps_actual = current_frame / elapsed if elapsed > 0 else 0
        
        print(f"📊 Progress: {current_frame:,}/{total_estimated_frames:,} frames ({progress_percent:.1f}%) | "
              f"Elapsed: {elapsed:.0f}s | Actual FPS: {fps_actual:.1f}")
    
    def reset_statistics(self):
        """Reset all tracking statistics"""
        for key in self.stats:
            self.stats[key] = 0
        self.stare_detector.reset()
    
    def print_final_comprehensive_stats(self, start_time):
        """Print comprehensive final statistics"""
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"📊 COMPREHENSIVE FINAL STATISTICS")
        print(f"{'='*70}")
        
        print(f"⏱️  PROCESSING TIME:")
        print(f"   Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"   Average FPS: {self.stats['frames_processed']/elapsed:.2f}")
        
        print(f"\n📹 VIDEO ANALYSIS:")
        print(f"   Frames processed: {self.stats['frames_processed']:,}")
        print(f"   Total detections: {self.stats['total_detections']:,}")
        print(f"   Unique people tracked: {self.stats['total_people_tracked']}")
        
        print(f"\n👀 STARING EVENTS:")
        print(f"   Staring events detected: {self.stats['stare_events_detected']}")
        print(f"   Unique events saved: {self.stats['stare_events_saved']}")
        print(f"   Duplicate events prevented: {self.stats['stare_events_detected'] - self.stats['stare_events_saved']}")
        
        # Get final detector statistics
        final_stare_stats = self.stare_detector.get_stare_statistics()
        print(f"   Active tracking pairs: {final_stare_stats['active_pairs']}")
        print(f"   Confirmed staring pairs: {final_stare_stats['confirmed_pairs']}")
        
        print(f"\n📁 OUTPUT:")
        if self.stats['stare_events_saved'] > 0:
            print(f"   Saved images: {self.stats['stare_events_saved']} files in {OUTPUT_DIR}")
        else:
            print(f"   No staring events detected/saved")
        
        if ENABLE_LOGGING:
            print(f"   Log file: {LOGS_DIR}")
        
        print(f"\n🔧 ENHANCED FEATURES SUMMARY:")
        print(f"   ✅ Same-side filtering: Prevented invalid detections")
        print(f"   ✅ Duplicate prevention: Avoided repeated saves")
        print(f"   ✅ Enhanced visualization: Red lines and detailed labels")
        print(f"   ✅ Only stare_event_frame images: No individual face crops")
        
        efficiency = (self.stats['stare_events_saved'] / self.stats['frames_processed'] * 100) if self.stats['frames_processed'] > 0 else 0
        print(f"\n📈 DETECTION EFFICIENCY:")
        print(f"   Staring event rate: {efficiency:.3f}% of processed frames")
        
        print(f"{'='*70}")
        print(f"✅ ENHANCED MUTUAL STARING DETECTION COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")


def main():
    """Enhanced main function with comprehensive setup"""
    print("🚀 ENHANCED MUTUAL STARING DETECTION SYSTEM")
    print("=" * 70)
    
    # Validate video file
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ ERROR: Video file not found!")
        print(f"   Expected path: {VIDEO_PATH}")
        print(f"   Please:")
        print(f"   1. Check the file path is correct")
        print(f"   2. Ensure the video file exists") 
        print(f"   3. Update VIDEO_PATH in the script configuration")
        input("Press Enter to exit...")
        return
    
    # Display configuration
    print(f"⚙️  CONFIGURATION:")
    print(f"   Video: {os.path.basename(VIDEO_PATH)}")
    print(f"   Output: {OUTPUT_DIR}")
    print(f"   Target FPS: {TARGET_FPS}")
    print(f"   Stare duration: {STARE_DURATION_SEC}s")
    print(f"   Enhanced filtering: {'ON' if ENHANCED_FILTERING else 'OFF'}")
    
    print(f"\n🎯 ENHANCED FEATURES:")
    print(f"   • Same-side filtering: People on same screen side are ignored")
    print(f"   • Duplicate prevention: Same event won't be saved multiple times")
    print(f"   • Single image type: Only stare_event_frame*.jpg images")
    print(f"   • Distance filtering: People must be reasonably apart")
    print(f"   • Enhanced visualization: Red connecting lines and labels")
    
    # Initialize and run detector
    try:
        detector = EnhancedMutualStareDetector()
        success = detector.process_video(VIDEO_PATH)
        
        if success:
            print(f"\n🎉 Processing completed successfully!")
            if detector.stats['stare_events_saved'] > 0:
                print(f"   Check {OUTPUT_DIR} for saved stare event images")
        else:
            print(f"\n⚠️  Processing completed with issues")
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to exit...")

website_script = os.path.join(os.path.dirname(__file__), "website.py")
# Run website.py with the same Python interpreter
subprocess.Popen([sys.executable, website_script])
if __name__ == "__main__":
    main()