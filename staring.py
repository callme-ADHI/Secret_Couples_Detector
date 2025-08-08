# staring.py - Enhanced staring detection with same-side filtering (COMPLETE VERSION)

import cv2
import os
import math
from collections import defaultdict
from datetime import datetime


class StareDetector:
    def __init__(self, fps=2, stare_duration_sec=1.0):
        """
        Initialize enhanced stare detection with improved filtering
        
        Args:
            fps: Processing frame rate
            stare_duration_sec: Minimum duration for staring event
        """
        self.fps = fps
        self.required_frames = max(1, int(fps * stare_duration_sec))
        self.stare_counters = defaultdict(int)
        self.saved_events = set()  # Track saved events to avoid duplicates
        
        # Enhanced detection parameters
        self.min_yaw_difference = 60  # Minimum angle difference for "facing each other"
        self.max_yaw_difference = 120  # Maximum angle difference (too extreme)
        self.min_head_turn = 15  # Minimum individual head turn to be considered
        self.min_distance_threshold = 100  # Minimum distance between people (pixels)
        self.same_side_margin = 50  # Margin for same-side detection
        
        print(f"🎯 Enhanced Stare Detector initialized:")
        print(f"   - Required duration: {stare_duration_sec}s ({self.required_frames} frames)")
        print(f"   - Yaw difference range: {self.min_yaw_difference}° - {self.max_yaw_difference}°")
        print(f"   - Minimum head turn: {self.min_head_turn}°")
        print(f"   - Minimum distance: {self.min_distance_threshold}px")
        print(f"   - Same-side filtering: ENABLED")
    
    def are_on_different_sides(self, person1, person2, frame_width):
        """
        Check if two people are on different sides of the screen
        This is crucial - both people can't be on the same side!
        
        Args:
            person1, person2: Person dictionaries with bbox info
            frame_width: Width of the video frame
            
        Returns:
            (bool, str): (True if different sides, explanation message)
        """
        # Get centers of both people
        bbox1 = person1['bbox']
        bbox2 = person2['bbox']
        center1_x = (bbox1[0] + bbox1[2]) // 2
        center2_x = (bbox2[0] + bbox2[2]) // 2
        
        # Define screen regions with margin
        screen_middle = frame_width // 2
        left_boundary = screen_middle - self.same_side_margin
        right_boundary = screen_middle + self.same_side_margin
        
        # Classify positions
        person1_side = "LEFT" if center1_x < left_boundary else "RIGHT" if center1_x > right_boundary else "CENTER"
        person2_side = "LEFT" if center2_x < left_boundary else "RIGHT" if center2_x > right_boundary else "CENTER"
        
        # Check for same side (invalid for staring)
        if person1_side == person2_side and person1_side != "CENTER":
            return False, f"Both on {person1_side} side: P1_x={center1_x}, P2_x={center2_x}, middle={screen_middle}"
        
        # Center positions are acceptable if one is clearly left/right of the other
        if person1_side == "CENTER" or person2_side == "CENTER":
            if abs(center1_x - center2_x) < self.same_side_margin:
                return False, f"Too close to center: P1_x={center1_x}, P2_x={center2_x}"
        
        return True, f"Different sides: P1={person1_side}({center1_x}), P2={person2_side}({center2_x})"
    
    def calculate_distance(self, person1, person2):
        """Calculate distance between two people's centers"""
        bbox1 = person1['bbox']
        bbox2 = person2['bbox']
        center1 = ((bbox1[0] + bbox1[2]) // 2, (bbox1[1] + bbox1[3]) // 2)
        center2 = ((bbox2[0] + bbox2[2]) // 2, (bbox2[1] + bbox2[3]) // 2)
        
        distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return distance, center1, center2
    
    def are_facing_each_other(self, person1, person2, frame_width):
        """
        Enhanced logic to determine if two people are facing each other
        
        FILTERING RULES:
        1. Must be on different sides of screen (LEFT vs RIGHT)
        2. Must be reasonable distance apart
        3. Both must have significant head movement
        4. Head directions must be logical based on positions
        
        Args:
            person1, person2: Person dictionaries with 'id', 'bbox', 'yaw'
            frame_width: Width of the video frame
            
        Returns:
            (bool, str): (True if facing each other, detailed reason)
        """
        yaw1 = person1.get('yaw', 0)
        yaw2 = person2.get('yaw', 0)
        
        # FILTER 1: Must be on different sides of screen
        different_sides, side_reason = self.are_on_different_sides(person1, person2, frame_width)
        if not different_sides:
            return False, f"❌ Same side filter: {side_reason}"
        
        # FILTER 2: Calculate distance and positions
        distance, center1, center2 = self.calculate_distance(person1, person2)
        if distance < self.min_distance_threshold:
            return False, f"❌ Too close: distance={distance:.1f}px < {self.min_distance_threshold}px"
        
        # FILTER 3: Both should be turning their heads significantly
        if abs(yaw1) < self.min_head_turn:
            return False, f"❌ P1 insufficient head turn: {yaw1:.1f}° < {self.min_head_turn}°"
        if abs(yaw2) < self.min_head_turn:
            return False, f"❌ P2 insufficient head turn: {yaw2:.1f}° < {self.min_head_turn}°"
        
        # FILTER 4: Logical facing directions based on positions
        person1_left = center1[0] < center2[0]
        
        if person1_left:
            # Person1 is LEFT, Person2 is RIGHT
            # Person1 should look RIGHT (positive yaw), Person2 should look LEFT (negative yaw)
            correct_directions = yaw1 > self.min_head_turn and yaw2 < -self.min_head_turn
            scenario = f"P1(LEFT) looking RIGHT({yaw1:.1f}°), P2(RIGHT) looking LEFT({yaw2:.1f}°)"
        else:
            # Person1 is RIGHT, Person2 is LEFT  
            # Person1 should look LEFT (negative yaw), Person2 should look RIGHT (positive yaw)
            correct_directions = yaw1 < -self.min_head_turn and yaw2 > self.min_head_turn
            scenario = f"P1(RIGHT) looking LEFT({yaw1:.1f}°), P2(LEFT) looking RIGHT({yaw2:.1f}°)"
        
        # FILTER 5: Alternative check - approximately opposite angles
        yaw_difference = abs(yaw1 - yaw2)
        if yaw_difference > 180:
            yaw_difference = 360 - yaw_difference
        
        opposite_angles = (self.min_yaw_difference <= yaw_difference <= self.max_yaw_difference)
        
        # FINAL DECISION: Either correct directions OR opposite angles
        is_facing = correct_directions or opposite_angles
        
        if is_facing:
            success_reason = (
                f"✅ STARING DETECTED: {scenario}, distance={distance:.1f}px, "
                f"yaw_diff={yaw_difference:.1f}°, logical={correct_directions}, opposite={opposite_angles}"
            )
            return True, success_reason
        else:
            fail_reason = (
                f"❌ Not facing: {scenario}, distance={distance:.1f}px, "
                f"yaw_diff={yaw_difference:.1f}°, logical={correct_directions}, opposite={opposite_angles}"
            )
            return False, fail_reason
    
    def detect_mutual_stare(self, tracked_persons, frame_width):
        """
        Detect mutual staring between all pairs with enhanced filtering
        
        Args:
            tracked_persons: List of person dictionaries with 'id', 'bbox', 'yaw'
            frame_width: Width of the video frame
            
        Returns:
            List of staring pairs (tuples of person dicts)
        """
        staring_pairs = []
        
        if len(tracked_persons) < 2:
            return staring_pairs
        
        # Check all possible pairs
        for i in range(len(tracked_persons)):
            for j in range(i + 1, len(tracked_persons)):
                person1 = tracked_persons[i]
                person2 = tracked_persons[j]
                
                # Create unique pair key for tracking
                pair_key = tuple(sorted([person1['id'], person2['id']]))
                
                # Apply enhanced facing detection logic
                facing, reason = self.are_facing_each_other(person1, person2, frame_width)
                
                if facing:
                    # Increment consecutive staring counter
                    self.stare_counters[pair_key] += 1
                    
                    # Check if minimum duration requirement is met
                    if self.stare_counters[pair_key] >= self.required_frames:
                        staring_pairs.append((person1, person2))
                        
                        # Debug output only on first detection
                        if self.stare_counters[pair_key] == self.required_frames:
                            print(f"   🎯 NEW STARING PAIR: Person_{person1['id']} ↔ Person_{person2['id']}")
                            print(f"      {reason}")
                    else:
                        # Still building up to required duration
                        frames_left = self.required_frames - self.stare_counters[pair_key]
                        print(f"   ⏳ Building stare: P{person1['id']} ↔ P{person2['id']} ({frames_left} frames left)")
                else:
                    # Not facing - reset or decay counter
                    if pair_key in self.stare_counters:
                        old_count = self.stare_counters[pair_key]
                        self.stare_counters[pair_key] = max(0, self.stare_counters[pair_key] - 2)  # Faster decay
                        
                        if old_count > 0 and self.stare_counters[pair_key] == 0:
                            print(f"   ❌ Lost stare: P{person1['id']} ↔ P{person2['id']} - {reason}")
        
        return staring_pairs
    
    def should_save_event(self, person1_id, person2_id, frame_num):
        """
        Prevent duplicate saves of the same staring event
        
        Args:
            person1_id, person2_id: IDs of the staring persons
            frame_num: Current frame number
            
        Returns:
            bool: True if should save, False if already saved recently
        """
        # Create standardized pair key
        pair_key = tuple(sorted([person1_id, person2_id]))
        
        # Check if we've saved this pair recently (within last 30 frames = ~15 seconds at 2 FPS)
        recent_threshold = 30
        recent_saves = [
            event_key for event_key in self.saved_events 
            if event_key[0] == pair_key and (frame_num - event_key[1]) < recent_threshold
        ]
        
        if recent_saves:
            last_save_frame = max([event_key[1] for event_key in recent_saves])
            frames_since = frame_num - last_save_frame
            print(f"   ⏭️  Skipping duplicate save: P{person1_id} ↔ P{person2_id} (saved {frames_since} frames ago)")
            return False
        
        # Mark as saved
        event_key = (pair_key, frame_num)
        self.saved_events.add(event_key)
        
        # Clean up old events (remove events older than 100 frames)
        cleanup_threshold = 100
        self.saved_events = {
            key for key in self.saved_events 
            if (frame_num - key[1]) < cleanup_threshold
        }
        
        return True
    
    def get_stare_statistics(self):
        """Get comprehensive staring detection statistics"""
        active_pairs = len([count for count in self.stare_counters.values() if count > 0])
        confirmed_pairs = len([count for count in self.stare_counters.values() if count >= self.required_frames])
        
        return {
            'active_pairs': active_pairs,
            'confirmed_pairs': confirmed_pairs,
            'total_tracked_pairs': len(self.stare_counters),
            'total_saved_events': len(self.saved_events),
            'stare_counters': dict(self.stare_counters),  # For detailed debugging
        }
    
    def reset(self):
        """Reset all staring counters and saved events"""
        self.stare_counters.clear()
        self.saved_events.clear()
        print("🔄 Stare detector reset")


def save_stare_event(frame, person1, person2, frame_num, output_dir):
    """
    Save ONLY the stare event frame with enhanced visualization
    NO individual face crops - just the full frame with annotations
    
    Args:
        frame: Current video frame
        person1, person2: Person dictionaries with bbox info
        frame_num: Current frame number
        output_dir: Directory to save images
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename for stare event
    event_filename = f"stare_event_frame{frame_num}_P{person1['id']}_P{person2['id']}_{timestamp}.jpg"
    full_frame_path = os.path.join(output_dir, event_filename)
    
    # Create annotated frame copy
    save_frame = frame.copy()
    
    # Get centers and info for both persons
    persons_info = []
    for person in [person1, person2]:
        x1, y1, x2, y2 = person['bbox']
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        yaw = person.get('yaw', 0)
        persons_info.append({
            'bbox': (x1, y1, x2, y2),
            'center': center,
            'id': person['id'],
            'yaw': yaw
        })
    
    # Draw enhanced annotations
    for person_info in persons_info:
        x1, y1, x2, y2 = person_info['bbox']
        center = person_info['center']
        person_id = person_info['id']
        yaw = person_info['yaw']
        
        # Draw RED bounding box for staring event
        cv2.rectangle(save_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        
        # Create detailed label
        direction_symbol = "→" if yaw > 0 else "←" if yaw < 0 else "↑"
        label = f"Person_{person_id} {direction_symbol} ({yaw:.0f}°) [STARING]"
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(save_frame, (x1, y1 - 35), (x1 + label_size[0] + 10, y1), (0, 0, 255), -1)
        
        # Draw label text
        cv2.putText(save_frame, label, (x1 + 5, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw center point
        cv2.circle(save_frame, center, 8, (0, 255, 255), -1)
    
    # Draw connecting line between the staring persons
    center1 = persons_info[0]['center']
    center2 = persons_info[1]['center']
    
    # Main connecting line
    cv2.line(save_frame, center1, center2, (0, 0, 255), 4)
    
    # Draw "STARING" text at midpoint
    mid_x = (center1[0] + center2[0]) // 2
    mid_y = (center1[1] + center2[1]) // 2
    
    stare_text = "MUTUAL STARING"
    text_size = cv2.getTextSize(stare_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
    
    # Background for staring text
    cv2.rectangle(save_frame, 
                  (mid_x - text_size[0]//2 - 10, mid_y - 20), 
                  (mid_x + text_size[0]//2 + 10, mid_y + 10), 
                  (0, 0, 255), -1)
    
    # Staring text
    cv2.putText(save_frame, stare_text, (mid_x - text_size[0]//2, mid_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    
    # Add event header information
    header_text = f"MUTUAL STARE EVENT - Frame {frame_num} - {timestamp}"
    cv2.putText(save_frame, header_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Add distance information
    distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    distance_text = f"Distance: {distance:.0f}px"
    cv2.putText(save_frame, distance_text, (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Save the enhanced frame
    success = cv2.imwrite(full_frame_path, save_frame)
    
    if success:
        print(f"   💾 SAVED: {event_filename}")
        print(f"   📍 Location: {output_dir}")
    else:
        print(f"   ❌ FAILED to save: {event_filename}")


def log_stare_event(logger, person1, person2, frame_num):
    """
    Log detailed staring event information
    
    Args:
        logger: Logger instance
        person1, person2: Person dictionaries
        frame_num: Current frame number
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract detailed information
    bbox1 = person1['bbox']
    bbox2 = person2['bbox']
    center1 = ((bbox1[0] + bbox1[2]) // 2, (bbox1[1] + bbox1[3]) // 2)
    center2 = ((bbox2[0] + bbox2[2]) // 2, (bbox2[1] + bbox2[3]) // 2)
    
    yaw1 = person1.get('yaw', 0)
    yaw2 = person2.get('yaw', 0)
    
    distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    # Create comprehensive log message
    log_message = (
        f"🔴 MUTUAL STARE EVENT 🔴 | "
        f"Frame: {frame_num} | "
        f"Time: {timestamp} | "
        f"Person_{person1['id']} (pos: {center1}, yaw: {yaw1:.1f}°) ↔ "
        f"Person_{person2['id']} (pos: {center2}, yaw: {yaw2:.1f}°) | "
        f"Distance: {distance:.1f}px | "
        f"Duration: ≥1.0s"
    )
    
    logger.info(log_message)


def test_enhanced_stare_detection():
    """
    Comprehensive test function for enhanced stare detection logic
    Tests all the new filtering rules
    """
    print("🧪 TESTING ENHANCED STARE DETECTION LOGIC")
    print("=" * 60)
    
    detector = StareDetector(fps=2, stare_duration_sec=1.0)
    frame_width = 640  # Mock frame width for testing
    
    test_cases = [
        # TEST 1: Perfect valid staring scenario
        {
            'person1': {'id': 1, 'bbox': (50, 100, 150, 200), 'yaw': 30},   # Left side, looking right
            'person2': {'id': 2, 'bbox': (450, 100, 550, 200), 'yaw': -30}, # Right side, looking left
            'expected': True,
            'description': "✅ VALID: Left person looking right, right person looking left"
        },
        
        # TEST 2: Both on LEFT side - SHOULD BE REJECTED
        {
            'person1': {'id': 1, 'bbox': (50, 100, 150, 200), 'yaw': 25},   # Left side
            'person2': {'id': 2, 'bbox': (200, 100, 300, 200), 'yaw': -25}, # Also left side
            'expected': False,
            'description': "❌ INVALID: Both people on LEFT side of screen"
        },
        
        # TEST 3: Both on RIGHT side - SHOULD BE REJECTED
        {
            'person1': {'id': 1, 'bbox': (400, 100, 500, 200), 'yaw': 25},  # Right side
            'person2': {'id': 2, 'bbox': (500, 100, 600, 200), 'yaw': -25}, # Also right side
            'expected': False,
            'description': "❌ INVALID: Both people on RIGHT side of screen"
        },
        
        # TEST 4: Too close together - SHOULD BE REJECTED
        {
            'person1': {'id': 1, 'bbox': (280, 100, 330, 200), 'yaw': 25},  # Near center
            'person2': {'id': 2, 'bbox': (340, 100, 390, 200), 'yaw': -25}, # Too close
            'expected': False,
            'description': "❌ INVALID: People too close together (<100px)"
        },
        
        # TEST 5: Insufficient head movement - SHOULD BE REJECTED
        {
            'person1': {'id': 1, 'bbox': (100, 100, 200, 200), 'yaw': 5},   # Minimal turn
            'person2': {'id': 2, 'bbox': (400, 100, 500, 200), 'yaw': -10}, # Some turn but P1 insufficient
            'expected': False,
            'description': "❌ INVALID: Person 1 insufficient head movement (5° < 15°)"
        },
        
        # TEST 6: Wrong directions - SHOULD BE REJECTED
        {
            'person1': {'id': 1, 'bbox': (100, 100, 200, 200), 'yaw': -25}, # Left person looking more left
            'person2': {'id': 2, 'bbox': (400, 100, 500, 200), 'yaw': 25},  # Right person looking more right
            'expected': False,
            'description': "❌ INVALID: Wrong directions (both looking away from each other)"
        },
        
        # TEST 7: Extreme opposite angles - SHOULD BE ACCEPTED
        {
            'person1': {'id': 1, 'bbox': (100, 100, 200, 200), 'yaw': 45},  # Strong right turn
            'person2': {'id': 2, 'bbox': (450, 100, 550, 200), 'yaw': -40}, # Strong left turn
            'expected': True,
            'description': "✅ VALID: Strong opposite angles (85° difference)"
        }
    ]
    
    print(f"\nRunning {len(test_cases)} test cases...\n")
    
    passed = 0
    failed = 0
    
    for i, case in enumerate(test_cases, 1):
        print(f"TEST {i}: {case['description']}")
        
        facing, reason = detector.are_facing_each_other(case['person1'], case['person2'], frame_width)
        
        if facing == case['expected']:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        
        print(f"   Result: {status}")
        print(f"   Expected: {case['expected']}, Got: {facing}")
        print(f"   Reason: {reason}")
        print()
    
    print("=" * 60)
    print(f"📊 TEST RESULTS: {passed} PASSED, {failed} FAILED")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Enhanced filtering is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logic above.")
    
    print("=" * 60)


if __name__ == "__main__":
    test_enhanced_stare_detection()