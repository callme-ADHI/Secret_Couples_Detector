# utils/tracker.py - Person tracking system

import numpy as np
import math
from collections import OrderedDict


class PersonTracker:
    def __init__(self, max_disappeared=10, max_distance=100):
        """
        Initialize the centroid tracker
        
        Args:
            max_disappeared: Maximum number of frames a person can disappear before deletion
            max_distance: Maximum distance for matching detections to existing tracks
        """
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        print(f"🎯 Person Tracker initialized:")
        print(f"   - Max disappeared frames: {max_disappeared}")
        print(f"   - Max matching distance: {max_distance}px")
    
    def register(self, detection):
        """Register a new person with unique ID"""
        self.objects[self.next_id] = {
            'id': self.next_id,
            'bbox': detection['bbox'],
            'center': detection['center'],
            'confidence': detection['confidence'],
            'history': [detection['center']],
            'yaw': 0.0  # Will be updated by pose estimator
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def deregister(self, object_id):
        """Remove a person from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dictionaries with 'bbox', 'center', 'confidence'
            
        Returns:
            List of tracked person objects
        """
        # If no detections, increment disappeared counter for all objects
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # Remove if disappeared too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return list(self.objects.values())
        
        # If no existing objects, register all detections as new
        if len(self.objects) == 0:
            for detection in detections:
                self.register(detection)
        else:
            # Match detections to existing objects
            object_centers = [obj['center'] for obj in self.objects.values()]
            object_ids = list(self.objects.keys())
            detection_centers = [det['center'] for det in detections]
            
            # Compute distance matrix
            distances = np.zeros((len(object_centers), len(detection_centers)))
            for i, obj_center in enumerate(object_centers):
                for j, det_center in enumerate(detection_centers):
                    distances[i][j] = self.calculate_distance(obj_center, det_center)
            
            # Find minimum distances for assignment
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            
            # Keep track of used indices
            used_row_indices = set()
            used_col_indices = set()
            
            # Update existing objects with matched detections
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                if distances[row, col] > self.max_distance:
                    continue
                
                # Update object with matched detection
                object_id = object_ids[row]
                detection = detections[col]
                
                self.objects[object_id]['bbox'] = detection['bbox']
                self.objects[object_id]['center'] = detection['center']
                self.objects[object_id]['confidence'] = detection['confidence']
                
                # Update position history (keep last 10 positions)
                self.objects[object_id]['history'].append(detection['center'])
                if len(self.objects[object_id]['history']) > 10:
                    self.objects[object_id]['history'].pop(0)
                
                # Reset disappeared counter
                self.disappeared[object_id] = 0
                
                used_row_indices.add(row)
                used_col_indices.add(col)
            
            # Handle unmatched detections and objects
            unused_row_indices = set(range(0, distances.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, distances.shape[1])).difference(used_col_indices)
            
            # If more objects than detections, mark objects as disappeared
            if distances.shape[0] >= distances.shape[1]:
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            
            # If more detections than objects, register new objects
            else:
                for col in unused_col_indices:
                    self.register(detections[col])
        
        return list(self.objects.values())
    
    def get_tracking_statistics(self):
        """Get current tracking statistics"""
        active_tracks = len(self.objects)
        disappeared_tracks = len([d for d in self.disappeared.values() if d > 0])
        
        return {
            'active_tracks': active_tracks,
            'disappeared_tracks': disappeared_tracks,
            'next_id': self.next_id,
            'total_ever_tracked': self.next_id
        }
    
    def smooth_position(self, object_id, alpha=0.7):
        """
        Apply exponential smoothing to object position
        
        Args:
            object_id: ID of the object to smooth
            alpha: Smoothing factor (0 = no smoothing, 1 = no history)
        """
        if object_id not in self.objects:
            return
        
        obj = self.objects[object_id]
        if len(obj['history']) < 2:
            return
        
        # Get current and previous positions
        current_pos = obj['history'][-1]
        prev_pos = obj['history'][-2]
        
        # Apply exponential smoothing
        smoothed_x = alpha * current_pos[0] + (1 - alpha) * prev_pos[0]
        smoothed_y = alpha * current_pos[1] + (1 - alpha) * prev_pos[1]
        
        # Update center with smoothed position
        obj['center'] = (int(smoothed_x), int(smoothed_y))
    
    def predict_next_position(self, object_id):
        """
        Predict next position based on movement history
        
        Args:
            object_id: ID of the object
            
        Returns:
            Predicted (x, y) position or None if not enough history
        """
        if object_id not in self.objects:
            return None
        
        obj = self.objects[object_id]
        if len(obj['history']) < 3:
            return None
        
        # Use last 3 positions to estimate velocity
        positions = obj['history'][-3:]
        
        # Calculate average velocity
        dx1 = positions[1][0] - positions[0][0]
        dy1 = positions[1][1] - positions[0][1]
        dx2 = positions[2][0] - positions[1][0]
        dy2 = positions[2][1] - positions[1][1]
        
        avg_dx = (dx1 + dx2) / 2
        avg_dy = (dy1 + dy2) / 2
        
        # Predict next position
        current_pos = positions[-1]
        predicted_x = current_pos[0] + avg_dx
        predicted_y = current_pos[1] + avg_dy
        
        return (int(predicted_x), int(predicted_y))
    
    def is_stationary(self, object_id, threshold=10):
        """
        Check if an object has been stationary
        
        Args:
            object_id: ID of the object
            threshold: Movement threshold in pixels
            
        Returns:
            True if object is stationary, False otherwise
        """
        if object_id not in self.objects:
            return False
        
        obj = self.objects[object_id]
        if len(obj['history']) < 5:
            return False
        
        # Check movement over last 5 positions
        positions = obj['history'][-5:]
        
        max_distance = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = self.calculate_distance(positions[i], positions[j])
                max_distance = max(max_distance, distance)
        
        return max_distance < threshold
    
    def cleanup_old_tracks(self):
        """Remove tracks that have been disappeared for too long"""
        to_remove = []
        for object_id, disappeared_count in self.disappeared.items():
            if disappeared_count > self.max_disappeared:
                to_remove.append(object_id)
        
        for object_id in to_remove:
            self.deregister(object_id)
        
        return len(to_remove)


def test_tracker():
    """Test function for person tracker"""
    print("🧪 Testing person tracker...")
    
    tracker = PersonTracker()
    
    # Simulate detections over multiple frames
    frames = [
        # Frame 1: Two people detected
        [
            {'bbox': (100, 100, 200, 200), 'center': (150, 150), 'confidence': 0.8},
            {'bbox': (300, 100, 400, 200), 'center': (350, 150), 'confidence': 0.9}
        ],
        # Frame 2: Same people, slightly moved
        [
            {'bbox': (105, 105, 205, 205), 'center': (155, 155), 'confidence': 0.8},
            {'bbox': (295, 105, 395, 205), 'center': (345, 155), 'confidence': 0.85}
        ],
        # Frame 3: One person disappeared
        [
            {'bbox': (110, 110, 210, 210), 'center': (160, 160), 'confidence': 0.75}
        ],
        # Frame 4: New person appears
        [
            {'bbox': (115, 115, 215, 215), 'center': (165, 165), 'confidence': 0.8},
            {'bbox': (500, 200, 600, 300), 'center': (550, 250), 'confidence': 0.9}
        ]
    ]
    
    for frame_idx, detections in enumerate(frames):
        print(f"\nFrame {frame_idx + 1}:")
        print(f"  Detections: {len(detections)}")
        
        tracked_objects = tracker.update(detections)
        
        print(f"  Tracked objects: {len(tracked_objects)}")
        for obj in tracked_objects:
            print(f"    Person_{obj['id']}: center={obj['center']}, conf={obj['confidence']:.2f}")
        
        stats = tracker.get_tracking_statistics()
        print(f"  Stats: {stats}")
    
    print("\n✅ Tracker test completed!")


if __name__ == "__main__":
    test_tracker()