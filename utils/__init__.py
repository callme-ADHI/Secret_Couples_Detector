# utils/__init__.py - Package initialization for utilities

"""
Utilities package for Mutual Staring Detection System

This package contains:
- pose_estimation.py: Head pose estimation using MediaPipe and OpenCV
- tracker.py: Person tracking across video frames
"""

from .pose_estimation import HeadPoseEstimator
from .tracker import PersonTracker

__all__ = ['HeadPoseEstimator', 'PersonTracker']
__version__ = '1.0.0'
__author__ = 'Mutual Staring Detection System'

print("📦 Utils package loaded successfully!")