# farmer_inventory_yolo11.py
# Advanced Farmer Inventory Management System with YOLOv11
# Complete implementation with YOLOv11 detection and Twilio notifications

import cv2
import numpy as np
import sqlite3
import threading
import time
from datetime import datetime
import json
from flask import Flask, render_template, request, jsonify, Response
from twilio.rest import Client
import os
from collections import defaultdict
import logging
import torch
from ultralytics import YOLO
import supervision as sv
from typing import Dict, List, Tuple, Optional
import math
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()  # This loads variables from .env file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Handle all database operations for inventory management"""
    
    def __init__(self, db_path="farmer_inventory.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create inventory table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_name TEXT UNIQUE NOT NULL,
                current_stock INTEGER DEFAULT 0,
                min_threshold INTEGER DEFAULT 10,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create detection log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_name TEXT,
                detected_count INTEGER,
                confidence REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create tracking table for line crossings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS crossing_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER,
                item_name TEXT,
                confidence REAL,
                crossing_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_item(self, item_name, initial_stock=0, min_threshold=10):
        """Add new item to inventory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO inventory 
                (item_name, current_stock, min_threshold) 
                VALUES (?, ?, ?)
            ''', (item_name, initial_stock, min_threshold))
            conn.commit()
            logger.info(f"Added/Updated item: {item_name} with stock: {initial_stock}")
            return True
        except Exception as e:
            logger.error(f"Error adding item: {e}")
            return False
        finally:
            conn.close()
    
    def update_stock(self, item_name, quantity_change):
        """Update stock quantity (positive for addition, negative for removal)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE inventory 
                SET current_stock = current_stock + ?, 
                    last_updated = CURRENT_TIMESTAMP
                WHERE item_name = ?
            ''', (quantity_change, item_name))
            conn.commit()
            
            # Get updated stock
            cursor.execute('SELECT current_stock FROM inventory WHERE item_name = ?', (item_name,))
            result = cursor.fetchone()
            new_stock = result[0] if result else 0
            
            logger.info(f"Updated {item_name} stock by {quantity_change}. New stock: {new_stock}")
            return new_stock
        except Exception as e:
            logger.error(f"Error updating stock: {e}")
            return None
        finally:
            conn.close()
    
    def get_inventory(self):
        """Get all inventory items"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT item_name, current_stock, min_threshold, last_updated 
            FROM inventory 
            ORDER BY item_name
        ''')
        
        items = []
        for row in cursor.fetchall():
            items.append({
                'name': row[0],
                'stock': row[1],
                'threshold': row[2],
                'last_updated': row[3],
                'low_stock': row[1] <= row[2]
            })
        
        conn.close()
        return items
    
    def log_detection(self, item_name, count, confidence):
        """Log detection event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detection_log (item_name, detected_count, confidence) 
            VALUES (?, ?, ?)
        ''', (item_name, count, confidence))
        
        conn.commit()
        conn.close()
    
    def log_crossing(self, track_id, item_name, confidence):
        """Log line crossing event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO crossing_log (track_id, item_name, confidence) 
            VALUES (?, ?, ?)
        ''', (track_id, item_name, confidence))
        
        conn.commit()
        conn.close()

class TwilioNotifier:
    """Handle Twilio SMS notifications"""
    
    def __init__(self, account_sid=None, auth_token=None, from_number=None, to_number=None):
        self.account_sid = account_sid or os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = auth_token or os.getenv('TWILIO_AUTH_TOKEN')
        self.from_number = from_number or os.getenv('TWILIO_FROM_NUMBER')
        self.to_number = to_number or os.getenv('TWILIO_TO_NUMBER')
        
        if all([self.account_sid, self.auth_token, self.from_number, self.to_number]):
            self.client = Client(self.account_sid, self.auth_token)
            self.enabled = True
        else:
            logger.warning("Twilio credentials not configured. SMS notifications disabled.")
            self.enabled = False
    
    def send_low_stock_alert(self, item_name, current_stock, threshold):
        """Send low stock alert"""
        if not self.enabled:
            logger.info(f"LOW STOCK ALERT: {item_name} has {current_stock} items (threshold: {threshold})")
            return False
        
        try:
            message = f" \n\n 🚨FARM INVENTORY ALERT 🚨\n\n{item_name} is running low!\nCurrent stock: {current_stock}\nMinimum threshold: {threshold}\n\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=self.to_number
            )
            logger.info(f"SMS alert sent for {item_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            return False

class YOLOv11Detector:
    """YOLOv11-based advanced item detection and tracking"""
    
    def __init__(self, model_path="yolo11n.pt", confidence_threshold=0.5):
        """
        Initialize YOLOv11 detector
        
        Args:
            model_path: Path to YOLO model (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)
            confidence_threshold: Minimum confidence for detections
        """
        try:
            self.model = YOLO(model_path)
            logger.info(f"YOLOv11 model loaded: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            # Fallback to a lighter model
            try:
                self.model = YOLO("yolo11n.pt")
                logger.info("Loaded fallback YOLOv11 nano model")
            except:
                raise Exception("Could not load any YOLO model. Please install ultralytics: pip install ultralytics")
        
        self.confidence_threshold = confidence_threshold
        
        # Mapping of COCO class names to our inventory items
        self.class_mapping = {
            # Fruits
            "apple": ["apple"],
            "orange": ["orange"],
            "banana": ["banana"],
            # Add more mappings as needed
            "carrot": ["carrot"],
            "broccoli": ["broccoli"],
            "tomato": ["tomato"],  # Custom trained classes
            "seeds": ["seeds"],    # Custom trained classes
        }
        
        # Reverse mapping for quick lookup
        self.coco_to_inventory = {}
        for inventory_name, coco_classes in self.class_mapping.items():
            for coco_class in coco_classes:
                self.coco_to_inventory[coco_class] = inventory_name
        
        # Detection line configuration
        self.detection_line_y = 300
        self.line_thickness = 3
        
        # Initialize ByteTrack for object tracking
        self.tracker = sv.ByteTrack()
        
        # Track objects that have crossed the line
        self.crossed_tracks = set()
        
        # Line crossing detection
        self.line_counter = sv.LineZone(
            start=sv.Point(0, self.detection_line_y),
            end=sv.Point(1920, self.detection_line_y)  # Adjust based on your resolution
        )
        
        # Annotators for visualization
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()
    
    def detect_and_track(self, frame):
        """
        Detect objects and track them across frames
        
        Returns:
            annotated_frame: Frame with annotations
            detections: Detection results
            tracked_objects: Tracked object information
        """
        # Run YOLO inference
        results = self.model(frame, verbose=False)[0]
        
        # Convert to supervision format
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter by confidence threshold
        detections = detections[detections.confidence > self.confidence_threshold]
        
        # Filter for relevant classes only
        relevant_detections = []
        for i, class_id in enumerate(detections.class_id):
            class_name = self.model.names[class_id].lower()
            if class_name in self.coco_to_inventory:
                relevant_detections.append(i)
        
        if relevant_detections:
            detections = detections[np.array(relevant_detections)]
        else:
            detections = sv.Detections.empty()
        
        # Update tracker
        detections = self.tracker.update_with_detections(detections)
        
        # Prepare tracking information
        tracked_objects = self._prepare_tracking_info(detections)
        
        # Check for line crossings
        line_crossings = self._check_line_crossings(detections)
        
        # Annotate frame
        annotated_frame = self._annotate_frame(frame, detections, tracked_objects)
        
        return annotated_frame, detections, tracked_objects, line_crossings
    
    def _prepare_tracking_info(self, detections):
        """Prepare tracking information for detected objects"""
        tracked_objects = defaultdict(list)
        
        if len(detections) > 0:
            for i, (bbox, confidence, class_id, track_id) in enumerate(
                zip(detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id)
            ):
                if track_id is not None:
                    class_name = self.model.names[class_id].lower()
                    inventory_name = self.coco_to_inventory.get(class_name, class_name)
                    
                    x1, y1, x2, y2 = bbox
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    tracked_objects[inventory_name].append({
                        'track_id': int(track_id),
                        'bbox': bbox,
                        'center': (center_x, center_y),
                        'confidence': float(confidence),
                        'class_name': class_name
                    })
        
        return dict(tracked_objects)
    
    def _check_line_crossings(self, detections):
        """Check for objects crossing the detection line"""
        line_crossings = defaultdict(int)
        
        if len(detections) > 0:
            # Update line counter
            line_counter_result = self.line_counter.trigger(detections)
            
            # Check each detection for line crossing
            for i, (bbox, confidence, class_id, track_id) in enumerate(
                zip(detections.xyxy, detections.confidence, detections.class_id, detections.tracker_id)
            ):
                if track_id is not None and line_counter_result[i]:
                    # Object crossed the line
                    if track_id not in self.crossed_tracks:
                        self.crossed_tracks.add(track_id)
                        class_name = self.model.names[class_id].lower()
                        inventory_name = self.coco_to_inventory.get(class_name, class_name)
                        line_crossings[inventory_name] += 1
                        
                        logger.info(f"Track ID {track_id} ({inventory_name}) crossed detection line")
            
            # Clean up old tracks (prevent memory buildup)
            if len(self.crossed_tracks) > 1000:
                self.crossed_tracks.clear()
        
        return dict(line_crossings)
    
    def _annotate_frame(self, frame, detections, tracked_objects):
        """Annotate frame with detection and tracking information"""
        annotated_frame = frame.copy()
        
        # Draw detection line
        cv2.line(annotated_frame, (0, self.detection_line_y), 
                (frame.shape[1], self.detection_line_y), 
                (0, 255, 0), self.line_thickness)
        
        # Add line label
        # Add line label
        cv2.putText(annotated_frame, "DETECTION LINE ", (10, self.detection_line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if len(detections) > 0:
        # Count objects by class for labeling
            class_counts = {}
        for class_id in detections.class_id:
            class_name = self.model.names[class_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
        # Create labels for detections with counts
        labels = []
        for i, (confidence, class_id, track_id) in enumerate(
            zip(detections.confidence, detections.class_id, detections.tracker_id)
        ):
            class_name = self.model.names[class_id]
            count = class_counts[class_name]
            
            # Show count instead of track ID (e.g., "Apple #1/2" instead of "#12 Apple")
            labels.append(f"{class_name} #{i+1}/{count} {confidence:.2f}")
            
            # Draw bounding boxes and labels
            annotated_frame = self.box_annotator.annotate(
                scene=annotated_frame, 
                detections=detections
            )
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame, 
                detections=detections, 
                labels=labels
            )
            
            # Draw tracking traces
            annotated_frame = self.trace_annotator.annotate(
                scene=annotated_frame, 
                detections=detections
            )
        
        # Draw count summary
        y_offset = 30
        total_count = 0
        for item_name, objects in tracked_objects.items():
            count = len(objects)
            total_count += count
            cv2.putText(annotated_frame, f"{item_name.title()}: {count}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 30
        
        # Draw total count
        cv2.putText(annotated_frame, f"Total Objects: {total_count}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Draw model info
        cv2.putText(annotated_frame, f"YOLOv11 | Conf: {self.confidence_threshold}", 
                   (frame.shape[1] - 300, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def update_detection_line(self, y_position):
        """Update detection line position"""
        self.detection_line_y = y_position
        self.line_counter = sv.LineZone(
            start=sv.Point(0, self.detection_line_y),
            end=sv.Point(1920, self.detection_line_y)
        )
        logger.info(f"Detection line updated to Y position: {y_position}")
    
    def update_confidence_threshold(self, threshold):
        """Update confidence threshold"""
        self.confidence_threshold = max(0.1, min(1.0, threshold))
        logger.info(f"Confidence threshold updated to: {self.confidence_threshold}")

class InventoryManager:
    """Main inventory management system with YOLOv11"""
    
    def __init__(self, yolo_model="yolo11n.pt"):
        self.db_manager = DatabaseManager()
        self.notifier = TwilioNotifier()
        
        try:
            self.detector = YOLOv11Detector(model_path=yolo_model)
            logger.info("YOLOv11 detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize YOLOv11 detector: {e}")
            raise
        
        self.camera = None
        self.is_monitoring = False
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Initialize with sample items for common COCO classes
        sample_items = [
            ("apple", 50, 10),
            ("orange", 30, 5),
            ("banana", 25, 8),
            ("carrot", 40, 12),
            ("broccoli", 20, 6),
            ("tomato", 35, 8),    # Custom class (requires custom training)
            ("seeds", 100, 20),   # Custom class (requires custom training)
        ]
        
        for item_name, stock, threshold in sample_items:
            self.db_manager.add_item(item_name, stock, threshold)
    
    def start_camera_monitoring(self, camera_index=0):
        """Start camera-based monitoring with YOLOv11"""
        try:
            self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                logger.error("Failed to open camera")
                return False
            
            # Set camera properties for better performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._yolo_monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            logger.info("YOLOv11 camera monitoring started")
            return True
        except Exception as e:
            logger.error(f"Error starting camera monitoring: {e}")
            return False
    
    def stop_camera_monitoring(self):
        """Stop camera monitoring"""
        self.is_monitoring = False
        if self.camera:
            self.camera.release()
        logger.info("Camera monitoring stopped")
    
    def _yolo_monitoring_loop(self):
        """Main monitoring loop with YOLOv11"""
        while self.is_monitoring:
            try:
                if self.camera and self.camera.isOpened():
                    ret, frame = self.camera.read()
                    if ret:
                        # Run YOLOv11 detection and tracking
                        annotated_frame, detections, tracked_objects, line_crossings = \
                            self.detector.detect_and_track(frame)
                        
                        # Update inventory for crossed objects
                        for item_name, count in line_crossings.items():
                            if count > 0:
                                new_stock = self.db_manager.update_stock(item_name, -count)
                                
                                # Log the detection with confidence
                                avg_confidence = self._get_average_confidence(tracked_objects.get(item_name, []))
                                self.db_manager.log_detection(item_name, count, avg_confidence)
                                
                                if new_stock is not None:
                                    # Check for low stock
                                    inventory = self.db_manager.get_inventory()
                                    for item in inventory:
                                        if item['name'] == item_name and item['low_stock']:
                                            self.notifier.send_low_stock_alert(
                                                item_name, item['stock'], item['threshold']
                                            )
                        
                        # Update FPS counter
                        self._update_fps_counter()
                        
                        # Store current frame for streaming
                        self.current_frame = annotated_frame
                
                time.sleep(0.03)  # ~30 FPS max
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)  # Wait before retrying
    
    def _get_average_confidence(self, objects):
        """Calculate average confidence for detected objects"""
        if not objects:
            return 0.0
        return sum(obj['confidence'] for obj in objects) / len(objects)
    
    def _update_fps_counter(self):
        """Update FPS counter"""
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def get_camera_frame(self):
        """Get current camera frame with YOLOv11 annotations"""
        if hasattr(self, 'current_frame'):
            # Add FPS info
            frame = self.current_frame.copy()
            cv2.putText(frame, f"FPS: {self.current_fps}", 
                       (frame.shape[1] - 100, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return frame
        return None
    
    def update_yolo_settings(self, confidence=None, line_position=None):
        """Update YOLOv11 detection settings"""
        if confidence is not None:
            self.detector.update_confidence_threshold(confidence)
        if line_position is not None:
            self.detector.update_detection_line(line_position)

# Flask Web Application with YOLOv11 support
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Initialize with YOLOv11 nano model (fastest)
# You can change to 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', or 'yolo11x.pt' for better accuracy
inventory_manager = InventoryManager(yolo_model="yolo11n.pt")

@app.route('/')
def api_info():
    """API information endpoint - React handles the frontend"""
    return jsonify({
        "message": "Farmer Inventory API Server with YOLOv11",
        "status": "running",
        "backend": "Flask",
        "frontend": "React (port 3000)",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/inventory')
def get_inventory():
    """Get current inventory data"""
    return jsonify(inventory_manager.db_manager.get_inventory())

@app.route('/api/add_item', methods=['POST'])
def add_item():
    """Add new item to inventory"""
    data = request.json
    success = inventory_manager.db_manager.add_item(
        data['name'], 
        data.get('stock', 0), 
        data.get('threshold', 10)
    )
    return jsonify({'success': success})

@app.route('/api/update_stock', methods=['POST'])
def update_stock():
    """Update item stock"""
    data = request.json
    new_stock = inventory_manager.db_manager.update_stock(
        data['item_name'], 
        data['quantity_change']
    )
    return jsonify({'success': new_stock is not None, 'new_stock': new_stock})

@app.route('/api/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start YOLOv11 camera monitoring"""
    success = inventory_manager.start_camera_monitoring()
    return jsonify({'success': success})

@app.route('/api/health')
def health_check():
    return jsonify({"status": "ok", "message": "Backend server is running"})



@app.route('/api/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stop camera monitoring"""
    inventory_manager.stop_camera_monitoring()
    return jsonify({'success': True})

@app.route('/api/update_yolo_settings', methods=['POST'])
def update_yolo_settings():
    """Update YOLOv11 detection settings"""
    data = request.json
    confidence = data.get('confidence')
    line_position = data.get('line_position')
    
    inventory_manager.update_yolo_settings(confidence, line_position)
    return jsonify({'success': True})

def generate_frames():
    """Generate camera frames for streaming with YOLOv11"""
    while True:
        frame = inventory_manager.get_camera_frame()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

@app.route('/api/video_feed')
def video_feed():
    """Video streaming route with YOLOv11"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("🤖 Advanced Farmer Inventory Management System with YOLOv11")
    print("=" * 65)
    print("\n📋 Setup Instructions:")
    print("1. Install required packages:")
    print("   pip install ultralytics opencv-python flask twilio supervision")
    print("\n2. YOLOv11 Models (choose one based on your hardware):")
    print("   • yolo11n.pt - Nano (fastest, least accurate)")
    print("   • yolo11s.pt - Small (balanced)")
    print("   • yolo11m.pt - Medium (more accurate)")
    print("   • yolo11l.pt - Large (very accurate)")
    print("   • yolo11x.pt - Extra Large (most accurate, slowest)")
    print("\n3. Set up Twilio credentials (optional):")
    print("   export TWILIO_ACCOUNT_SID='your_account_sid'")
    print("   export TWILIO_AUTH_TOKEN='your_auth_token'")
    print("   export TWILIO_FROM_NUMBER='+1234567890'")
    print("   export TWILIO_TO_NUMBER='+0987654321'")
    print("\n4. Connect camera and run: python farmer_inventory_yolo11.py")
    print("5. Open browser: http://localhost:5000")
    
    print("\n🎯 YOLOv11 Features:")
    print("• State-of-the-art object detection accuracy")
    print("• Real-time object tracking with ByteTrack")
    print("• 80+ COCO classes supported out of the box")
    print("• Custom class training support")
    print("• Advanced line crossing detection")
    print("• Confidence-based filtering")
    print("• FPS monitoring and optimization")
    
    print("\n📊 Supported Items (COCO classes):")
    print("• Fruits: apple, orange, banana")
    print("• Vegetables: carrot, broccoli")
    print("• Custom: tomato, seeds (requires training)")
    
    try:
        # Check if YOLO model is available
        test_model = YOLO("yolo11n.pt")
        print("\n✅ YOLOv11 model loaded successfully!")
        
        # Start the Flask application
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please install ultralytics: pip install ultralytics")
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down system...")
        inventory_manager.stop_camera_monitoring()
        print("✅ System stopped successfully!")