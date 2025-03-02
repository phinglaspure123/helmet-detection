# detction.py
import cv2
import torch
import numpy as np
import uuid
import os
import time
from datetime import datetime
import shutil
import requests
from pathlib import Path

# Global variables
MODELS_DIR = "models"
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")
VIOLATIONS_DIR = "violations"

# Create necessary directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIOLATIONS_DIR, exist_ok=True)

# Initialize models
yolo_model = None

class UniversalLicensePlateOCR:
    def __init__(self):
        self.reader = None
        self.initialize_ocr()

    def initialize_ocr(self):
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), download_enabled=True, detector=True)
            print("OCR model loaded and ready")
        except ImportError:
            print("ERROR: EasyOCR is not installed! Please install it with: pip install easyocr")
            raise RuntimeError("OCR is required but not available")

    def detect_license_plate(self, frame, vehicle_bbox):
        x1, y1, x2, y2 = vehicle_bbox
        vehicle_region = frame[y1:y2, x1:x2]

        if vehicle_region.size == 0 or vehicle_region.shape[0] < 10 or vehicle_region.shape[1] < 10:
            raise ValueError("Vehicle region too small for license plate detection")

        gray = cv2.cvtColor(vehicle_region, cv2.COLOR_BGR2GRAY)
        ocr_results = self.reader.readtext(
            gray,
            detail=1,
            paragraph=False,
            height_ths=0.8,
            width_ths=0.8,
            batch_size=4,
            decoder='greedy',
            beamWidth=5
        )

        plate_candidates = []
        for text_bbox, text, confidence in ocr_results:
            text = text.strip()
            if len(text) >= 4 and confidence > 0.2:
                plate_candidates.append((text, confidence))

        if plate_candidates:
            plate_candidates.sort(key=lambda x: x[1], reverse=True)
            best_plate = plate_candidates[0]
            return best_plate[0], best_plate[1]

        raise ValueError("No license plate detected")

ocr_system = UniversalLicensePlateOCR()

def download_yolo_model():
    """Download YOLOv8 model if not already present"""
    if os.path.exists(YOLO_MODEL_PATH):
        print(f"YOLO model already exists at {YOLO_MODEL_PATH}")
        return True
    
    print(f"Downloading YOLOv8 model to {YOLO_MODEL_PATH}...")
    model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        with open(YOLO_MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("YOLO model download complete")
        return True
    except Exception as e:
        print(f"Error downloading YOLO model: {e}")
        return False

def init_models():
    """Initialize all models when server starts"""
    global yolo_model
    
    # Download and load YOLO model
    try:
        download_yolo_model()
        from ultralytics import YOLO
        print("Loading YOLO model...")
        yolo_model = YOLO(YOLO_MODEL_PATH)
        
        # Use GPU if available
        if torch.cuda.is_available():
            print("Using GPU for YOLO detection")
            yolo_model.to("cuda")
        else:
            print("GPU not available, using CPU for YOLO")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        yolo_model = None

def detect_violations(video_path):
    """
    Process a video file to detect traffic violations (no helmet)
    Returns a list of violation records
    """
    global yolo_model
    
    if yolo_model is None:
        print("YOLO model not initialized. Trying to load...")
        init_models()
        
    if yolo_model is None:
        print("Failed to load YOLO model. Cannot detect violations.")
        return []
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    violations = []
    processed_vehicles = set()  # Track vehicles we've already processed
    potential_violations = {}  # Track potential violations for confirmation
    
    # Process every frame in the video
    print(f"Video has {total_frames} frames, processing every frame")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
            
        print(f"Processing frame {frame_count}/{total_frames}")
        
        # Perform detection using YOLO
        results = yolo_model(frame, classes=[0, 3], conf=0.25)  # Lower confidence threshold to catch more
        
        # Process results
        detected_objects = process_results(results, frame)
        
        # Check for violations
        frame_violations = check_violations(detected_objects, frame, processed_vehicles, potential_violations)
        
        # Add any confirmed violations found in this frame
        violations.extend(frame_violations)
        
        # Update our set of processed vehicles
        for v in frame_violations:
            processed_vehicles.add(v["license_plate"])
        
    cap.release()
    
    print(f"Detected {len(violations)} violations")
    return violations

def process_results(results, frame):
    """
    Process YOLO detection results
    Returns a list of detected objects with coordinates and classes
    """
    detected_objects = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            
            # Class mapping - YOLO class 0 is person, class 3 is motorcycle
            class_name = "person" if class_id == 0 else "motorcycle"
            
            detected_objects.append({
                "class": class_name,
                "confidence": confidence,
                "bbox": (int(x1), int(y1), int(x2), int(y2))
            })
    
    return detected_objects

def check_violations(detected_objects, frame, processed_vehicles=None, potential_violations=None):
    """
    Check for traffic violations based on detected objects
    Returns a list of confirmed violations in this frame
    """
    if processed_vehicles is None:
        processed_vehicles = set()
    if potential_violations is None:
        potential_violations = {}
        
    frame_violations = []
    motorcycles = [obj for obj in detected_objects if obj["class"] == "motorcycle"]
    persons = [obj for obj in detected_objects if obj["class"] == "person"]
    
    # For each motorcycle, check if rider is wearing a helmet
    for motorcycle in motorcycles:
        m_x1, m_y1, m_x2, m_y2 = motorcycle["bbox"]
        
        # Find nearby persons (potential riders)
        for person in persons:
            p_x1, p_y1, p_x2, p_y2 = person["bbox"]
            
            # Check if person is near the motorcycle (likely the rider)
            if is_rider(motorcycle["bbox"], person["bbox"]):
                # Check if the person's head region has a helmet
                wearing_helmet = check_for_helmet(frame, person["bbox"])
                
                if not wearing_helmet:
                    try:
                        # Try to detect the license plate with our improved function
                        license_plate, plate_confidence = ocr_system.detect_license_plate(frame, motorcycle["bbox"])
                        
                        # Skip if we've already processed this vehicle
                        if license_plate in processed_vehicles:
                            continue
                        
                        # Check if this is a potential violation
                        if license_plate in potential_violations:
                            # Confirm the violation if it persists
                            violation_id = potential_violations.pop(license_plate)
                            image_path = save_violation_image(frame, motorcycle["bbox"], person["bbox"], violation_id, license_plate)
                            
                            # Create a violation record
                            violation = {
                                "id": violation_id,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "violation_type": "No Helmet",
                                "license_plate": license_plate,
                                "vehicle_type": "Motorcycle",
                                "confidence": motorcycle["confidence"],
                                "plate_confidence": plate_confidence,
                                "image_path": image_path
                            }
                            
                            frame_violations.append(violation)
                            print(f"Confirmed violation with license plate: {license_plate}")
                        else:
                            # Add to potential violations for confirmation in subsequent frames
                            potential_violations[license_plate] = str(uuid.uuid4())
                            print(f"Potential violation detected for license plate: {license_plate}")
                        
                    except Exception as e:
                        print(f"Error detecting license plate: {e}")
    
    return frame_violations

def is_rider(motorcycle_bbox, person_bbox):
    """
    Determine if a person is riding a motorcycle based on improved bounding box analysis
    """
    m_x1, m_y1, m_x2, m_y2 = motorcycle_bbox
    p_x1, p_y1, p_x2, p_y2 = person_bbox
    
    # Calculate motorcycle and person dimensions
    m_width = m_x2 - m_x1
    m_height = m_y2 - m_y1
    p_width = p_x2 - p_x1
    p_height = p_y2 - p_y1
    
    # Calculate the center points
    m_center_x = (m_x1 + m_x2) / 2
    m_center_y = (m_y1 + m_y2) / 2
    p_center_x = (p_x1 + p_x2) / 2
    p_center_y = (p_y1 + p_y2) / 2
    
    # Calculate horizontal and vertical distances between centers
    h_distance = abs(m_center_x - p_center_x)
    v_distance = abs(m_center_y - p_center_y)
    
    # Check horizontal overlap (centers must be reasonably close)
    horizontal_match = h_distance < (m_width * 0.7)
    
    # Check vertical position - person should be on top of motorcycle
    vertical_match = (p_y2 > m_y1) and (p_y1 < m_y2) and (p_center_y <= m_center_y)
    
    # Improved overlap calculation
    intersection_width = min(p_x2, m_x2) - max(p_x1, m_x1)
    intersection_height = min(p_y2, m_y2) - max(p_y1, m_y1)
    
    if intersection_width > 0 and intersection_height > 0:
        intersection_area = intersection_width * intersection_height
        person_area = p_width * p_height
        overlap_ratio = intersection_area / person_area
        significant_overlap = overlap_ratio > 0.2  # At least 20% of person overlaps with motorcycle
    else:
        significant_overlap = False
    
    # Final decision - person is likely a rider if there's both horizontal match and significant overlap
    return horizontal_match and (significant_overlap or vertical_match)

def check_for_helmet(frame, person_bbox):
    """
    Check if person is wearing a helmet by analyzing the upper portion of person bbox
    
    In a real application, you would use a dedicated helmet detection model,
    but this improved version tries to do better than random by analyzing colors and shapes
    """
    x1, y1, x2, y2 = person_bbox
    
    # Focus on the upper 1/3 of the person's bounding box (where the head would be)
    head_height = (y2 - y1) // 3
    head_region = frame[y1:y1+head_height, x1:x2]
    
    # If the head region is invalid or too small, default to no helmet
    if head_region.size == 0 or head_region.shape[0] < 5 or head_region.shape[1] < 5:
        return False
    
    try:
        # Simple color-based analysis for helmet detection
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
        
        # Define color ranges that are common for helmets
        # Safety helmets are often white, black, red, blue, yellow, orange
        helmet_color_ranges = [
            ((0, 0, 0), (180, 40, 40)),      # Black/dark
            ((0, 0, 200), (180, 30, 255)),   # White/light
            ((0, 100, 100), (10, 255, 255)),  # Red
            ((160, 100, 100), (180, 255, 255)), # Red (wrap-around)
            ((100, 100, 100), (140, 255, 255)), # Blue
            ((20, 100, 100), (40, 255, 255)),  # Yellow
            ((5, 100, 100), (20, 255, 255))    # Orange
        ]
        
        # Check for significant presence of helmet colors
        helmet_color_pixels = 0
        total_pixels = head_region.shape[0] * head_region.shape[1]
        
        for low, high in helmet_color_ranges:
            mask = cv2.inRange(hsv, np.array(low), np.array(high))
            helmet_color_pixels += cv2.countNonZero(mask)
        
        helmet_color_ratio = helmet_color_pixels / total_pixels
        
        # If a significant portion has helmet-like colors, likely a helmet
        return helmet_color_ratio > 0.6
    except Exception as e:
        print(f"Error in helmet detection: {e}")
        return False

def save_violation_image(frame, motorcycle_bbox, person_bbox, violation_id, license_plate=None):
    """
    Save an image of the violation with bounding boxes drawn
    """
    # Create a copy of the frame
    img = frame.copy()
    
    # Draw bounding boxes
    m_x1, m_y1, m_x2, m_y2 = motorcycle_bbox
    cv2.rectangle(img, (m_x1, m_y1), (m_x2, m_y2), (0, 0, 255), 2)  # Red for motorcycle
    
    p_x1, p_y1, p_x2, p_y2 = person_bbox
    cv2.rectangle(img, (p_x1, p_y1), (p_x2, p_y2), (255, 0, 0), 2)  # Blue for person
    
    # Add text labels
    cv2.putText(img, "NO HELMET", (p_x1, p_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Add license plate text if available
    if license_plate:
        cv2.putText(img, f"PLATE: {license_plate}", (m_x1, m_y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Save the image
    image_path = os.path.join(VIOLATIONS_DIR, f"violation_{violation_id}.jpg")
    cv2.imwrite(image_path, img)
    
    return image_path

def process_live_stream():
    """
    Process a live video stream from camera
    """
    global yolo_model
    
    if yolo_model is None:
        print("YOLO model not initialized. Trying to load...")
        init_models()
        
    if yolo_model is None:
        print("Failed to load YOLO model. Cannot process live stream.")
        return
    
    # Open webcam
    cap = cv2.VideoCapture(0)  # 0 for default camera
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    frame_count = 0
    processed_vehicles = set()  # Track vehicles we've already processed
    potential_violations = {}  # Track potential violations for confirmation
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Perform detection
        results = yolo_model(frame, classes=[0, 3], conf=0.5)  # Classes 0 (person), 3 (motorcycle)
        
        # Process results
        detected_objects = process_results(results, frame)
        
        # Check for violations
        frame_violations = check_violations(detected_objects, frame, processed_vehicles, potential_violations)
        
        # Save violations to database
        if frame_violations:
            import database
            for violation in frame_violations:
                database.save_violation(violation)
                processed_vehicles.add(violation["license_plate"])
        
        # Display the frame with detections (for debugging)
        # cv2.imshow("Live Detection", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        
        # Add a small delay to prevent CPU overload
        time.sleep(0.1)
    
    cap.release()
    cv2.destroyAllWindows()

# Initialize models when module is imported
init_models()