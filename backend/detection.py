# detction.py
import cv2
import torch
import numpy as np
import uuid
import os
import time
from datetime import datetime, timedelta
import shutil
import requests
from pathlib import Path
from difflib import SequenceMatcher
from dotenv import load_dotenv
from openai import OpenAI
import base64
import io
from PIL import Image
import json
import re

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Global variables
MODELS_DIR = "models"
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")
VIOLATIONS_DIR = "violations"
stream_active = False
PLATE_COOLDOWN = 60  # Reduce cooldown to 1 minute for testing
MIN_PLATE_CONFIDENCE = 0.2  # Lower confidence threshold
PLATE_SIMILARITY_THRESHOLD = 0.7  # Lower similarity threshold

# Create necessary directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIOLATIONS_DIR, exist_ok=True)

# Initialize models
yolo_model = None

# Add new tracking set for unique violations
processed_violations = set()

# Add this as a class variable in the Detection class
last_detection = {}  # Store last detection time for each license plate

class VehicleTracker:
    def __init__(self):
        self.tracked_vehicles = {}
    
    def is_similar_plate(self, plate1, plate2):
        """Check if two plate numbers are similar (handles OCR variations)"""
        if not plate1 or not plate2:
            return False
        # Remove spaces and convert to uppercase
        plate1 = ''.join(plate1.upper().split())
        plate2 = ''.join(plate2.upper().split())
        # Calculate similarity ratio
        similarity = SequenceMatcher(None, plate1, plate2).ratio()
        print(f"Comparing plates: {plate1} vs {plate2}, Similarity: {similarity}")
        return similarity > PLATE_SIMILARITY_THRESHOLD
    
    def can_record_vehicle(self, plate_number):
        """Check if a vehicle can be recorded based on cooldown and similarity"""
        current_time = time.time()
        print(f"Checking if can record plate: {plate_number}")
        
        # Check against all tracked vehicles
        for tracked_plate, data in self.tracked_vehicles.items():
            if self.is_similar_plate(plate_number, tracked_plate):
                last_seen = data['last_seen']
                time_diff = current_time - last_seen
                print(f"Found similar plate: {tracked_plate}")
                print(f"Time since last detection: {time_diff} seconds")
                
                if time_diff < PLATE_COOLDOWN:
                    print(f"Skipping due to cooldown: {time_diff} seconds since last detection")
                    return False
                
                print(f"Cooldown expired, allowing new detection")
                self.tracked_vehicles[tracked_plate] = {
                    'last_seen': current_time,
                    'count': data['count'] + 1
                }
                return True
        
        print(f"New vehicle detected: {plate_number}")
        self.tracked_vehicles[plate_number] = {
            'last_seen': current_time,
            'count': 1
        }
        return True

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

# Create global instance of vehicle tracker
vehicle_tracker = VehicleTracker()

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

def analyze_image_with_openai(image):
    """
    Analyze image using OpenAI's API with gpt-4o-mini model with improved prompting
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    width, height = image.size
    
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this motorcycle rider image with high precision:\n"
                                           "1. Is the rider wearing a helmet? Look carefully at the head area.\n"
                                           "2. What is the license plate number? Focus on the number plate area.\n"
                                           "3. Provide precise relative coordinates (0.0 to 1.0) for:\n"
                                           "   - Head region: Focus tightly on the rider's head area only\n"
                                           "   - Vehicle region: Include rider and motorcycle, exclude background\n"
                                           "   - License plate: Tight box around just the plate numbers\n"
                                           "Return in JSON format: {\n"
                                           "  \"motorcycle_rider_without_helmet\": boolean,\n"
                                           "  \"license_plate_number\": string,\n"
                                           "  \"head_bbox\": {\"x1\": float, \"y1\": float, \"x2\": float, \"y2\": float},\n"
                                           "  \"vehicle_bbox\": {\"x1\": float, \"y1\": float, \"x2\": float, \"y2\": float},\n"
                                           "  \"plate_bbox\": {\"x1\": float, \"y1\": float, \"x2\": float, \"y2\": float}\n"
                                           "}\nNote: Ensure coordinates are precise and tight around the specified regions."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }]
        )
        
        content = response.choices[0].message.content
        print(f"OpenAI Response: {content}")
        
        content = content.replace('```json', '').replace('```', '').strip()
        
        try:
            json_result = json.loads(content)
            
            # Convert relative coordinates to absolute pixels
            for bbox_key in ['head_bbox', 'vehicle_bbox', 'plate_bbox']:
                if bbox_key in json_result and json_result[bbox_key]:
                    bbox = json_result[bbox_key]
                    json_result[bbox_key] = {
                        'x1': int(bbox['x1'] * width),
                        'y1': int(bbox['y1'] * height),
                        'x2': int(bbox['x2'] * width),
                        'y2': int(bbox['y2'] * height)
                    }
            
            # Get helmet status and other information
            wearing_helmet = not json_result.get("motorcycle_rider_without_helmet", True)
            plate = json_result.get("license_plate_number", "")
            
            if plate:
                plate = plate.replace(' ', '')
                if not re.match(r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$', plate):
                    print(f"Warning: Invalid license plate format: {plate}")
                    plate = None

            result = {
                "wearing_helmet": wearing_helmet,
                "license_plate": plate,
                "confidence": 0.9 if plate else 0.6,
                "head_bbox": json_result.get("head_bbox"),
                "vehicle_bbox": json_result.get("vehicle_bbox"),
                "plate_bbox": json_result.get("plate_bbox")
            }
            
            print(f"Parsed result with bounding boxes: {result}")
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return None
            
    except Exception as e:
        print(f"OpenAI API Error: {str(e)}")
        return None

def extract_plate_from_text(text):
    """Helper function to extract license plate from text response"""
    # Look for common patterns in the response
    import re
    plate_patterns = [
        r'plate.*?([A-Z]{2}[-\s]?\d{1,4}[-\s]?[A-Z]{1,2}[-\s]?\d{1,4})',
        r'license.*?([A-Z]{2}[-\s]?\d{1,4}[-\s]?[A-Z]{1,2}[-\s]?\d{1,4})',
        r'number.*?([A-Z]{2}[-\s]?\d{1,4}[-\s]?[A-Z]{1,2}[-\s]?\d{1,4})'
    ]
    
    for pattern in plate_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).replace(' ', '').replace('-', '')
    
    return None

def is_unique_violation(license_plate):
    """
    Check if this is a unique violation
    """
    if license_plate in processed_violations:
        return False
    processed_violations.add(license_plate)
    return True

def detect_violations(video_path):
    """
    Process video file for violations using OpenAI
    """
    from database import save_violation, check_existing_violation  # Import at top
    
    violations = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    frame_interval = 30  # Process every 30th frame
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        analysis = analyze_image_with_openai(frame)
        if analysis:
            has_violation = not analysis.get('wearing_helmet', True)
            license_plate = analysis.get('license_plate')
            
            if license_plate:
                license_plate = license_plate.replace(' ', '').replace('-', '')
                
                # Skip if license plate is invalid
                if license_plate in ['N/A', 'unknown', None]:
                    continue
                    
                # Only process if we have a violation and valid license plate
                if has_violation:
                    # Check if violation already exists in database
                    if check_existing_violation(license_plate):
                        print(f"Skipping duplicate violation for plate: {license_plate}")
                        continue
                    
                    # Generate violation ID and save image
                    violation_id = str(uuid.uuid4())
                    image_path = save_violation_image(frame, analysis, violation_id)
                    
                    violation = {
                        "id": violation_id,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "violation_type": "No Helmet",
                        "license_plate": license_plate,
                        "vehicle_type": "Motorcycle",
                        "confidence": analysis.get('confidence', 0.9),
                        "image_path": image_path
                    }
                    
                    try:
                        save_violation(violation)
                        violations.append(violation)
                        print(f"New violation saved: {license_plate}")
                    except Exception as e:
                        print(f"Error saving violation: {e}")
    
    cap.release()
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
    if processed_vehicles is None:
        processed_vehicles = set()
    if potential_violations is None:
        potential_violations = {}
        
    frame_violations = []
    motorcycles = [obj for obj in detected_objects if obj["class"] == "motorcycle"]
    persons = [obj for obj in detected_objects if obj["class"] == "person"]
    
    print(f"Found {len(motorcycles)} motorcycles and {len(persons)} persons")
    
    for motorcycle in motorcycles:
        m_x1, m_y1, m_x2, m_y2 = motorcycle["bbox"]
        
        for person in persons:
            p_x1, p_y1, p_x2, p_y2 = person["bbox"]
            
            if is_rider(motorcycle["bbox"], person["bbox"]):
                print("Found rider on motorcycle")
                wearing_helmet = check_for_helmet(frame, person["bbox"])
                
                if not wearing_helmet:
                    print("Rider not wearing helmet")
                    try:
                        license_plate, plate_confidence = ocr_system.detect_license_plate(frame, motorcycle["bbox"])
                        print(f"Detected plate: {license_plate} with confidence: {plate_confidence}")
                        
                        if plate_confidence < MIN_PLATE_CONFIDENCE:
                            print(f"Skipping due to low confidence: {plate_confidence}")
                            continue
                        
                        if vehicle_tracker.can_record_vehicle(license_plate):
                            print(f"Recording violation for plate: {license_plate}")
                            violation_id = str(uuid.uuid4())
                            image_path = save_violation_image(frame, motorcycle["bbox"], person["bbox"], violation_id, license_plate)
                            
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
                            print(f"Violation recorded successfully")
                        
                    except Exception as e:
                        print(f"Error in violation processing: {str(e)}")
    
    print(f"Found {len(frame_violations)} violations in this frame")
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

def draw_bounding_boxes(frame, analysis_result):
    """
    Draw enhanced bounding boxes with improved visual presentation
    """
    img = frame.copy()
    height, width = img.shape[:2]
    
    def get_safe_bbox(bbox_data, default_ratio, box_type):
        """Enhanced helper function for more accurate bbox positioning"""
        if bbox_data and all(k in bbox_data for k in ['x1', 'y1', 'x2', 'y2']):
            # Ensure coordinates are within image bounds
            x1 = max(0, min(width-1, bbox_data['x1']))
            y1 = max(0, min(height-1, bbox_data['y1']))
            x2 = max(0, min(width-1, bbox_data['x2']))
            y2 = max(0, min(height-1, bbox_data['y2']))
            
            # Adjust box sizes based on type
            if box_type == 'head':
                # Ensure head box isn't too large
                box_width = x2 - x1
                box_height = y2 - y1
                if box_width > width * 0.3:  # If box is too wide
                    center_x = (x1 + x2) / 2
                    x1 = center_x - (width * 0.15)
                    x2 = center_x + (width * 0.15)
                if box_height > height * 0.3:  # If box is too tall
                    center_y = (y1 + y2) / 2
                    y1 = center_y - (height * 0.15)
                    y2 = center_y + (height * 0.15)
            elif box_type == 'plate':
                # Ensure plate box isn't too small
                if x2 - x1 < 60:  # Minimum width for readable plate
                    center_x = (x1 + x2) / 2
                    x1 = center_x - 30
                    x2 = center_x + 30
            
            return (int(x1), int(y1), int(x2), int(y2))
        else:
            # Improved fallback positions
            if box_type == 'vehicle':
                # Vehicle should cover most of the frame but not all
                y1 = int(height * 0.2)  # Start higher to include rider
                y2 = int(height * 0.95)  # Leave small margin at bottom
                x1 = int(width * 0.1)
                x2 = int(width * 0.9)
            elif box_type == 'head':
                # Head should be in upper-middle portion, smaller box
                y1 = int(height * 0.15)
                y2 = int(height * 0.35)
                x1 = int(width * 0.35)
                x2 = int(width * 0.65)
            else:  # License plate
                # Plate should be in middle-lower portion, smaller box
                y1 = int(height * 0.45)
                y2 = int(height * 0.55)
                x1 = int(width * 0.4)
                x2 = int(width * 0.6)
            
            return (x1, y1, x2, y2)
    
    # Draw semi-transparent overlay for better visibility
    overlay = img.copy()
    
    # Draw vehicle bounding box (red)
    vehicle_bbox = get_safe_bbox(analysis_result.get('vehicle_bbox'), None, 'vehicle')
    cv2.rectangle(overlay, (vehicle_bbox[0], vehicle_bbox[1]), 
                 (vehicle_bbox[2], vehicle_bbox[3]), (0, 0, 255), 2)
    
    # Draw head region (blue) with "No Helmet" text
    head_bbox = get_safe_bbox(analysis_result.get('head_bbox'), None, 'head')
    cv2.rectangle(overlay, (head_bbox[0], head_bbox[1]), 
                 (head_bbox[2], head_bbox[3]), (255, 0, 0), 2)
    
    # Add "No Helmet" text with background
    text = "No Helmet"
    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    text_x = head_bbox[0]
    text_y = max(text_size[1] + 10, head_bbox[1] - 10)
    
    # Draw text background
    cv2.rectangle(overlay, 
                 (text_x - 5, text_y - text_size[1] - 5),
                 (text_x + text_size[0] + 5, text_y + 5),
                 (255, 0, 0), -1)
    # Draw text
    cv2.putText(overlay, text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw license plate bounding box (green)
    plate_bbox = get_safe_bbox(analysis_result.get('plate_bbox'), None, 'plate')
    cv2.rectangle(overlay, (plate_bbox[0], plate_bbox[1]), 
                 (plate_bbox[2], plate_bbox[3]), (0, 255, 0), 2)
    
    if analysis_result.get('license_plate'):
        plate_text = f"Plate: {analysis_result['license_plate']}"
        text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = max(0, min(plate_bbox[0], width - text_size[0]))
        text_y = min(height - 10, plate_bbox[3] + 25)
        
        # Draw text background
        cv2.rectangle(overlay,
                     (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5),
                     (0, 255, 0), -1)
        # Draw text
        cv2.putText(overlay, plate_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add timestamp with background
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_size = cv2.getTextSize(f"Time: {timestamp}", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    
    # Draw timestamp background
    cv2.rectangle(overlay,
                 (5, height - time_size[1] - 15),
                 (time_size[0] + 15, height - 5),
                 (0, 0, 0), -1)
    # Draw timestamp
    cv2.putText(overlay, f"Time: {timestamp}", (10, height-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Blend the overlay with the original image
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    return img

def save_violation_image(frame, analysis_result, violation_id, license_plate=None):
    """
    Save violation image with bounding box annotations
    """
    # Add bounding boxes to the image
    annotated_img = draw_bounding_boxes(frame, analysis_result)
    
    # Save the annotated image
    image_path = os.path.join(VIOLATIONS_DIR, f"violation_{violation_id}.jpg")
    cv2.imwrite(image_path, annotated_img)
    
    return image_path

def process_live_stream():
    """
    Process live stream using OpenAI
    """
    global stream_active
    
    stream_active = True
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        yield b'--frame\r\nContent-Type: text/plain\r\n\r\nError: Could not access camera\r\n'
        return
    
    frame_skip = 30  # Process every 30th frame
    frame_count = 0
    
    while stream_active:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every nth frame with OpenAI
        if frame_count % frame_skip == 0:
            analysis = analyze_image_with_openai(frame)
            if analysis:
                # Add bounding boxes to the frame
                frame = draw_bounding_boxes(frame, analysis)
        
        # Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

def stop_live_stream():
    """
    Stop the live stream
    """
    global stream_active
    stream_active = False

class Detection:
    def __init__(self):
        self.last_detection = {}
    
    def check_and_save_violation(self, frame_result, frame):
        """
        Check if enough time has passed since last violation for this license plate
        """
        from database import check_existing_violation  # Import at top
        
        license_plate = frame_result.get('license_plate')
        if not license_plate:
            return None
            
        # Check if violation exists in database
        if check_existing_violation(license_plate):
            print(f"Skipping duplicate detection for {license_plate}")
            return None
            
        # Return frame result for further processing
        return frame_result
    
    def process_frame(self, frame):
        """
        Process a single frame for violations
        """
        try:
            frame_result = analyze_image_with_openai(frame)
            print(f"Frame analysis result: {frame_result}")
            
            if frame_result and not frame_result.get('wearing_helmet'):
                # Check for duplicate violations
                violation_data = self.check_and_save_violation(frame_result, frame)
                if violation_data:
                    # Generate violation ID
                    violation_id = str(uuid.uuid4())
                    
                    # Save annotated image
                    image_path = save_violation_image(frame, frame_result, violation_id, frame_result.get('license_plate'))
                    
                    # Create violation record
                    violation = {
                        "id": violation_id,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "violation_type": "No Helmet",
                        "license_plate": frame_result.get('license_plate'),
                        "vehicle_type": "Motorcycle",
                        "confidence": frame_result.get('confidence', 0.9),
                        "image_path": image_path
                    }
                    
                    return violation
            
            return None
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return None

# Create an instance of the Detection class
detector = Detection()

# Initialize models when module is imported
init_models()