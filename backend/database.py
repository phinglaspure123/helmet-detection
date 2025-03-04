import csv
import os
import json
from datetime import datetime
import numpy as np  # Import numpy for random email generation

# Define database file
DATABASE_FILE = "violations.csv"

# Define CSV headers
CSV_HEADERS = [
    "id", 
    "timestamp", 
    "violation_type", 
    "license_plate", 
    "vehicle_type", 
    "confidence", 
    "image_path", 
    "email"
]

def initialize_database():
    """
    Initialize the CSV database if it doesn't exist
    """
    if not os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(CSV_HEADERS)
        print(f"Created database file: {DATABASE_FILE}")

def save_violation(violation_data):
    """
    Save a violation record to the CSV database
    """
    initialize_database()
    
    # Generate a random email if not provided
    if "email" not in violation_data or not violation_data["email"]:
        violation_data["email"] = generate_random_email()
    
    # Prepare row data in correct order
    row_data = [violation_data.get(header, "") for header in CSV_HEADERS]
    
    # Append to CSV
    with open(DATABASE_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_data)
    
    print(f"Saved violation: {violation_data['id']}")
    return violation_data

def generate_random_email():
    """
    Generate a random email address for the violation database
    In a real app, this would be retrieved from a government database
    """
    # Predefined list of emails
    predefined_emails = ["vaibhaviingole24@gmail.com", "sharayugulhane1@gmail.com"]
    
    # Randomly select an email from the predefined list
    email = np.random.choice(predefined_emails)
    
    return email

def get_all_violations():
    """
    Get all violations from the database
    """
    initialize_database()
    
    violations = []
    try:
        with open(DATABASE_FILE, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                violations.append(dict(row))
    except Exception as e:
        print(f"Error reading database: {e}")
    
    return violations

def get_violation_by_id(violation_id):
    """
    Get a specific violation by ID
    """
    violations = get_all_violations()
    for violation in violations:
        if violation["id"] == violation_id:
            return violation
    return None

def get_violations_by_email(email):
    """
    Get all violations associated with an email
    """
    violations = get_all_violations()
    return [v for v in violations if v["email"] == email]

def get_violations_by_license_plate(license_plate):
    """
    Get all violations for a specific license plate
    """
    violations = get_all_violations()
    return [v for v in violations if v["license_plate"] == license_plate]

def get_recent_violations(limit=10):
    """
    Get the most recent violations
    """
    violations = get_all_violations()
    
    # Sort by timestamp (most recent first)
    violations.sort(key=lambda x: datetime.strptime(x["timestamp"], "%Y-%m-%d %H:%M:%S"), reverse=True)
    
    # Return limited number of violations
    return violations[:limit]

def delete_violation(violation_id):
    """
    Delete a violation from the database
    """
    violations = get_all_violations()
    updated_violations = [v for v in violations if v["id"] != violation_id]
    
    # Write the updated list back to CSV
    with open(DATABASE_FILE, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(updated_violations)
    
    # Check if an image file needs to be deleted
    for violation in violations:
        if violation["id"] == violation_id and violation["image_path"]:
            try:
                if os.path.exists(violation["image_path"]):
                    os.remove(violation["image_path"])
            except Exception as e:
                print(f"Error deleting image file: {e}")
    
    return len(violations) - len(updated_violations)  # Return count of deleted items

def get_statistics():
    """
    Get statistics about violations
    """
    violations = get_all_violations()
    
    # Count by violation type
    violation_types = {}
    for v in violations:
        v_type = v["violation_type"]
        if v_type in violation_types:
            violation_types[v_type] += 1
        else:
            violation_types[v_type] = 1
    
    # Count by vehicle type
    vehicle_types = {}
    for v in violations:
        v_type = v["vehicle_type"]
        if v_type in vehicle_types:
            vehicle_types[v_type] += 1
        else:
            vehicle_types[v_type] = 1
    
    # Count by date
    dates = {}
    for v in violations:
        try:
            date = datetime.strptime(v["timestamp"], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
            if date in dates:
                dates[date] += 1
            else:
                dates[date] = 1
        except:
            pass
    
    return {
        "total_violations": len(violations),
        "by_violation_type": violation_types,
        "by_vehicle_type": vehicle_types,
        "by_date": dates
    }