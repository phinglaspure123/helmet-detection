import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
import os
from datetime import datetime
import pdf

# Email configuration - update with your SMTP settings
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = "helmetdetection12@gmail.com"  # Replace with your email
SMTP_PASSWORD = "auzj iroi oidh islh"     # Replace with your app password (for Gmail)
SENDER_EMAIL = "helmetdetection12@gmail.com"   # Replace with your email

def send_violation_email(recipient_email, violations):
    """
    Send email notification about traffic violations
    """
    # Skip if no violations
    if not violations:
        print("No violations to report")
        return False
    
    # Create message
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = recipient_email
    msg['Subject'] = f"Traffic Violation Detection - {datetime.now().strftime('%Y-%m-%d')}"
    
    # Email body
    email_body = f"""
    <html>
    <body>
        <h2>Traffic Violation Report</h2>
        <p>We have detected the following traffic violations:</p>
        <table border="1" cellpadding="5">
            <tr>
                <th>Time</th>
                <th>Violation Type</th>
                <th>License Plate</th>
                <th>Vehicle Type</th>
            </tr>
    """
    
    # Add each violation to the email
    attachments = []
    for violation in violations:
        # Add row to table
        email_body += f"""
        <tr>
            <td>{violation['timestamp']}</td>
            <td>{violation['violation_type']}</td>
            <td>{violation['license_plate']}</td>
            <td>{violation['vehicle_type']}</td>
        </tr>
        """
        
        # Add image to attachments if it exists
        if 'image_path' in violation and os.path.exists(violation['image_path']):
            attachments.append(violation['image_path'])
    
    # Close table and body
    email_body += """
        </table>
        <p>Please see the attached images for details.</p>
        <p>A challan (fine) will be issued for these violations.</p>
        <p>This is an automated message, please do not reply.</p>
    </body>
    </html>
    """
    
    # Attach HTML body
    msg.attach(MIMEText(email_body, 'html'))
    
    # Attach violation images
    for i, attachment in enumerate(attachments):
        with open(attachment, 'rb') as f:
            img_data = f.read()
            img = MIMEImage(img_data)
            img.add_header('Content-Disposition', 'attachment', 
                           filename=f"violation_{i+1}.jpg")
            msg.attach(img)
    
    # Generate PDF challans for each violation and attach
    for i, violation in enumerate(violations):
        # Generate challan PDF
        pdf_path = pdf.generate_challan(violation)
        
        # Attach PDF
        if os.path.exists(pdf_path):
            with open(pdf_path, 'rb') as f:
                pdf_attachment = MIMEApplication(f.read(), _subtype="pdf")
                pdf_attachment.add_header('Content-Disposition', 'attachment', 
                                     filename=f"challan_{violation['id']}.pdf")
                msg.attach(pdf_attachment)
    
    try:
        # Connect to SMTP server
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        
        # Send email
        server.send_message(msg)
        server.quit()
        
        print(f"Email sent to {recipient_email} with {len(violations)} violations")
        return True
    
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

def send_test_email(recipient_email):
    """
    Send a test email to verify the configuration
    """
    # Create message
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = recipient_email
    msg['Subject'] = "Test Email - Helmet Detection System"
    
    # Email body
    body = """
    <html>
    <body>
        <h2>Test Email</h2>
        <p>This is a test email from the Helmet Detection System.</p>
        <p>If you received this email, the email configuration is working correctly.</p>
    </body>
    </html>
    """
    
    # Attach HTML body
    msg.attach(MIMEText(body, 'html'))
    
    try:
        # Connect to SMTP server
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        
        # Send email
        server.send_message(msg)
        server.quit()
        
        print(f"Test email sent to {recipient_email}")
        return True
    
    except Exception as e:
        print(f"Failed to send test email: {e}")
        return False