document.addEventListener('DOMContentLoaded', () => {
    // API URLs
    const API_BASE_URL = 'http://127.0.0.1:8000';
    const UPLOAD_VIDEO_URL = `${API_BASE_URL}/upload-video`;
    const DETECT_URL = `${API_BASE_URL}/detect`;
    const VIOLATIONS_URL = `${API_BASE_URL}/violations`;
    const GENERATE_CHALLAN_URL = `${API_BASE_URL}/generate-challan`;

    // DOM Elements
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('video-upload');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const videoPreview = document.getElementById('video-preview');
    const previewPlayer = document.getElementById('preview-player');
    const uploadBtn = document.getElementById('upload-btn');
    const startDetectionBtn = document.getElementById('start-detection-btn');
    const detectionLoader = document.getElementById('detection-loader');
    const violationsContainer = document.getElementById('violations-container');
    const violationCardTemplate = document.getElementById('violation-card-template');
    const toast = document.getElementById('toast');
    const modal = document.getElementById('modal');
    const closeModal = document.querySelector('.close-modal');
    const closeModalBtn = document.getElementById('close-modal-btn');
    const downloadChallanBtn = document.getElementById('download-challan-btn');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');

    // State variables
    let selectedFile = null;
    let uploadedVideoPath = null;
    let pollingInterval = null;
    let currentViolations = [];

    // Event Listeners
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    uploadBtn.addEventListener('click', uploadVideo);
    startDetectionBtn.addEventListener('click', startDetection);
    closeModal.addEventListener('click', () => modal.style.display = 'none');
    closeModalBtn.addEventListener('click', () => modal.style.display = 'none');
    downloadChallanBtn.addEventListener('click', downloadChallan);

    // Initialize the application
    fetchViolations();

    // File handling functions
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            console.log('File selected:', file.name);
            processSelectedFile(file);
        }
    }

    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.add('active');
        console.log('Drag over event');
    }

    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('active');
        console.log('Drag leave event');
    }

    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('active');
        
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('video/')) {
            console.log('File dropped:', file.name);
            processSelectedFile(file);
        } else {
            showToast('Please upload a valid video file');
            console.log('Invalid file type dropped');
        }
    }

    function processSelectedFile(file) {
        selectedFile = file;
        const videoUrl = URL.createObjectURL(file);
        previewPlayer.src = videoUrl;
        videoPreview.style.display = 'block';
        uploadBtn.disabled = false;

        // Update upload area to show selected file name
        const fileName = document.createElement('p');
        fileName.textContent = `Selected file: ${file.name}`;
        fileName.classList.add('selected-file-name');
        
        // Remove previous file name if exists
        const prevFileName = uploadArea.querySelector('.selected-file-name');
        if (prevFileName) {
            uploadArea.removeChild(prevFileName);
        }
        
        uploadArea.appendChild(fileName);
        console.log('File processed:', file.name);
    }

    // API interaction functions
    async function uploadVideo() {
        if (!selectedFile) {
            showToast('Please select a video file first');
            console.log('Upload attempt without file selection');
            return;
        }

        try {
            progressContainer.style.display = 'block';
            uploadBtn.disabled = true;
            console.log('Uploading video:', selectedFile.name);

            const formData = new FormData();
            formData.append('file', selectedFile);

            const xhr = new XMLHttpRequest();
            xhr.open('POST', UPLOAD_VIDEO_URL, true);

            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) {
                    const percentComplete = Math.round((e.loaded / e.total) * 100);
                    progressBar.style.width = `${percentComplete}%`;
                    progressText.textContent = `${percentComplete}%`;
                    console.log('Upload progress:', percentComplete + '%');
                }
            };

            xhr.onload = function() {
                if (xhr.status >= 200 && xhr.status < 300) {
                    const response = JSON.parse(xhr.responseText);
                    uploadedVideoPath = response.path; // Use path instead of filename
                    showToast('Video uploaded successfully');
                    startDetectionBtn.disabled = false;
                    console.log('Video uploaded successfully:', uploadedVideoPath);
                } else {
                    showToast('Upload failed: ' + xhr.statusText);
                    uploadBtn.disabled = false;
                    console.error('Upload failed:', xhr.statusText);
                }
            };

            xhr.onerror = function() {
                showToast('Upload failed. Please try again');
                uploadBtn.disabled = false;
                console.error('Upload error occurred');
            };

            xhr.send(formData);
        } catch (error) {
            showToast('Error uploading video: ' + error.message);
            uploadBtn.disabled = false;
            console.error('Error uploading video:', error.message);
        }
    }

    async function startDetection() {
        if (!uploadedVideoPath) {
            showToast('Please upload a video first');
            console.log('Detection attempt without video upload');
            return;
        }

        try {
            startDetectionBtn.disabled = true;
            detectionLoader.style.display = 'flex';
            violationsContainer.innerHTML = '';
            console.log('Starting detection for video:', uploadedVideoPath);

            const formData = new FormData();
            formData.append('video_path', uploadedVideoPath);

            const response = await fetch(DETECT_URL, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            if (data.status === 'processing') {
                showToast('Detection started. Polling for results...');
                console.log('Detection started, polling for results...');
                startPollingForViolations();
            } else {
                showToast('Detection failed to start');
                startDetectionBtn.disabled = false;
                detectionLoader.style.display = 'none';
                console.error('Detection failed to start');
            }
        } catch (error) {
            showToast('Error starting detection: ' + error.message);
            startDetectionBtn.disabled = false;
            detectionLoader.style.display = 'none';
            console.error('Error starting detection:', error.message);
        }
    }

    function startPollingForViolations() {
        let pollCount = 0;
        const maxPolls = 60; // Stop polling after 5 minutes (5 seconds interval * 60)
        
        if (pollingInterval) {
            clearInterval(pollingInterval);
        }
        
        pollingInterval = setInterval(async () => {
            pollCount++;
            console.log('Polling for violations, attempt:', pollCount);
            
            try {
                const newViolations = await fetchViolations();
                
                if (newViolations.length > currentViolations.length || pollCount >= maxPolls) {
                    clearInterval(pollingInterval);
                    detectionLoader.style.display = 'none';
                    startDetectionBtn.disabled = false;
                    
                    if (newViolations.length > currentViolations.length) {
                        showToast('New violations detected!');
                        console.log('New violations detected:', newViolations.length - currentViolations.length);
                    } else {
                        showToast('Detection complete. No new violations found.');
                        console.log('Detection complete. No new violations found.');
                    }
                }
            } catch (error) {
                console.error('Error polling for violations:', error);
            }
        }, 5000); // Poll every 5 seconds
    }

    async function fetchViolations() {
        try {
            const response = await fetch(VIOLATIONS_URL);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            const violations = data.violations || [];
            
            // Store current violations for comparison
            const previousCount = currentViolations.length;
            currentViolations = violations;
            
            // Display violations
            displayViolations(violations, previousCount);
            console.log('Fetched violations:', violations.length);
            
            return violations;
        } catch (error) {
            console.error('Error fetching violations:', error);
            showToast('Error fetching violations: ' + error.message);
            return [];
        }
    }

    async function generateChallan(violationId) {
        try {
            showToast('Generating challan...');
            console.log('Generating challan for violation ID:', violationId);
            
            const response = await fetch(`${GENERATE_CHALLAN_URL}/${violationId}`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const challanData = await response.json();
            
            // Display challan in modal
            modalTitle.textContent = 'Challan Generated';
            modalBody.innerHTML = `
                <div class="challan-details">
                    <p><strong>Challan ID:</strong> ${challanData.id || violationId}</p>
                    <p><strong>Issued Date:</strong> ${challanData.issued_date || new Date().toLocaleString()}</p>
                    <p><strong>Violation Type:</strong> ${challanData.violation_type || 'Traffic Violation'}</p>
                    <p><strong>License Plate:</strong> ${challanData.license_plate || 'N/A'}</p>
                    <p><strong>Vehicle Type:</strong> ${challanData.vehicle_type || 'N/A'}</p>
                    <p><strong>Fine Amount:</strong> ₹${challanData.fine_amount || '500'}</p>
                    <p><strong>Payment Due:</strong> ${challanData.payment_due || '15 days'}</p>
                </div>
                <div class="challan-image">
                    <img src="${API_BASE_URL}/${challanData.image_path || ''}" alt="Violation Image">
                </div>
            `;
            
            modal.style.display = 'flex';
            console.log('Challan generated:', challanData);
            
            return challanData;
        } catch (error) {
            showToast('Error generating challan: ' + error.message);
            console.error('Error generating challan:', error.message);
            throw error;
        }
    }

    function downloadChallan() {
        // This would typically call an API endpoint to download the PDF
        // For now, we'll just show a toast notification
        showToast('Challan PDF downloaded');
        modal.style.display = 'none';
        console.log('Challan PDF downloaded');
    }

    // UI functions
    function displayViolations(violations, previousCount) {
        if (!violations.length) {
            violationsContainer.innerHTML = '<p class="no-violations">No violations detected yet.</p>';
            console.log('No violations detected yet.');
            return;
        }
        
        // If this is the initial load, clear the container
        if (previousCount === 0) {
            violationsContainer.innerHTML = '';
        }
        
        // Add new violations
        violations.slice(previousCount).forEach((violation, index) => {
            const violationCard = violationCardTemplate.content.cloneNode(true);
            
            // Set image source - ensure path is properly formatted
            const imgElement = violationCard.querySelector('.violation-image img');
            imgElement.src = `${API_BASE_URL}/${violation.image_path}`;
            imgElement.alt = `${violation.violation_type} Violation`;
            
            // Set violation details
            violationCard.querySelector('.violation-type').textContent = violation.violation_type;
            violationCard.querySelector('.violation-id').textContent = violation.id;
            violationCard.querySelector('.violation-timestamp').textContent = violation.timestamp;
            violationCard.querySelector('.violation-license').textContent = violation.license_plate;
            violationCard.querySelector('.violation-vehicle').textContent = violation.vehicle_type;
            
            // Format confidence as percentage
            const confidence = parseFloat(violation.confidence) * 100;
            violationCard.querySelector('.violation-confidence').textContent = `${confidence.toFixed(2)}%`;
            
            violationCard.querySelector('.violation-email').textContent = violation.email;
            
            // Set up challan generation button
            const challanBtn = violationCard.querySelector('.generate-challan-btn');
            challanBtn.dataset.id = violation.id;
            challanBtn.addEventListener('click', (e) => {
                const id = e.target.dataset.id;
                generateChallan(id);
            });
            
            // Add animation class
            const cardElement = violationCard.querySelector('.violation-card');
            cardElement.classList.add('fade-in');
            cardElement.style.animationDelay = `${index * 0.1}s`;
            
            // Add to container
            violationsContainer.appendChild(violationCard);
            console.log('Violation card added:', violation.id);
        });
    }

    function showToast(message) {
        toast.textContent = message;
        toast.classList.add('show');
        console.log('Toast message:', message);
        
        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    }
});