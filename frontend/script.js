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
    startDetectionBtn.addEventListener('click', startDetection);
    closeModal.addEventListener('click', () => modal.style.display = 'none');
    closeModalBtn.addEventListener('click', () => modal.style.display = 'none');
    downloadChallanBtn.addEventListener('click', downloadChallan);

    // Hide the upload button since we've integrated upload + detection
    if (uploadBtn) {
        uploadBtn.style.display = 'none';
    }

    // File handling functions
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            processSelectedFile(file);
            uploadVideo(); // Automatically trigger upload when a file is selected
        }
    }

    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.add('active');
    }

    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('active');
    }

    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('active');
        
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('video/')) {
            processSelectedFile(file);
            uploadVideo(); // Automatically trigger upload when a file is dropped
        } else {
            showToast('Please upload a valid video file');
        }
    }

    function processSelectedFile(file) {
        selectedFile = file;
        const videoUrl = URL.createObjectURL(file);
        previewPlayer.src = videoUrl;
        videoPreview.style.display = 'block';
        if (uploadBtn) {
            uploadBtn.disabled = false;
        }

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
    }

    // API interaction functions
    async function uploadVideo() {
        if (!selectedFile) {
            showToast('Please select a video file first');
            return;
        }

        try {
            progressContainer.style.display = 'block';
            if (uploadBtn) {
                uploadBtn.disabled = true;
            }

            const formData = new FormData();
            formData.append('file', selectedFile);

            const xhr = new XMLHttpRequest();
            xhr.open('POST', UPLOAD_VIDEO_URL, true);

            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable) {
                    const percentComplete = Math.round((e.loaded / e.total) * 100);
                    progressBar.style.width = `${percentComplete}%`;
                    progressText.textContent = `${percentComplete}%`;
                }
            };

            xhr.onload = function() {
                if (xhr.status >= 200 && xhr.status < 300) {
                    const response = JSON.parse(xhr.responseText);
                    // Use path instead of filename for the video_path
                    uploadedVideoPath = response.path;
                    showToast('Video uploaded successfully');
                    startDetectionBtn.disabled = false;
                } else {
                    showToast('Upload failed: ' + xhr.statusText);
                    if (uploadBtn) {
                        uploadBtn.disabled = false;
                    }
                }
            };

            xhr.onerror = function() {
                showToast('Upload failed. Please try again');
                if (uploadBtn) {
                    uploadBtn.disabled = false;
                }
            };

            xhr.send(formData);
        } catch (error) {
            showToast('Error uploading video: ' + error.message);
            if (uploadBtn) {
                uploadBtn.disabled = false;
            }
        }
    }

    async function startDetection() {
        if (!uploadedVideoPath) {
            showToast('Please upload a video first');
            return;
        }

        try {
            startDetectionBtn.disabled = true;
            detectionLoader.style.display = 'flex';
            violationsContainer.innerHTML = '';

            // Create formData with the exact field name expected by the backend
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
            
            // Show toast indicating detection has started
            showToast('Detection started. This may take some time...');
            
            // Check if there are any violations
            if (data.violations_count > 0) {
                // Start polling for violations only if there are violations
                startPollingForViolations();
            } else {
                // No violations found, show appropriate message
                showToast('Detection complete. No new violations found.');
                detectionLoader.style.display = 'none';
                startDetectionBtn.disabled = false;
            }
        } catch (error) {
            showToast('Error starting detection: ' + error.message);
            startDetectionBtn.disabled = false;
            detectionLoader.style.display = 'none';
        }
    }

    function startPollingForViolations() {
        // Initial count of violations before polling
        const initialViolationCount = currentViolations.length;
        let pollCount = 0;
        const maxPolls = 60; // Stop polling after 5 minutes (5 seconds interval * 60)
        
        // Show loading indicator
        detectionLoader.style.display = 'flex';
        
        if (pollingInterval) {
            clearInterval(pollingInterval);
        }
        
        pollingInterval = setInterval(async () => {
            pollCount++;
            
            try {
                const newViolations = await fetchViolations();
                
                // Check if we found new violations or reached max polling attempts
                if (newViolations.length > initialViolationCount || pollCount >= maxPolls) {
                    // Stop polling
                    clearInterval(pollingInterval);
                    detectionLoader.style.display = 'none';
                    startDetectionBtn.disabled = false;
                    
                    // Show appropriate message
                    if (newViolations.length > initialViolationCount) {
                        showToast(`Detection complete! Found ${newViolations.length - initialViolationCount} new violations.`);
                    } else {
                        showToast('Detection complete. No new violations found.');
                    }
                } else if (pollCount % 4 === 0) {
                    // Every 20 seconds (4 polls * 5 seconds), update the user
                    showToast('Still processing video... Please wait.');
                }
            } catch (error) {
                console.error('Error polling for violations:', error);
                showToast('Error checking for violations. Still trying...');
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
            
            const response = await fetch(`${GENERATE_CHALLAN_URL}/${violationId}`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/pdf'
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            
            // Automatically download the challan PDF
            const link = document.createElement('a');
            link.href = url;
            link.download = `Challan_${violationId}.pdf`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
            
            showToast('Challan downloaded successfully');
        } catch (error) {
            showToast('Error generating challan: ' + error.message);
            throw error;
        }
    }

    function downloadChallan() {
        // This would typically call an API endpoint to download the PDF
        // For now, we'll just show a toast notification
        showToast('Challan PDF downloaded');
        modal.style.display = 'none';
    }

    // UI functions
    function displayViolations(violations, previousCount) {
        if (!violations.length) {
            violationsContainer.innerHTML = '<p class="no-violations">No violations detected yet.</p>';
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
        });
    }

    function showToast(message) {
        toast.textContent = message;
        toast.classList.add('show');
        
        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    }
});