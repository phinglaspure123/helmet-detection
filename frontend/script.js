console.log('Script loaded');

document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded');
    // Constants for API endpoints
    const API_BASE_URL = 'http://127.0.0.1:8000';
    const UPLOAD_URL = `${API_BASE_URL}/upload-video`;
    const START_DETECTION_URL = `${API_BASE_URL}/detect`;
    const GET_VIOLATIONS_URL = `${API_BASE_URL}/violations`;
    const GENERATE_CHALLAN_URL = `${API_BASE_URL}/generate-challan`;
    const GET_CHALLAN_URL = `${API_BASE_URL}/get-challan`;
    
    // Comment out live stream endpoint URLs
    // const LIVE_STREAM_START_URL = `${API_BASE_URL}/live-stream/start`;
    // const LIVE_STREAM_STOP_URL = `${API_BASE_URL}/live-stream/stop`;

    // DOM elements
    const videoFileInput = document.getElementById('video-upload');
    const uploadArea = document.getElementById('upload-area');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const videoPreview = document.getElementById('video-preview');
    const previewPlayer = document.getElementById('preview-player');
    const startDetectionBtn = document.getElementById('start-detection-btn');
    const detectionLoader = document.getElementById('detection-loader');
    const violationsContainer = document.getElementById('violations-container');
    const toast = document.getElementById('toast');
    const modal = document.getElementById('modal');
    const closeModal = document.querySelector('.close-modal');
    const closeModalBtn = document.getElementById('close-modal-btn');
    const downloadChallanBtn = document.getElementById('download-challan-btn');
    const violationCardTemplate = document.getElementById('violation-card-template');

    // Comment out live stream elements
    // const startStreamBtn = document.getElementById('start-stream-btn');
    // const stopStreamBtn = document.getElementById('stop-stream-btn');
    // const liveStreamPreview = document.getElementById('live-stream-preview');
    // const streamLoader = document.getElementById('stream-loader');
    // const streamPlaceholder = document.querySelector('.stream-placeholder');

    // State management
    let uploadedFile = null;
    let uploadedFileName = '';
    let detectionInProgress = false;
    let pollingInterval = null;
    let currentViolationsCount = 0;
    // Comment out stream state
    // let streamActive = false;

    // Event Listeners
    if (uploadArea) {
        uploadArea.addEventListener('click', () => videoFileInput.click());
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);
    } else {
        console.error('Upload area element not found');
    }

    if (videoFileInput) {
        videoFileInput.addEventListener('change', handleFileSelect);
    } else {
        console.error('Video input element not found');
    }

    if (startDetectionBtn) {
        startDetectionBtn.addEventListener('click', startDetection);
    } else {
        console.error('Start detection button not found');
    }

    if (closeModal) {
        closeModal.addEventListener('click', () => modal.style.display = 'none');
    }

    if (closeModalBtn) {
        closeModalBtn.addEventListener('click', () => modal.style.display = 'none');
    }

    if (downloadChallanBtn) {
        downloadChallanBtn.addEventListener('click', downloadChallan);
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
        uploadedFile = file;
        uploadedFileName = file.name;
        
        if (previewPlayer) {
            const videoUrl = URL.createObjectURL(file);
            previewPlayer.src = videoUrl;
        }
        
        if (videoPreview) {
            videoPreview.style.display = 'block';
        }

        // Update upload area to show selected file name
        if (uploadArea) {
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
    }

    // API interaction functions
    async function uploadVideo() {
        if (!uploadedFile) {
            showToast('Please select a video file first');
            return;
        }

        try {
            if (progressContainer) {
                progressContainer.style.display = 'block';
            }

            const formData = new FormData();
            formData.append('file', uploadedFile);

            const xhr = new XMLHttpRequest();
            xhr.open('POST', UPLOAD_URL, true);

            xhr.upload.onprogress = (e) => {
                if (e.lengthComputable && progressBar && progressText) {
                    const percentComplete = Math.round((e.loaded / e.total) * 100);
                    progressBar.style.width = `${percentComplete}%`;
                    progressText.textContent = `${percentComplete}%`;
                }
            };

            xhr.onload = function() {
                if (xhr.status >= 200 && xhr.status < 300) {
                    const response = JSON.parse(xhr.responseText);
                    // Use path instead of filename for the video_path
                    uploadedFileName = response.path;
                    showToast('Video uploaded successfully');
                    if (startDetectionBtn) {
                        startDetectionBtn.disabled = false;
                    }
                } else {
                    showToast('Upload failed: ' + xhr.statusText);
                }
            };

            xhr.onerror = function() {
                showToast('Upload failed. Please try again');
            };

            xhr.send(formData);
        } catch (error) {
            showToast('Error uploading video: ' + error.message);
        }
    }

    async function startDetection() {
        if (!uploadedFileName) {
            showToast('Please upload a video first');
            return;
        }

        try {
            startDetectionBtn.disabled = true;
            detectionLoader.style.display = 'flex';
            violationsContainer.innerHTML = '';

            // Create formData with the exact field name expected by the backend
            const formData = new FormData();
            formData.append('video_path', uploadedFileName);

            console.log('Sending detection request with video path:', uploadedFileName);
            
            const response = await fetch(START_DETECTION_URL, {
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
            console.error('Error starting detection:', error);
            showToast('Error starting detection: ' + error.message);
            startDetectionBtn.disabled = false;
            detectionLoader.style.display = 'none';
        }
    }

    function startPollingForViolations() {
        // Initial count of violations before polling
        const initialViolationCount = currentViolationsCount;
        let pollCount = 0;
        const maxPolls = 12; // Poll for 1 minute max (5 seconds interval * 12)
        
        // Show loading indicator
        detectionLoader.style.display = 'flex';
        
        if (pollingInterval) {
            clearInterval(pollingInterval);
        }
        
        pollingInterval = setInterval(async () => {
            pollCount++;
            console.log(`Polling for violations (${pollCount}/${maxPolls})`);
            
            try {
                const newViolations = await fetchViolations();
                
                // Debug log
                console.log(`Current violations: ${newViolations.length}, Initial: ${initialViolationCount}`);
                
                // Check if we found new violations or reached max polling attempts
                if (pollCount >= maxPolls) {
                    // Stop polling after max attempts
                    clearInterval(pollingInterval);
                    detectionLoader.style.display = 'none';
                    startDetectionBtn.disabled = false;
                    
                    if (newViolations.length > initialViolationCount) {
                        showToast(`Detection complete! Found ${newViolations.length - initialViolationCount} new violations.`);
                    } else {
                        showToast('Detection complete. No new violations found.');
                        // Still display existing violations even if no new ones
                        displayViolations(newViolations, 0);
                    }
                } else if (newViolations.length > initialViolationCount) {
                    // New violations found - stop polling
                    clearInterval(pollingInterval);
                    detectionLoader.style.display = 'none';
                    startDetectionBtn.disabled = false;
                    showToast(`Detection complete! Found ${newViolations.length - initialViolationCount} new violations.`);
                } else if (pollCount % 2 === 0) {
                    // Every 10 seconds, update the user
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
            console.log('Fetching violations from server');
            const response = await fetch(GET_VIOLATIONS_URL);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            const violations = data.violations || [];
            console.log(`Received ${violations.length} violations from server`);
            
            // Always display violations, even if count didn't change
            displayViolations(violations, 0);
            
            // Store current violations count
            currentViolationsCount = violations.length;
            
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
        if (toast) {
            toast.textContent = message;
            toast.classList.add('show');
            
            setTimeout(() => {
                toast.classList.remove('show');
            }, 3000);
        } else {
            console.log("Toast message:", message);
        }
    }

    // Remove any existing stream button event listeners
    // and replace with this new implementation
    /* Commented out live stream functionality
    function initializeStreamControls() {
        const startStreamBtn = document.getElementById('start-stream-btn');
        const stopStreamBtn = document.getElementById('stop-stream-btn');
        const streamPreview = document.getElementById('live-stream-preview');
        const streamLoader = document.getElementById('stream-loader');
        
        // Add state tracking
        let isStoppingStream = false;
        let streamErrorCount = 0;
        const MAX_ERROR_RETRIES = 3;

        console.log('Initializing stream controls');
        console.log('Start button found:', !!startStreamBtn);
        console.log('Stop button found:', !!stopStreamBtn);

        // Define the stopLiveStream function
        async function stopLiveStream(isError = false) {
            // Prevent multiple stop requests
            if (isStoppingStream) {
                console.log('Stop already in progress, skipping');
                return;
            }

            console.log('Stopping stream due to', isError ? 'error' : 'user action');
            isStoppingStream = true;

            try {
                // Update UI first
                startStreamBtn.disabled = false;
                stopStreamBtn.disabled = true;
                streamPreview.style.display = 'none';
                streamLoader.style.display = 'none';
                
                // Only try to stop the stream if it wasn't due to a connection error
                if (!isError) {
                    const response = await fetch(`${API_BASE_URL}/live-stream/stop`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    });
                    
                    if (!response.ok) {
                        throw new Error(`Failed to stop stream: ${response.status}`);
                    }
                }
                
                showToast('Stream stopped');
            } catch (error) {
                console.error('Error stopping stream:', error);
                // Don't show error toast if it was already an error condition
                if (!isError) {
                    showToast('Error stopping stream');
                }
            } finally {
                isStoppingStream = false;
                streamErrorCount = 0; // Reset error count
            }
        }

        if (startStreamBtn) {
            startStreamBtn.addEventListener('click', function() {
                console.log('Start button clicked');
                streamLoader.style.display = 'flex';
                streamErrorCount = 0; // Reset error count on new start
                
                fetch(`${API_BASE_URL}/live-stream/start`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                })
                .then(response => {
                    console.log('Start stream response:', response.status);
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    // Update UI
                    startStreamBtn.disabled = true;
                    stopStreamBtn.disabled = false;
                    
                    // Set up the stream display
                    streamLoader.style.display = 'none';
                    streamPreview.style.display = 'block';
                    
                    // Use the feed endpoint for the stream
                    streamPreview.src = `${API_BASE_URL}/live-stream/feed`;
                    
                    // Add error handler for stream
                    streamPreview.onerror = (error) => {
                        console.error('Stream connection failed:', error);
                        streamErrorCount++;
                        
                        if (streamErrorCount <= MAX_ERROR_RETRIES) {
                            console.log(`Retrying stream (attempt ${streamErrorCount})`);
                            showToast(`Stream connection lost. Retrying... (${streamErrorCount}/${MAX_ERROR_RETRIES})`);
                            
                            // Retry the stream connection with exponential backoff
                            setTimeout(() => {
                                if (streamPreview) {
                                    streamPreview.src = `${API_BASE_URL}/live-stream/feed?t=${Date.now()}`;
                                }
                            }, 1000 * Math.pow(2, streamErrorCount - 1));
                        } else {
                            stopLiveStream(true);
                            showToast('Stream connection lost after multiple retries. Please try again.');
                        }
                    };
                    
                    // Add a load event handler
                    streamPreview.onload = () => {
                        console.log('Stream connected successfully');
                        streamErrorCount = 0; // Reset error count on successful connection
                        streamLoader.style.display = 'none';
                    };
                    
                    showToast('Live stream started');
                })
                .catch(error => {
                    console.error('Error starting stream:', error);
                    streamLoader.style.display = 'none';
                    showToast('Failed to start stream: ' + error.message);
                });
            });
        }

        if (stopStreamBtn) {
            stopStreamBtn.addEventListener('click', () => stopLiveStream(false));
        }
    }

    // Call the initialization function
    initializeStreamControls();
    */
})