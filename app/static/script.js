/**
 * CosmoClassifier - Professional Frontend JavaScript
 * Handles form submission, tab navigation, and UI interactions
 */

document.addEventListener("DOMContentLoaded", function() {
    // Initialize components
    initTabNavigation();
    initPredictForm();
    initBatchPredictForm();
    initKeyboardShortcuts();
});

/**
 * Tab Navigation System
 */
function initTabNavigation() {
    const navBtns = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.dataset.tab;
            
            // Update active nav button
            navBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Show target tab content
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === `${targetTab}-tab`) {
                    content.classList.add('active');
                }
            });
            
            // Scroll to top of content
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    });
}

/**
 * Prediction Form Handler
 */
function initPredictForm() {
    const form = document.getElementById("predictForm");
    const resultDiv = document.getElementById("result");
    const awaitingDiv = document.getElementById("awaiting-prediction");
    const predClassDiv = document.getElementById("predClass");
    const probsDiv = document.getElementById("probabilities");
    const submitBtn = document.getElementById("predictBtn");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        // Set loading state
        setLoadingState(submitBtn, true);
        
        // Hide awaiting state, show result
        awaitingDiv.style.display = 'none';
        resultDiv.classList.add("hidden");
        document.getElementById('result-stats').classList.add('hidden');
        
        try {
            const formData = new FormData(form);
            const dataObj = Object.fromEntries(formData.entries());
            
            // Convert numeric strings to numbers
            for (let key in dataObj) {
                if (!isNaN(dataObj[key]) && dataObj[key] !== "") {
                    dataObj[key] = parseFloat(dataObj[key]);
                }
            }

            const response = await fetch("/predict", { 
                method: "POST", 
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(dataObj)
            });

            if (!response.ok) {
                const errorData = await response.json();
                const errorMessage = errorData.error || errorData.message || "Prediction failed";
                throw new Error(errorMessage);
            }

            const data = await response.json();
            
            // Render prediction result
            renderPredictionResult(data, predClassDiv, probsDiv);
            
            // Show results with animation
            resultDiv.classList.remove("hidden");
            document.getElementById('result-stats').classList.remove('hidden');
            
            // Scroll to result if on mobile
            if (window.innerWidth < 900) {
                resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
            
            // Show success toast
            showToast('Classification complete!', 'success');
            
        } catch (error) {
            console.error(error);
            showToast(`Error: ${error.message}`, 'error');
            // Show awaiting state again on error
            awaitingDiv.style.display = 'flex';
            resultDiv.classList.add("hidden");
            document.getElementById('result-stats').classList.add('hidden');
        } finally {
            setLoadingState(submitBtn, false);
        }
    });

    // Clear Form Handler
    const clearBtn = document.getElementById("clearBtn");
    clearBtn.addEventListener("click", () => {
        clearForm();
    });

    function clearForm() {
        form.reset();
        
        // Reset UI states
        resultDiv.classList.add("hidden");
        document.getElementById('result-stats').classList.add('hidden');
        awaitingDiv.style.display = 'flex';
        
        // Show success toast
        showToast('Form cleared', 'info');
    }

    // Export clearForm for use in keyboard shortcuts
    window.clearForm = clearForm;
}

/**
 * Set button loading state
 */
function setLoadingState(btn, isLoading) {
    const btnText = btn.querySelector('.btn-text');
    const btnIcon = btn.querySelector('.btn-icon');
    
    if (isLoading) {
        btn.disabled = true;
        btnText.textContent = 'Analyzing...';
        btnIcon.innerHTML = '<div class="spinner"></div>';
        btn.style.opacity = '0.8';
    } else {
        btn.disabled = false;
        btnText.textContent = 'Analyze Spectrum';
        btnIcon.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M5 12h14M12 5l7 7-7 7"/>
        </svg>`;
        btn.style.opacity = '1';
    }
}

/**
 * Render prediction results with animations
 */
function renderPredictionResult(data, predClassDiv, probsDiv) {
    const pred = data.prediction;
    const probs = data.probabilities;
    
    // Set prediction class with styling
    predClassDiv.innerHTML = `<span class="pred-class" data-type="${pred.toLowerCase()}">${pred}</span>`;
    
    // Render probability bars
    probsDiv.innerHTML = "";
    
    Object.entries(probs).forEach(([cls, prob], index) => {
        const percentage = (prob * 100).toFixed(1);
        const clsType = cls.toLowerCase();
        
        probsDiv.innerHTML += `
            <div class="prob-row" style="animation-delay: ${index * 0.1}s">
                <span class="prob-name ${clsType}">${cls}</span>
                <div class="prob-bar-container">
                    <div class="prob-fill ${clsType}" style="width: 0%" data-width="${percentage}%"></div>
                </div>
                <span class="prob-value">${percentage}%</span>
            </div>`;
    });
    
    // Trigger animation after small delay
    setTimeout(() => {
        const bars = document.querySelectorAll(".prob-fill");
        bars.forEach(bar => {
            bar.style.width = bar.getAttribute("data-width");
        });
    }, 100);
}

/**
 * Keyboard shortcuts for power users
 */
function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Tab switching with number keys (when not in input)
        if (!e.target.matches('input, textarea')) {
            switch(e.key) {
                case '1':
                    document.querySelector('[data-tab="predict"]').click();
                    break;
                case '2':
                    document.querySelector('[data-tab="model"]').click();
                    break;
            }
        }
        
        // Escape to clear form
        if (e.key === 'Escape') {
            if (window.clearForm) window.clearForm();
        }
    });
}

/**
 * Toast Notification System
 */
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            ${getToastIcon(type)}
        </svg>
        <span class="toast-message">${message}</span>
    `;
    
    container.appendChild(toast);
    
    // Auto-remove after 4 seconds
    setTimeout(() => {
        toast.classList.add('toast-out');
        setTimeout(() => {
            toast.remove();
        }, 300);
    }, 4000);
}

/**
 * Get appropriate icon for toast type
 */
function getToastIcon(type) {
    const icons = {
        success: '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
        error: '<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>',
        warning: '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
        info: '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>'
    };
    return icons[type] || icons.info;
}

/**
 * Utility: Format numbers with commas
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/**
 * Utility: Debounce function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Export for potential module use
window.CosmoClassifier = {
    showToast,
    formatNumber,
    debounce
};

/**
 * Batch Prediction Form Handler
 */
function initBatchPredictForm() {
    const fileInput = document.getElementById("batchFile");
    const dropzone = document.getElementById("fileUploadDropzone");
    const fileInfo = document.getElementById("fileInfo");
    const selectedFileName = document.getElementById("selectedFileName");
    const selectedFileSize = document.getElementById("selectedFileSize");
    const removeFileBtn = document.getElementById("removeFileBtn");
    const submitBtn = document.getElementById("batchPredictBtn");
    const form = document.getElementById("batchPredictForm");
    
    if(!fileInput) return;

    const awaitingBatch = document.getElementById("awaiting-batch");
    const batchResult = document.getElementById("batch-result");
    const tbody = document.getElementById("batchTableBody");

    const MAX_SIZE = 5 * 1024 * 1024; // 5MB

    function handleFile(file) {
        if (!file) return;
        
        if (!file.name.toLowerCase().endsWith('.csv')) {
            showToast("Only .csv files are allowed", "error");
            fileInput.value = "";
            return;
        }
        
        if (file.size > MAX_SIZE) {
            showToast("File size exceeds 5MB limit", "error");
            fileInput.value = "";
            return;
        }

        selectedFileName.textContent = file.name;
        selectedFileSize.textContent = (file.size / 1024 / 1024).toFixed(2) + " MB";
        
        dropzone.classList.add("hidden");
        fileInfo.classList.remove("hidden");
        submitBtn.disabled = false;
    }

    fileInput.addEventListener("change", (e) => {
        handleFile(e.target.files[0]);
    });

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => {
            dropzone.style.borderColor = "var(--accent-secondary)";
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => {
            dropzone.style.borderColor = "";
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        let dt = e.dataTransfer;
        let files = dt.files;
        if (files.length) {
            fileInput.files = files;
            handleFile(files[0]);
        }
    }, false);

    removeFileBtn.addEventListener("click", () => {
        fileInput.value = "";
        dropzone.classList.remove("hidden");
        fileInfo.classList.add("hidden");
        submitBtn.disabled = true;
    });

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        if (!fileInput.files[0]) return;

        setLoadingState(submitBtn, true);
        awaitingBatch.style.display = 'none';
        batchResult.classList.add("hidden");

        try {
            const formData = new FormData();
            formData.append("payload", fileInput.files[0]);

            const response = await fetch("/predict/file", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || errorData.message || "Batch prediction failed");
            }

            const data = await response.json();
            
            renderBatchResults(data);
            showToast('Batch classification complete!', 'success');
            
        } catch (error) {
            console.error(error);
            showToast(`Error: ${error.message}`, 'error');
            awaitingBatch.style.display = 'flex';
        } finally {
            setLoadingState(submitBtn, false);
        }
    });

    let batchChart = null;

    function renderBatchResults(data) {
        const preds = data.prediction;
        const probs = data.probabilities;
        
        tbody.innerHTML = "";
        
        let counts = { "GALAXY": 0, "STAR": 0, "QSO": 0 };

        preds.forEach((pred, index) => {
            counts[pred] = (counts[pred] || 0) + 1;
            
            const maxProb = Math.max(...probs[index]);
            
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>Row ${index + 1}</td>
                <td><span class="pred-class ${pred.toLowerCase()}" style="font-size: 0.85rem; padding: 2px 6px; border-radius: 4px; background: rgba(255,255,255,0.1);">${pred}</span></td>
                <td>${maxProb}%</td>
            `;
            tbody.appendChild(tr);
        });

        batchResult.classList.remove("hidden");

        // Render Chart
        const ctx = document.getElementById('batchPieChart').getContext('2d');
        if (batchChart) batchChart.destroy();

        batchChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['GALAXY', 'STAR', 'QSO'],
                datasets: [{
                    data: [counts['GALAXY'], counts['STAR'], counts['QSO']],
                    backgroundColor: ['#7000ff', '#00d4ff', '#ff6b6b'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#ffffff' }
                    }
                }
            }
        });
    }
}
