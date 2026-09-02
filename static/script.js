document.addEventListener('DOMContentLoaded', () => {
    // State Variables
    let metricsData = null;
    let historyData = null;
    let activeTab = 'dashboard';
    
    // DOM Elements
    const navButtons = document.querySelectorAll('.nav-btn');
    const viewPanels = document.querySelectorAll('.view-panel');
    const viewTitle = document.getElementById('view-title');
    const viewSubtitle = document.getElementById('view-subtitle');
    
    // Tab switching routing
    const viewMeta = {
        dashboard: { title: "Dashboard Overview", subtitle: "Summary of pedestrian detection models and performance." },
        detection: { title: "Image Detection", subtitle: "Upload an image or select a sample frame to detect pedestrians." },
        comparison: { title: "Performance Comparison", subtitle: "Interactive chart comparing Precision, Recall, and Accuracy." },
        history: { title: "Training History", subtitle: "Learning curves showing training accuracy and loss over 10 epochs." }
    };

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.target;
            
            // Update active state in nav buttons
            navButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Switch panels
            viewPanels.forEach(p => p.classList.remove('active'));
            const targetPanel = document.getElementById(`${target}-view`);
            if (targetPanel) {
                targetPanel.classList.add('active');
            }
            
            // Update header text
            if (viewMeta[target]) {
                viewTitle.textContent = viewMeta[target].title;
                viewSubtitle.textContent = viewMeta[target].subtitle;
            }

            activeTab = target;
            
            // Initialize charts if target view loaded
            if (target === 'comparison') {
                renderMetricsChart();
            } else if (target === 'history') {
                renderHistoryCharts();
            }
        });
    });

    // --- FETCH DATA FROM SERVER ON LOAD ---
    fetchMetrics();
    fetchHistory();
    fetchSamples();

    async function fetchMetrics() {
        try {
            const response = await fetch('/api/metrics');
            metricsData = await response.json();
            populateMetricsTable(metricsData);
        } catch (error) {
            console.error('Error fetching metrics:', error);
            document.getElementById('dashboard-table-body').innerHTML = `
                <tr><td colspan="5" class="loading-cell" style="color: #f87171;">Failed to load metrics. Make sure backend is running.</td></tr>
            `;
        }
    }

    async function fetchHistory() {
        try {
            const response = await fetch('/api/history');
            historyData = await response.json();
        } catch (error) {
            console.error('Error fetching history:', error);
        }
    }

    function populateMetricsTable(data) {
        const tbody = document.getElementById('dashboard-table-body');
        tbody.innerHTML = '';
        
        const modelsConfig = [
            { id: 'frcnn', name: 'Existing Faster R-CNN' },
            { id: 'yolo', name: 'Proposed YOLOv5 + Coordinated Attention' },
            { id: 'yolov6', name: 'Extension YOLOv6 (VGG16 Backbone)' }
        ];

        modelsConfig.forEach(m => {
            const rowMetrics = data[m.id];
            if (!rowMetrics) return;
            
            const tr = document.createElement('tr');
            
            // Highlight Proposed and Extension models
            let displayName = m.name;
            if (m.id === 'yolo') displayName = `✨ <strong>${m.name}</strong>`;
            if (m.id === 'yolov6') displayName = `🚀 <strong>${m.name}</strong>`;

            tr.innerHTML = `
                <td>${displayName}</td>
                <td><span style="color: ${m.id === 'yolov6' ? '#10b981' : '#3b82f6'}; font-weight: 600;">${rowMetrics.accuracy.toFixed(2)}%</span></td>
                <td>${rowMetrics.precision.toFixed(2)}%</td>
                <td>${rowMetrics.recall.toFixed(2)}%</td>
                <td>${rowMetrics.f1.toFixed(2)}%</td>
            `;
            tbody.appendChild(tr);
        });
    }

    // --- CHART PLOTTING WITH CHARTJS ---
    let metricsChartInstance = null;
    function renderMetricsChart() {
        if (!metricsData) return;
        const ctx = document.getElementById('metricsChart').getContext('2d');
        if (metricsChartInstance) {
            metricsChartInstance.destroy();
        }

        const labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score'];
        metricsChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Existing Faster R-CNN',
                        data: [metricsData.frcnn.accuracy, metricsData.frcnn.precision, metricsData.frcnn.recall, metricsData.frcnn.f1],
                        backgroundColor: 'rgba(59, 130, 246, 0.7)',
                        borderColor: '#3b82f6',
                        borderWidth: 1.5
                    },
                    {
                        label: 'Proposed YOLOv5 + CA',
                        data: [metricsData.yolo.accuracy, metricsData.yolo.precision, metricsData.yolo.recall, metricsData.yolo.f1],
                        backgroundColor: 'rgba(139, 92, 246, 0.7)',
                        borderColor: '#8b5cf6',
                        borderWidth: 1.5
                    },
                    {
                        label: 'Extension YOLOv6',
                        data: [metricsData.yolov6.accuracy, metricsData.yolov6.precision, metricsData.yolov6.recall, metricsData.yolov6.f1],
                        backgroundColor: 'rgba(16, 185, 129, 0.7)',
                        borderColor: '#10b981',
                        borderWidth: 1.5
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#94a3b8', font: { family: 'Outfit' } } }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#94a3b8', font: { family: 'Outfit' } }
                    },
                    y: {
                        min: 50,
                        max: 100,
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#94a3b8', font: { family: 'Outfit' } }
                    }
                }
            }
        });
    }

    let accuracyChartInstance = null;
    let lossChartInstance = null;
    
    function renderHistoryCharts() {
        if (!historyData) return;
        
        const chartOptions = (title, minVal, maxVal) => ({
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8', font: { family: 'Outfit' } } }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8', font: { family: 'Outfit' } },
                    title: { display: true, text: 'Epoch', color: '#94a3b8', font: { family: 'Outfit' } }
                },
                y: {
                    min: minVal,
                    max: maxVal,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8', font: { family: 'Outfit' } }
                }
            }
        });

        // Accuracy Chart
        const ctxAcc = document.getElementById('accuracyChart').getContext('2d');
        if (accuracyChartInstance) accuracyChartInstance.destroy();
        accuracyChartInstance = new Chart(ctxAcc, {
            type: 'line',
            data: {
                labels: historyData.epochs,
                datasets: [
                    {
                        label: 'Existing Faster R-CNN',
                        data: historyData.frcnn.accuracy,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.3,
                        pointBackgroundColor: '#10b981'
                    },
                    {
                        label: 'Proposed YOLOv5 + CA',
                        data: historyData.yolo.accuracy,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.3,
                        pointBackgroundColor: '#3b82f6'
                    },
                    {
                        label: 'Extension YOLOv6',
                        data: historyData.yolov6.accuracy,
                        borderColor: '#facc15',
                        backgroundColor: 'rgba(250, 204, 21, 0.1)',
                        tension: 0.3,
                        pointBackgroundColor: '#facc15'
                    }
                ]
            },
            options: chartOptions('Accuracy', 0.4, 1.05)
        });

        // Loss Chart
        const ctxLoss = document.getElementById('lossChart').getContext('2d');
        if (lossChartInstance) lossChartInstance.destroy();
        lossChartInstance = new Chart(ctxLoss, {
            type: 'line',
            data: {
                labels: historyData.epochs,
                datasets: [
                    {
                        label: 'Existing Faster R-CNN',
                        data: historyData.frcnn.loss,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.3
                    },
                    {
                        label: 'Proposed YOLOv5 + CA',
                        data: historyData.yolo.loss,
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.3
                    },
                    {
                        label: 'Extension YOLOv6',
                        data: historyData.yolov6.loss,
                        borderColor: '#facc15',
                        backgroundColor: 'rgba(250, 204, 21, 0.1)',
                        tension: 0.3
                    }
                ]
            },
            options: chartOptions('Loss', 0.0, 7.0)
        });
    }

    // --- DETECT COMPONENT & UPLOAD HANDLING ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const imgOrig = document.getElementById('img-orig');
    const imgResult = document.getElementById('img-result');
    const placeholderOrig = document.getElementById('placeholder-orig');
    const placeholderResult = document.getElementById('placeholder-result');
    const loader = document.getElementById('loader');
    const samplesContainer = document.getElementById('samples-container');
    const inferenceSummary = document.getElementById('inference-summary');
    const pedCount = document.getElementById('ped-count');
    const pedStatus = document.getElementById('ped-status');

    // Drag-and-drop styling triggers
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    function handleFileSelection(file) {
        // Clear active class on sample frame list
        document.querySelectorAll('.sample-img-card').forEach(c => c.classList.remove('selected'));
        
        // Render preview of selected input image
        const reader = new FileReader();
        reader.onload = (e) => {
            imgOrig.src = e.target.result;
            imgOrig.style.display = 'block';
            placeholderOrig.style.display = 'none';
        };
        reader.readAsDataURL(file);

        // Upload and Predict
        uploadAndPredict(file);
    }

    async function uploadAndPredict(file) {
        // UI Loading States
        placeholderResult.style.display = 'none';
        imgResult.style.display = 'none';
        loader.style.display = 'block';
        inferenceSummary.style.display = 'none';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Prediction API failed');
            }

            const data = await response.json();
            
            // Show annotated result
            imgResult.src = data.image;
            imgResult.style.display = 'block';
            loader.style.display = 'none';

            // Show metrics summary
            pedCount.textContent = data.pedestrians_count;
            pedStatus.textContent = data.class_label;
            inferenceSummary.style.display = 'flex';
            
            // Apply status badge color
            if (data.pedestrians_count > 0) {
                pedStatus.style.background = 'rgba(239, 68, 68, 0.15)';
                pedStatus.style.color = '#f87171';
                pedStatus.style.borderColor = 'rgba(239, 68, 68, 0.25)';
            } else {
                pedStatus.style.background = 'rgba(59, 130, 246, 0.15)';
                pedStatus.style.color = '#60a5fa';
                pedStatus.style.borderColor = 'rgba(59, 130, 246, 0.25)';
            }

        } catch (error) {
            console.error('Error run prediction:', error);
            loader.style.display = 'none';
            placeholderResult.style.display = 'block';
            placeholderResult.innerHTML = `
                <div class="placeholder-icon" style="color: #f87171;">⚠️</div>
                <p style="color: #f87171;">Error performing prediction on image. Make sure server is online.</p>
            `;
        }
    }

    // --- SAMPLE CAROUSEL LOADER ---
    async function fetchSamples() {
        try {
            const response = await fetch('/api/samples');
            const files = await response.json();
            renderSamples(files);
        } catch (error) {
            console.error('Error fetching sample list:', error);
        }
    }

    function renderSamples(files) {
        samplesContainer.innerHTML = '';
        if (files.length === 0) {
            samplesContainer.innerHTML = '<p class="section-desc">No sample images found.</p>';
            return;
        }

        // Limit to first 8 to avoid cluttering
        files.slice(0, 10).forEach(file => {
            const card = document.createElement('div');
            card.className = 'sample-img-card';
            card.innerHTML = `<img src="/api/samples/${file}" alt="${file}">`;
            
            card.addEventListener('click', async () => {
                // Toggle active border styling
                document.querySelectorAll('.sample-img-card').forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');

                // Render original image preview directly
                imgOrig.src = `/api/samples/${file}`;
                imgOrig.style.display = 'block';
                placeholderOrig.style.display = 'none';

                try {
                    const res = await fetch(`/api/samples/${file}`);
                    const blob = await res.blob();
                    const fileObj = new File([blob], file, { type: blob.type });
                    uploadAndPredict(fileObj);
                } catch (error) {
                    console.error('Failed to load sample image preview:', error);
                }
            });

            samplesContainer.appendChild(card);
        });
    }

    // Base64 helper for sample frame blob upload
    function base64ToBlob(base64, mime) {
        const byteCharacters = atob(base64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        return new Blob([byteArray], { type: mime });
    }
});
