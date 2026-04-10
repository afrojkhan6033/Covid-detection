document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('results');
    const heroSection = document.querySelector('.hero-section');
    const previewImg = document.getElementById('preview-img');
    const resetBtn = document.getElementById('reset-btn');

    const API_URL = 'http://127.0.0.1:8000/predict';
    let confidenceChart = null;

    // --- Event Listeners ---
    
    // Drag & Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('highlight-drag'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('highlight-drag'), false);
    });

    dropZone.addEventListener('drop', handleDrop, false);
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    resetBtn.addEventListener('click', resetUI);

    // --- Main Logic ---

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            processFile(files[0]);
        }
    }

    function handleFileSelect(e) {
        if (e.target.files.length > 0) {
            processFile(e.target.files[0]);
        }
    }

    async function processFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file (JPG, PNG).');
            return;
        }

        // Show preview immediately
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
        };
        reader.readAsDataURL(file);

        // UI State: Loading
        heroSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        loader.classList.remove('hidden');

        // Prepare Data
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Prediction failed');
            }

            const data = await response.json();
            displayResults(data);
        } catch (err) {
            console.error(err);
            alert(`Error: ${err.message}`);
            resetUI();
        }
    }

    function displayResults(data) {
        // UI State: Results
        loader.classList.add('hidden');
        resultsSection.classList.remove('hidden');

        // 1. Final Verdict
        const verdictEl = document.getElementById('final-verdict');
        verdictEl.textContent = data.final_verdict;
        verdictEl.className = 'verdict-value ' + (data.final_verdict === 'COVID-19' ? 'covid' : 'normal');
        
        document.getElementById('avg-confidence').textContent = `Confidence: ${data.average_confidence_percent}%`;

        // 2. Individual Model Details
        const models = data.model_results;
        
        // ResNet50
        updateModelUI('res', models.ResNet50);
        // VGG16
        updateModelUI('vgg', models.VGG16);
        // Xception
        updateModelUI('xcp', models.Xception);

        // 3. Render Chart
        renderChart(models);
    }

    function updateModelUI(prefix, data) {
        const label = document.getElementById(`${prefix}-label`);
        const bar = document.getElementById(`${prefix}-bar`);
        const conf = document.getElementById(`${prefix}-conf`);

        label.textContent = data.label;
        label.style.color = data.label === 'COVID-19' ? 'var(--danger)' : 'var(--success)';
        
        conf.textContent = `${data.confidence_percent}%`;
        
        // Use a short delay to trigger the transition
        setTimeout(() => {
            bar.style.width = `${data.confidence_percent}%`;
            bar.style.backgroundColor = data.label === 'COVID-19' ? 'var(--danger)' : 'var(--success)';
        }, 50);
    }

    function renderChart(models) {
        const ctx = document.getElementById('confidenceChart').getContext('2d');
        
        const labels = Object.keys(models);
        const covidData = labels.map(key => models[key].raw_scores[0]);
        const normalData = labels.map(key => models[key].raw_scores[1]);

        if (confidenceChart) {
            confidenceChart.destroy();
        }

        confidenceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'COVID-19',
                        data: covidData,
                        backgroundColor: 'rgba(239, 68, 68, 0.7)',
                        borderColor: '#ef4444',
                        borderWidth: 1
                    },
                    {
                        label: 'NORMAL',
                        data: normalData,
                        backgroundColor: 'rgba(16, 185, 129, 0.7)',
                        borderColor: '#10b981',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#f0f4f9', font: { family: 'Inter' } }
                    }
                }
            }
        });
    }

    function resetUI() {
        heroSection.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        loader.classList.add('hidden');
        fileInput.value = '';
        
        // Reset bars
        ['res-bar', 'vgg-bar', 'xcp-bar'].forEach(id => {
            document.getElementById(id).style.width = '0%';
        });
    }
});
