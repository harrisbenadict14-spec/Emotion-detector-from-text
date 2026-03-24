// AI Emotion Detector - Frontend JavaScript

// Configuration
const API_BASE_URL = 'http://localhost:8000';

// State Management
let analysisHistory = JSON.parse(localStorage.getItem('emotionHistory')) || [];
let currentAnalysis = null;

// DOM Elements
const elements = {
    textInput: document.getElementById('text-input'),
    analyzeBtn: document.getElementById('analyze-btn'),
    resultsSection: document.getElementById('results-section'),
    primaryEmotion: document.getElementById('primary-emotion'),
    emotionIcon: document.getElementById('emotion-icon'),
    emotionName: document.getElementById('emotion-name'),
    confidenceFill: document.getElementById('confidence-fill'),
    confidenceText: document.getElementById('confidence-text'),
    probabilities: document.getElementById('probabilities'),
    probabilityBars: document.getElementById('probability-bars'),
    cleanedText: document.getElementById('cleaned-text'),
    cleanedTextContent: document.getElementById('cleaned-text-content'),
    saveBtn: document.getElementById('save-btn'),
    exportBtn: document.getElementById('export-btn'),
    clearBtn: document.getElementById('clear-btn'),
    historySection: document.getElementById('history-section'),
    historyContent: document.getElementById('history-content'),
    clearHistoryBtn: document.getElementById('clear-history-btn'),
    statsSection: document.getElementById('stats-section'),
    totalAnalyses: document.getElementById('total-analyses'),
    mostCommon: document.getElementById('most-common'),
    avgConfidence: document.getElementById('avg-confidence'),
    loadingOverlay: document.getElementById('loading-overlay'),
    toast: document.getElementById('toast'),
    toastMessage: document.getElementById('toast-message'),
    themeBtn: document.getElementById('theme-btn'),
    charCurrent: document.getElementById('char-current'),
    typingText: document.getElementById('typing-text')
};

// Emotion Configuration
const emotionConfig = {
    happy: { icon: 'fa-smile', color: '#f6ad55' },
    sad: { icon: 'fa-sad-tear', color: '#4299e1' },
    angry: { icon: 'fa-angry', color: '#fc8181' },
    fear: { icon: 'fa-grimace', color: '#9f7aea' },
    surprise: { icon: 'fa-surprise', color: '#48bb78' },
    neutral: { icon: 'fa-meh', color: '#718096' }
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    setupEventListeners();
    loadHistory();
    updateStatistics();
    createParticles();
    startTypingAnimation();
    checkApiHealth();
}

// Event Listeners
function setupEventListeners() {
    elements.analyzeBtn.addEventListener('click', analyzeEmotion);
    elements.textInput.addEventListener('input', updateCharCount);
    elements.textInput.addEventListener('keydown', handleEnterKey);
    elements.saveBtn.addEventListener('click', saveToHistory);
    elements.exportBtn.addEventListener('click', exportResults);
    elements.clearBtn.addEventListener('click', clearResults);
    elements.clearHistoryBtn.addEventListener('click', clearHistory);
    elements.themeBtn.addEventListener('click', toggleTheme);
}

// Character Counter
function updateCharCount() {
    const current = elements.textInput.value.length;
    elements.charCurrent.textContent = current;
    
    if (current > 900) {
        elements.charCurrent.style.color = 'var(--error-color)';
    } else if (current > 700) {
        elements.charCurrent.style.color = 'var(--warning-color)';
    } else {
        elements.charCurrent.style.color = 'var(--text-secondary)';
    }
}

// Handle Enter Key
function handleEnterKey(event) {
    if (event.key === 'Enter' && event.ctrlKey) {
        analyzeEmotion();
    }
}

// Analyze Emotion
async function analyzeEmotion() {
    const text = elements.textInput.value.trim();
    
    if (!text) {
        showToast('Please enter some text to analyze', 'error');
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                include_probabilities: true
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        currentAnalysis = result;
        displayResults(result);
        showToast('Emotion analysis complete!', 'success');
        
    } catch (error) {
        console.error('Analysis error:', error);
        showToast(`Analysis failed: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

// Display Results
function displayResults(result) {
    // Update primary emotion
    const emotion = result.emotion;
    const confidence = result.confidence;
    const config = emotionConfig[emotion] || emotionConfig.neutral;
    
    elements.emotionName.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);
    elements.emotionIcon.innerHTML = `<i class="fas ${config.icon}"></i>`;
    elements.emotionIcon.style.color = config.color;
    elements.confidenceFill.style.width = `${confidence * 100}%`;
    elements.confidenceText.textContent = `${Math.round(confidence * 100)}%`;
    
    // Update probabilities
    if (result.probabilities) {
        displayProbabilities(result.probabilities);
    }
    
    // Update cleaned text
    if (result.cleaned_text) {
        elements.cleanedTextContent.textContent = result.cleaned_text;
        elements.cleanedText.style.display = 'block';
    } else {
        elements.cleanedText.style.display = 'none';
    }
    
    // Show results section
    elements.resultsSection.style.display = 'block';
    elements.resultsSection.classList.add('fade-in');
    
    // Scroll to results
    setTimeout(() => {
        elements.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 300);
}

// Display Probabilities
function displayProbabilities(probabilities) {
    elements.probabilityBars.innerHTML = '';
    
    // Sort probabilities by value
    const sortedProbs = Object.entries(probabilities)
        .sort(([,a], [,b]) => b - a);
    
    sortedProbs.forEach(([emotion, probability]) => {
        const config = emotionConfig[emotion] || emotionConfig.neutral;
        const percentage = Math.round(probability * 100);
        
        const probabilityItem = document.createElement('div');
        probabilityItem.className = 'probability-item';
        probabilityItem.innerHTML = `
            <span class="probability-label" style="color: ${config.color}">
                ${emotion.charAt(0).toUpperCase() + emotion.slice(1)}
            </span>
            <div class="probability-bar">
                <div class="probability-fill" style="width: ${percentage}%; background: ${config.color}"></div>
            </div>
            <span class="probability-value">${percentage}%</span>
        `;
        
        elements.probabilityBars.appendChild(probabilityItem);
    });
}

// Save to History
function saveToHistory() {
    if (!currentAnalysis) {
        showToast('No analysis to save', 'error');
        return;
    }
    
    const historyItem = {
        id: Date.now(),
        text: elements.textInput.value.trim(),
        emotion: currentAnalysis.emotion,
        confidence: currentAnalysis.confidence,
        probabilities: currentAnalysis.probabilities,
        cleanedText: currentAnalysis.cleaned_text,
        timestamp: new Date().toISOString()
    };
    
    analysisHistory.unshift(historyItem);
    
    // Keep only last 50 items
    if (analysisHistory.length > 50) {
        analysisHistory = analysisHistory.slice(0, 50);
    }
    
    localStorage.setItem('emotionHistory', JSON.stringify(analysisHistory));
    loadHistory();
    updateStatistics();
    showToast('Saved to history!', 'success');
}

// Load History
function loadHistory() {
    if (analysisHistory.length === 0) {
        elements.historyContent.innerHTML = '<p class="no-history">No analysis history yet. Start analyzing emotions!</p>';
        return;
    }
    
    elements.historyContent.innerHTML = '';
    
    analysisHistory.forEach(item => {
        const config = emotionConfig[item.emotion] || emotionConfig.neutral;
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        
        const date = new Date(item.timestamp);
        const formattedDate = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
        
        historyItem.innerHTML = `
            <div class="history-header-info">
                <span class="history-emotion" style="color: ${config.color}">
                    ${item.emotion.charAt(0).toUpperCase() + item.emotion.slice(1)}
                </span>
                <span class="history-confidence">${Math.round(item.confidence * 100)}% confidence</span>
            </div>
            <div class="history-text">${item.text}</div>
            <div class="history-timestamp">${formattedDate}</div>
        `;
        
        elements.historyContent.appendChild(historyItem);
    });
}

// Clear Results
function clearResults() {
    elements.resultsSection.style.display = 'none';
    elements.textInput.value = '';
    updateCharCount();
    currentAnalysis = null;
    showToast('Results cleared', 'success');
}

// Clear History
function clearHistory() {
    if (confirm('Are you sure you want to clear all history?')) {
        analysisHistory = [];
        localStorage.removeItem('emotionHistory');
        loadHistory();
        updateStatistics();
        showToast('History cleared', 'success');
    }
}

// Update Statistics
function updateStatistics() {
    if (analysisHistory.length === 0) {
        elements.statsSection.style.display = 'none';
        return;
    }
    
    elements.statsSection.style.display = 'block';
    
    // Total analyses
    elements.totalAnalyses.textContent = analysisHistory.length;
    
    // Most common emotion
    const emotionCounts = {};
    analysisHistory.forEach(item => {
        emotionCounts[item.emotion] = (emotionCounts[item.emotion] || 0) + 1;
    });
    
    const mostCommonEmotion = Object.entries(emotionCounts)
        .sort(([,a], [,b]) => b - a)[0][0];
    
    elements.mostCommon.textContent = mostCommonEmotion.charAt(0).toUpperCase() + mostCommonEmotion.slice(1);
    
    // Average confidence
    const avgConf = analysisHistory.reduce((sum, item) => sum + item.confidence, 0) / analysisHistory.length;
    elements.avgConfidence.textContent = Math.round(avgConf * 100) + '%';
}

// Export Results
function exportResults() {
    if (!currentAnalysis) {
        showToast('No results to export', 'error');
        return;
    }
    
    const csvContent = generateCSV();
    downloadCSV(csvContent, `emotion-analysis-${Date.now()}.csv`);
    showToast('Results exported!', 'success');
}

// Generate CSV
function generateCSV() {
    const headers = ['Timestamp', 'Text', 'Emotion', 'Confidence', 'Cleaned Text', ...Object.keys(emotionConfig)];
    const rows = [headers];
    
    if (currentAnalysis) {
        const timestamp = new Date().toISOString();
        const probabilities = currentAnalysis.probabilities || {};
        
        const row = [
            timestamp,
            `"${elements.textInput.value.trim()}"`,
            currentAnalysis.emotion,
            currentAnalysis.confidence,
            `"${currentAnalysis.cleaned_text || ''}"`,
            ...Object.keys(emotionConfig).map(emotion => probabilities[emotion] || 0)
        ];
        
        rows.push(row);
    }
    
    // Add history items
    analysisHistory.forEach(item => {
        const probabilities = item.probabilities || {};
        const row = [
            item.timestamp,
            `"${item.text}"`,
            item.emotion,
            item.confidence,
            `"${item.cleanedText || ''}"`,
            ...Object.keys(emotionConfig).map(emotion => probabilities[emotion] || 0)
        ];
        rows.push(row);
    });
    
    return rows.map(row => row.join(',')).join('\n');
}

// Download CSV
function downloadCSV(content, filename) {
    const blob = new Blob([content], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

// Theme Toggle
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    
    const icon = elements.themeBtn.querySelector('i');
    icon.className = newTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
}

// Load Theme
function loadTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    
    const icon = elements.themeBtn.querySelector('i');
    icon.className = savedTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
}

// Loading Overlay
function showLoading(show) {
    elements.loadingOverlay.style.display = show ? 'flex' : 'none';
}

// Toast Notification
function showToast(message, type = 'success') {
    elements.toastMessage.textContent = message;
    
    // Set background color based on type
    if (type === 'error') {
        elements.toast.style.background = 'var(--error-color)';
    } else if (type === 'warning') {
        elements.toast.style.background = 'var(--warning-color)';
    } else {
        elements.toast.style.background = 'var(--success-color)';
    }
    
    elements.toast.classList.add('show');
    
    setTimeout(() => {
        elements.toast.classList.remove('show');
    }, 3000);
}

// Create Particles
function createParticles() {
    const particlesContainer = document.getElementById('particles');
    const particleCount = 50;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 20 + 's';
        particle.style.animationDuration = (15 + Math.random() * 10) + 's';
        particlesContainer.appendChild(particle);
    }
}

// Typing Animation
function startTypingAnimation() {
    const phrases = [
        'Discover the emotions behind your words',
        'Unlock the power of emotional AI',
        'Analyze text with advanced machine learning',
        'Understand feelings through technology'
    ];
    
    let phraseIndex = 0;
    let charIndex = 0;
    let isDeleting = false;
    
    function typeText() {
        const currentPhrase = phrases[phraseIndex];
        
        if (isDeleting) {
            elements.typingText.textContent = currentPhrase.substring(0, charIndex - 1);
            charIndex--;
        } else {
            elements.typingText.textContent = currentPhrase.substring(0, charIndex + 1);
            charIndex++;
        }
        
        let typeSpeed = isDeleting ? 50 : 100;
        
        if (!isDeleting && charIndex === currentPhrase.length) {
            typeSpeed = 2000;
            isDeleting = true;
        } else if (isDeleting && charIndex === 0) {
            isDeleting = false;
            phraseIndex = (phraseIndex + 1) % phrases.length;
            typeSpeed = 500;
        }
        
        setTimeout(typeText, typeSpeed);
    }
    
    typeText();
}

// Check API Health
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        
        if (response.ok) {
            const health = await response.json();
            if (health.model_loaded) {
                console.log('✅ API is healthy and model is loaded');
            } else {
                console.warn('⚠️ API is healthy but model is not loaded');
                showToast('Model is not loaded. Please train the model first.', 'warning');
            }
        } else {
            console.error('❌ API health check failed');
            showToast('API is not available. Please start the backend server.', 'error');
        }
    } catch (error) {
        console.error('❌ Cannot connect to API:', error);
        showToast('Cannot connect to API. Please start the backend server.', 'error');
    }
}

// Utility Functions
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

function getEmotionColor(emotion) {
    const config = emotionConfig[emotion] || emotionConfig.neutral;
    return config.color;
}

function getEmotionIcon(emotion) {
    const config = emotionConfig[emotion] || emotionConfig.neutral;
    return config.icon;
}

// Initialize theme on load
loadTheme();
