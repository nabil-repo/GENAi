const serverUrl = document.getElementById('serverUrl');
const promptEl = document.getElementById('prompt');
const stepsEl = document.getElementById('steps');
const cfgEl = document.getElementById('cfg');
const seedEl = document.getElementById('seed');
const btnGenerate = document.getElementById('btnGenerate');
const btnClear = document.getElementById('btnClear');
const btnDownload = document.getElementById('btnDownload');
const enhancePromptEl = null; // Feature disabled
const useGanRefinerEl = document.getElementById('useGanRefiner');
const ganModelInfo = document.getElementById('ganModelInfo');
const btnRefreshGanInfo = document.getElementById('btnRefreshGanInfo');
const btnEnhancePrompt = null; // Feature disabled
const btnPromptVariations = null; // Feature disabled
const btnUseEnhanced = null; // Feature disabled
const btnDownloadOriginal = document.getElementById('btnDownloadOriginal');
const btnDownloadRefined = document.getElementById('btnDownloadRefined');
const btnDownloadControl = document.getElementById('btnDownloadControl');
const promptEnhancementResult = document.getElementById('promptEnhancementResult');
const enhancedPromptText = document.getElementById('enhancedPromptText');
const promptVariationsResult = document.getElementById('promptVariationsResult');
const variationsList = document.getElementById('variationsList');
const refinedContainer = document.getElementById('refinedContainer');
const refinedResult = document.getElementById('refinedResult');
const controlImageContainer = document.getElementById('controlImageContainer');
const controlImageResult = document.getElementById('controlImageResult');
const generationInfo = document.getElementById('generationInfo');
const pipelineInfo = document.getElementById('pipelineInfo');
const canvas = document.getElementById('sketch');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const result = document.getElementById('result');

// New sketch processing elements
const sketchUpload = document.getElementById('sketchUpload');
const sketchType = document.getElementById('sketchType');
const controlnetStrength = document.getElementById('controlnetStrength');
const controlnetStrengthValue = document.getElementById('controlnetStrengthValue');
const btnProcessSketch = document.getElementById('btnProcessSketch');
const btnClearSketch = document.getElementById('btnClearSketch');
const processedPreview = document.getElementById('processedPreview');
const processedSketch = document.getElementById('processedSketch');
const processingInfo = document.getElementById('processingInfo');

let uploadedSketchB64 = null;

// Model selector elements
const modelCategory = document.getElementById('modelCategory');

// Canvas drawing setup
let drawing = false;
let history = [];
let historyStep = -1;

const colorPicker = document.getElementById('colorPicker');
const brushSize = document.getElementById('brushSize');
const btnUndo = document.getElementById('btnUndo');
const btnRedo = document.getElementById('btnRedo');

ctx.lineWidth = 3;
ctx.strokeStyle = '#ffffff';
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.fillStyle = '#000000';
ctx.fillRect(0, 0, canvas.width, canvas.height);
saveState();

// Drawing event listeners
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseleave', stopDrawing);

// Touch support
canvas.addEventListener('touchstart', handleTouch);
canvas.addEventListener('touchmove', handleTouch);
canvas.addEventListener('touchend', stopDrawing);

function getMousePos(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY
  };
}

function startDrawing(e) {
  drawing = true;
  const pos = getMousePos(e);
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y);
}

function draw(e) {
  if (!drawing) return;
  const pos = getMousePos(e);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
}

function stopDrawing() {
  if (drawing) {
    drawing = false;
    saveState();
  }
}

function handleTouch(e) {
  e.preventDefault();
  const touch = e.touches[0];
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (touch.clientX - rect.left) * scaleX;
  const y = (touch.clientY - rect.top) * scaleY;
  
  if (e.type === 'touchstart') {
    drawing = true;
    ctx.beginPath();
    ctx.moveTo(x, y);
  } else if (e.type === 'touchmove' && drawing) {
    ctx.lineTo(x, y);
    ctx.stroke();
  }
}

function saveState() {
  historyStep++;
  if (historyStep < history.length) {
    history.length = historyStep;
  }
  history.push(canvas.toDataURL());
  updateUndoRedo();
}

function updateUndoRedo() {
  btnUndo.disabled = historyStep <= 0;
  btnRedo.disabled = historyStep >= history.length - 1;
}

// Color picker and brush size
colorPicker.addEventListener('change', (e) => {
  ctx.strokeStyle = e.target.value;
});

brushSize.addEventListener('input', (e) => {
  ctx.lineWidth = e.target.value;
});

// Undo/Redo functionality
btnUndo.addEventListener('click', () => {
  if (historyStep > 0) {
    historyStep--;
    restoreState();
  }
});

btnRedo.addEventListener('click', () => {
  if (historyStep < history.length - 1) {
    historyStep++;
    restoreState();
  }
});

function restoreState() {
  const img = new Image();
  img.onload = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
  };
  img.src = history[historyStep];
  updateUndoRedo();
}

btnClear.addEventListener('click', () => {
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  saveState();
});

// Download sketch button
const btnDownloadSketch = document.getElementById('btnDownloadSketch');
if (btnDownloadSketch) {
  btnDownloadSketch.addEventListener('click', () => {
    const link = document.createElement('a');
    link.download = 'sketch.png';
    link.href = canvas.toDataURL();
    link.click();
  });
}

// ControlNet strength slider
controlnetStrength.addEventListener('input', (e) => {
  controlnetStrengthValue.textContent = e.target.value;
});

// Sketch upload functionality
sketchUpload.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        // Clear canvas and draw uploaded image
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Scale image to fit canvas
        const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
        const scaledWidth = img.width * scale;
        const scaledHeight = img.height * scale;
        const x = (canvas.width - scaledWidth) / 2;
        const y = (canvas.height - scaledHeight) / 2;
        
        ctx.drawImage(img, x, y, scaledWidth, scaledHeight);
        
        // Store base64 version
        uploadedSketchB64 = canvasToB64();
        btnProcessSketch.style.display = 'inline-block';
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }
});

// Process sketch preview
btnProcessSketch.addEventListener('click', async () => {
  const sketchData = uploadedSketchB64 || canvasToB64();
  if (!sketchData) return;
  
  const payload = {
    sketch_image_b64: sketchData,
    sketch_type: sketchType.value,
    enhance_sketch: true
  };
  
  try {
    btnProcessSketch.disabled = true;
    btnProcessSketch.textContent = 'Processing...';
    
    const url = serverUrl.value.replace(/\/$/, '') + '/process-sketch';
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    const data = await res.json();
    processedSketch.src = 'data:image/png;base64,' + data.processed_control_b64;
    processingInfo.textContent = `Processing: ${data.sketch_type_used}`;
    processedPreview.style.display = 'block';
    
  } catch (error) {
    console.error('Error processing sketch:', error);
    alert('Error processing sketch: ' + error.message);
  } finally {
    btnProcessSketch.disabled = false;
    btnProcessSketch.textContent = 'Preview Processing';
  }
});

// Clear sketch functionality (if button exists)
const btnClearSketchAlt = document.getElementById('btnClearSketch');
if (btnClearSketchAlt) {
  btnClearSketchAlt.addEventListener('click', () => {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    uploadedSketchB64 = null;
    sketchUpload.value = '';
    if (btnProcessSketch) btnProcessSketch.style.display = 'none';
    processedPreview.style.display = 'none';
    saveState();
  });
}

function canvasToB64() {
  return canvas.toDataURL('image/png').split(',')[1];
}

console.log('Generate button element:', btnGenerate);

if (btnGenerate) {
  btnGenerate.addEventListener('click', async () => {
    console.log('Generate button clicked!');
    
    if(promptEl.value.trim() === '') {
      showNotification('Please enter a prompt.', 'warning');
      return;
    }
    
    const sketchData = uploadedSketchB64 || canvasToB64();
  const payload = {
    prompt: promptEl.value || 'a beautiful painting, watercolor, highly detailed',
    steps: parseInt(stepsEl.value || '30', 10),
    guidance_scale: parseFloat(cfgEl.value || '7.0'),
    seed: seedEl.value ? parseInt(seedEl.value, 10) : null,
    height: 512, width: 512,
    sketch_image_b64: sketchData,
    sketch_type: sketchType.value,
    controlnet_conditioning_scale: parseFloat(controlnetStrength.value),
    enhance_prompt: enhancePromptEl ? enhancePromptEl.checked : false,
    use_gan_refiner: useGanRefinerEl.checked,
    selected_model: modelCategory.value
  };
  
  const url = document.getElementById('serverUrl').value.replace(/\/$/, '') + '/generate';
  
  // Show result container with loading state
  const resultContainer = document.getElementById('resultContainer');
  resultContainer.style.display = 'block';
  result.style.display = 'none';
  document.querySelector('#resultContainer .image-loading').style.display = 'flex';
  
  result.src = '';
  btnGenerate.disabled = true;
  const originalText = btnGenerate.innerHTML;
  btnGenerate.innerHTML = '<span class="btn-icon">⏳</span><span>Generating...</span>';
  
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (!res.ok) {
      throw new Error(`Server error: ${res.status}`);
    }
    
    const data = await res.json();
    
    // Display original result
    result.onload = () => {
      document.querySelector('#resultContainer .image-loading').style.display = 'none';
      result.style.display = 'block';
    };
    result.src = 'data:image/png;base64,' + data.image_b64;
    if (btnDownloadOriginal) btnDownloadOriginal.style.display = 'inline-block';
    
    // Display refined result if available
    if (data.refined_image_b64 && refinedResult && refinedContainer && btnDownloadRefined) {
      refinedResult.src = 'data:image/png;base64,' + data.refined_image_b64;
      refinedContainer.style.display = 'block';
      btnDownloadRefined.style.display = 'inline-block';
    } else if (refinedContainer) {
      refinedContainer.style.display = 'none';
    }
    
    // Display processed control image if available
    if (data.processed_control_image_b64 && controlImageResult && controlImageContainer && btnDownloadControl) {
      controlImageResult.src = 'data:image/png;base64,' + data.processed_control_image_b64;
      controlImageContainer.style.display = 'block';
      btnDownloadControl.style.display = 'inline-block';
    } else if (controlImageContainer) {
      controlImageContainer.style.display = 'none';
    }
    
    // Show generation info
    const generationInfo = document.getElementById('generationInfo');
    let infoText = '<h4>✨ Generation Details</h4>';
    if (data.enhanced_prompt_used) {
      infoText += `<div>📝 <strong>Enhanced Prompt:</strong> "${data.enhanced_prompt_used}"</div>`;
    }
    if (data.used_gan_refiner) {
      infoText += '<div>🎨 <strong>GAN Refinement:</strong> Applied (4x upscaling)</div>';
    }
    if (data.used_sketch_processing) {
      infoText += `<div>🖊️ <strong>Sketch Processing:</strong> ${data.sketch_type_used} (strength: ${controlnetStrength.value})</div>`;
    }
    infoText += `<div>⚙️ <strong>Steps:</strong> ${payload.steps} | <strong>CFG:</strong> ${payload.guidance_scale}</div>`;
    
    if (generationInfo) {
      generationInfo.innerHTML = infoText;
      generationInfo.style.display = 'block';
    }
    
    showNotification('Image generated successfully!', 'success');
  } catch (e) {
    console.error('Generation error:', e);
    showNotification('Error generating image: ' + e.message, 'error');
    resultContainer.style.display = 'none';
  } finally {
    btnGenerate.disabled = false;
    btnGenerate.innerHTML = originalText;
  }
});
} else {
  console.error('Generate button not found! Check if ID="btnGenerate" exists in HTML');
}

btnDownload.addEventListener('click', () => {
  if (!result.src) return;
  const a = document.createElement('a');
  a.href = result.src;
  a.download = 'paintdiffusion.png';
  a.click();
});

btnDownloadOriginal.addEventListener('click', () => {
  if (!result.src) return;
  const a = document.createElement('a');
  a.href = result.src;
  a.download = 'paintdiffusion_original.png';
  a.click();
});

btnDownloadRefined.addEventListener('click', () => {
  if (!refinedResult.src) return;
  const a = document.createElement('a');
  a.href = refinedResult.src;
  a.download = 'paintdiffusion_refined.png';
  a.click();
});

btnDownloadControl.addEventListener('click', () => {
  if (!controlImageResult.src) return;
  const a = document.createElement('a');
  a.href = controlImageResult.src;
  a.download = 'paintdiffusion_control.png';
  a.click();
});

// Pipeline Info Function
async function loadPipelineInfo() {
  try {
    const url = serverUrl.value.replace(/\/$/, '') + '/pipeline-info';
    const res = await fetch(url);
    const data = await res.json();
    
    const info = data.pipeline_info;
    let infoHtml = '<h4>Pipeline Information</h4>';
    infoHtml += `<div><strong>Model:</strong> ${info.model}</div>`;
    infoHtml += `<div><strong>Device:</strong> ${info.device}</div>`;
    if (info.controlnet_model) {
      infoHtml += `<div><strong>ControlNet:</strong> ${info.controlnet_model}</div>`;
    }
    if (info.vae_model) {
      infoHtml += `<div><strong>Custom VAE:</strong> ${info.vae_model}</div>`;
    }
    
    if (info.capabilities) {
      infoHtml += '<div class="capability-list">';
      Object.entries(info.capabilities).forEach(([key, enabled]) => {
        const className = enabled ? 'capability-enabled' : 'capability-disabled';
        infoHtml += `<span class="capability-item ${className}">${key}</span>`;
      });
      infoHtml += '</div>';
    }
    
    if (pipelineInfo) {
      pipelineInfo.innerHTML = infoHtml;
      pipelineInfo.style.display = 'block';
    }
  } catch (error) {
    console.error('Error loading pipeline info:', error);
  }
}

// Load pipeline info on page load
window.addEventListener('load', loadPipelineInfo);

// Prompt Enhancement Functions (disabled)
if (btnEnhancePrompt) {
btnEnhancePrompt.addEventListener('click', async () => {
  if(promptEl.value.trim() === '') {
    alert('Please enter a prompt to enhance.');
    return;
  }
  
  const payload = {
    prompt: promptEl.value,
    max_length: 100,
    temperature: 0.8
  };
  
  const url = serverUrl.value.replace(/\/$/, '') + '/enhance-prompt';
  btnEnhancePrompt.disabled = true;
  btnEnhancePrompt.textContent = 'Enhancing...';
  
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    
    if (enhancedPromptText) {
      enhancedPromptText.innerHTML = `
        <div class="prompt-comparison">
          <div><strong>Original:</strong> ${data.original_prompt}</div>
          <div><strong>Enhanced:</strong> ${data.enhanced_prompt}</div>
        </div>
      `;
    }
    if (promptEnhancementResult) promptEnhancementResult.style.display = 'block';
    if (promptVariationsResult) promptVariationsResult.style.display = 'none';
    
    // Store enhanced prompt for use
    btnUseEnhanced.onclick = () => {
      promptEl.value = data.enhanced_prompt;
      promptEnhancementResult.style.display = 'none';
    };
    
  } catch (e) {
    alert('Error enhancing prompt: ' + e.message);
  } finally {
    btnEnhancePrompt.disabled = false;
    btnEnhancePrompt.textContent = 'Preview Enhancement';
  }
});
}

if (btnPromptVariations) {
btnPromptVariations.addEventListener('click', async () => {
  if(promptEl.value.trim() === '') {
    alert('Please enter a prompt to generate variations.');
    return;
  }
  
  const payload = {
    prompt: promptEl.value,
    num_variations: 3
  };
  
  const url = serverUrl.value.replace(/\/$/, '') + '/prompt-variations';
  btnPromptVariations.disabled = true;
  btnPromptVariations.textContent = 'Generating...';
  
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    
    if (variationsList) {
      variationsList.innerHTML = `
        <div><strong>Original:</strong> ${data.original_prompt}</div>
        <div><strong>Variations:</strong></div>
        ${data.variations.map((variation, index) => 
          `<div class="variation-item">
            <span>${index + 1}. ${variation}</span>
            <button onclick="useVariation('${variation.replace(/'/g, "\\'")}')">Use This</button>
          </div>`
        ).join('')}
      `;
    }
    if (promptVariationsResult) promptVariationsResult.style.display = 'block';
    if (promptEnhancementResult) promptEnhancementResult.style.display = 'none';
    
  } catch (e) {
    alert('Error generating variations: ' + e.message);
  } finally {
    btnPromptVariations.disabled = false;
    btnPromptVariations.textContent = 'Get Variations';
  }
});
}

function useVariation(variation) {
  promptEl.value = variation;
  promptVariationsResult.style.display = 'none';
}

// GAN Model Information Functions
async function fetchGanInfo() {
  try {
    const url = document.getElementById('serverUrl').value.replace(/\/$/, '') + '/gan-info';
    const response = await fetch(url);
    const data = await response.json();
    
    const info = data.gan_refiner;
    const status = data.status;
    
    let statusIcon = '[?]';
    let statusText = 'Unknown';
    let statusClass = 'status-unknown';
    
    if (status === 'ready') {
      statusIcon = '[OK]';
      statusText = 'Real-ESRGAN Ready';
      statusClass = 'status-ready';
    } else if (status === 'fallback_mode') {
      statusIcon = '[!]';
      statusText = 'Fallback Mode (PIL-based)';
      statusClass = 'status-fallback';
    } else if (status === 'disabled') {
      statusIcon = '[X]';
      statusText = 'Disabled';
      statusClass = 'status-disabled';
    }
    
    if (ganModelInfo) {
      ganModelInfo.innerHTML = `
        <div class="gan-status ${statusClass}">
          <div><strong>${statusIcon} ${statusText}</strong></div>
          ${info.enabled ? `
            <div>Model: ${info.model_name}</div>
            <div>Scale: ${info.scale_factor}x</div>
            <div>Mode: ${info.fallback_mode ? 'PIL Fallback' : 'Real-ESRGAN'}</div>
          ` : '<div>GAN refiner is disabled</div>'}
        </div>
      `;
    }
    
    // Enable/disable the checkbox based on status
    if (useGanRefinerEl) useGanRefinerEl.disabled = !info.enabled;
    
  } catch (error) {
    if (ganModelInfo) {
      ganModelInfo.innerHTML = `
        <div class="gan-status status-error">
          <div><strong>❌ Error checking GAN status</strong></div>
          <div>${error.message}</div>
        </div>
      `;
    }
  }
}

if (btnRefreshGanInfo) {
  btnRefreshGanInfo.addEventListener('click', fetchGanInfo);
}

// Load GAN info on page load
window.addEventListener('load', fetchGanInfo);

// Server status check
async function checkServerStatus() {
  const statusIndicator = document.getElementById('serverStatus');
  const statusDot = statusIndicator.querySelector('.status-dot');
  const statusText = statusIndicator.querySelector('.status-text');
  
  try {
    const url = document.getElementById('serverUrl').value.replace(/\/$/, '') + '/pipeline-info';
    const res = await fetch(url, { method: 'GET', timeout: 5000 });
    
    if (res.ok) {
      statusDot.style.background = 'var(--accent-success)';
      statusText.textContent = 'Server connected';
    } else {
      throw new Error('Server not responding');
    }
  } catch (error) {
    statusDot.style.background = 'var(--accent-error)';
    statusText.textContent = 'Server offline';
    statusDot.style.animation = 'none';
  }
}

// Check server status on load and periodically
window.addEventListener('load', checkServerStatus);
setInterval(checkServerStatus, 30000); // Check every 30 seconds

// Notification system
function showNotification(message, type = 'info') {
  const notification = document.createElement('div');
  notification.className = `notification notification-${type}`;
  notification.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 16px 24px;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    color: var(--text-primary);
    box-shadow: var(--shadow-lg);
    z-index: 1000;
    animation: slideInRight 0.3s ease-out;
    max-width: 400px;
  `;
  
  const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : type === 'warning' ? '⚠️' : 'ℹ️';
  notification.innerHTML = `<strong>${icon}</strong> ${message}`;
  
  document.body.appendChild(notification);
  
  setTimeout(() => {
    notification.style.animation = 'slideOutRight 0.3s ease-out';
    setTimeout(() => notification.remove(), 300);
  }, 3000);
}

// Add CSS animations for notifications
const style = document.createElement('style');
style.textContent = `
  @keyframes slideInRight {
    from {
      transform: translateX(400px);
      opacity: 0;
    }
    to {
      transform: translateX(0);
      opacity: 1;
    }
  }
  
  @keyframes slideOutRight {
    from {
      transform: translateX(0);
      opacity: 1;
    }
    to {
      transform: translateX(400px);
      opacity: 0;
    }
  }
`;
document.head.appendChild(style);
