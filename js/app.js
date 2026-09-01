/**
 * app.js
 * Main application orchestrator for Cek-Vital-Dari-Wajah rPPG monitor.
 */

import { FaceDetector } from './face-detector.js';
import { RPPG_DSP } from './rppg-dsp.js';
import { RPPG_Renderer } from './renderer.js';

class HeartbeatApp {
  constructor() {
    this.video = document.getElementById('webcam-video');
    this.canvas = document.getElementById('overlay-canvas');
    this.startBtn = document.getElementById('toggle-btn');
    this.btnIcon = document.getElementById('btn-icon');
    this.btnText = document.getElementById('btn-text');
    this.statusBadge = document.getElementById('status-badge');
    this.bpmDisplay = document.getElementById('bpm-metric');
    this.fpsDisplay = document.getElementById('fps-metric');
    this.deviceSelect = document.getElementById('camera-select');
    this.flipBtn = document.getElementById('flip-btn');
    this.mirrorBtn = document.getElementById('mirror-btn');
    this.placeholder = document.getElementById('video-placeholder');
    this.terminalLogs = document.getElementById('terminal-logs');

    this.detector = new FaceDetector({ modelPath: './models/version-RFB-320.onnx', confThreshold: 0.65 });
    this.dsp = new RPPG_DSP({ minSignalSeconds: 5.0, maxSignalSeconds: 5.0, samplingFrequency: 1.0 });
    this.renderer = new RPPG_Renderer(this.canvas);

    this.stream = null;
    this.isRunning = false;
    this.currentFacingMode = 'user'; // 'user' (front) or 'environment' (back)
    this.isMirrored = true;
    this.animationFrameId = null;
    this.isDetecting = false;
    this.lastDetectionTime = 0;
    this.detectionInterval = 100; // Face detector run interval in ms (tracking smoothly)
    this.currentDetection = null;

    // Standardized 30 FPS sampling controls
    this.targetFPS = 30.0;
    this.targetSampleInterval = 1000.0 / this.targetFPS; // 33.33ms
    this.lastSampleTime = 0;
    this.lastDspResult = {
      isReady: false,
      fps: 30.0,
      bpm: 0,
      meanBpm: 0,
      filteredSignal: [],
      powerSpectrum: [],
      bufferProgress: 0
    };

    // Offscreen canvas for sampling ROI pixel data
    this.roiCanvas = document.createElement('canvas');
    this.roiCtx = this.roiCanvas.getContext('2d', { willReadFrequently: true });

    this.init();
  }

  async init() {
    this.bindEvents();
    this.log('Initializing ONNX Runtime Web...', 'info');
    this.setStatus('Loading ONNX Model...', 'warning');

    try {
      await this.detector.init();
      this.log('Model loaded. Ready to start.', 'success');
      this.setStatus('Ready', 'ready');
      this.startBtn.disabled = false;
      await this.loadCameraDevices();
    } catch (err) {
      this.log(`Model init error: ${err.message}`, 'error');
      this.setStatus('Model Error', 'error');
    }
  }

  bindEvents() {
    this.startBtn.addEventListener('click', () => this.toggleStream());

    if (this.flipBtn) {
      this.flipBtn.addEventListener('click', () => this.flipCamera());
    }

    if (this.mirrorBtn) {
      this.mirrorBtn.addEventListener('click', () => {
        this.isMirrored = !this.isMirrored;
        this.video.classList.toggle('-scale-x-100', this.isMirrored);
        this.mirrorBtn.classList.toggle('text-emerald-400', this.isMirrored);
        this.mirrorBtn.classList.toggle('border-emerald-500/50', this.isMirrored);
      });
    }

    if (this.deviceSelect) {
      this.deviceSelect.addEventListener('change', () => {
        const val = this.deviceSelect.value;
        if (val === 'facing:environment') {
          this.currentFacingMode = 'environment';
          this.isMirrored = false;
        } else if (val === 'facing:user') {
          this.currentFacingMode = 'user';
          this.isMirrored = true;
        } else if (val.startsWith('device:')) {
          // If specific back device was chosen
          const label = this.deviceSelect.options[this.deviceSelect.selectedIndex]?.text.toLowerCase() || '';
          this.isMirrored = !label.includes('back') && !label.includes('rear') && !label.includes('environment');
        }
        this.video.classList.toggle('-scale-x-100', this.isMirrored);

        if (this.isRunning) {
          this.switchCamera();
        }
      });
    }

    window.addEventListener('resize', () => this.syncCanvasSize());
  }

  flipCamera() {
    if (this.currentFacingMode === 'user') {
      this.currentFacingMode = 'environment';
      this.isMirrored = false;
      if (this.deviceSelect) this.deviceSelect.value = 'facing:environment';
      this.log('Switched to Back Camera', 'info');
    } else {
      this.currentFacingMode = 'user';
      this.isMirrored = true;
      if (this.deviceSelect) this.deviceSelect.value = 'facing:user';
      this.log('Switched to Front Camera', 'info');
    }

    this.video.classList.toggle('-scale-x-100', this.isMirrored);

    if (this.isRunning) {
      this.switchCamera();
    }
  }

  async switchCamera() {
    if (!this.isRunning) return;
    this.setStatus('Switching Camera...', 'warning');

    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }

    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    this.video.srcObject = null;

    // Critical for iOS Safari: wait 250ms for camera daemon hardware lock release
    await new Promise(resolve => setTimeout(resolve, 250));

    const constraints = this.getMediaConstraints();
    try {
      this.stream = await this.acquireMediaStream(constraints);
      this.video.srcObject = this.stream;
      await this.video.play();

      this.dsp.reset();
      this.detector.resetTracking();
      this.lastSampleTime = 0;
      this.setStatus('Tracking Face', 'active');
      this.log(`Camera active (${this.currentFacingMode === 'environment' ? 'Back' : 'Front'}).`, 'success');
      this.syncCanvasSize();
      this.startLoop();
    } catch (err) {
      this.log(`Camera switch error: ${err.name || 'Error'} - ${err.message}`, 'error');
      this.setStatus('Camera Error', 'error');
    }
  }

  async loadCameraDevices() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(d => d.kind === 'videoinput');

      if (this.deviceSelect) {
        const currentVal = this.deviceSelect.value;
        this.deviceSelect.innerHTML = '';

        // Standard Front/Back facingMode options (essential for iOS Safari / iPadOS)
        const frontOpt = document.createElement('option');
        frontOpt.value = 'facing:user';
        frontOpt.text = 'Front Camera (Selfie)';
        this.deviceSelect.appendChild(frontOpt);

        const backOpt = document.createElement('option');
        backOpt.value = 'facing:environment';
        backOpt.text = 'Back Camera';
        this.deviceSelect.appendChild(backOpt);

        // Add specific hardware device entries if labels are accessible
        if (videoDevices.length > 1 && videoDevices.some(d => d.label)) {
          videoDevices.forEach((device, idx) => {
            const opt = document.createElement('option');
            opt.value = `device:${device.deviceId}`;
            opt.text = device.label || `Camera ${idx + 1}`;
            this.deviceSelect.appendChild(opt);
          });
        }

        // Restore selected value if valid
        if (currentVal && Array.from(this.deviceSelect.options).some(o => o.value === currentVal)) {
          this.deviceSelect.value = currentVal;
        } else {
          this.deviceSelect.value = this.currentFacingMode === 'environment' ? 'facing:environment' : 'facing:user';
        }
      }
    } catch (e) {
      console.warn('Camera enumeration error:', e);
    }
  }

  async toggleStream() {
    if (this.isRunning) {
      await this.stopStream();
    } else {
      await this.startStream();
    }
  }

  getMediaConstraints() {
    const selected = this.deviceSelect ? this.deviceSelect.value : '';

    if (selected.startsWith('device:')) {
      const deviceId = selected.replace('device:', '');
      return {
        audio: false,
        video: {
          deviceId: { exact: deviceId },
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 }
        }
      };
    }

    const isBack = selected === 'facing:environment' || this.currentFacingMode === 'environment';
    return {
      audio: false,
      video: {
        facingMode: { ideal: isBack ? 'environment' : 'user' },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 }
      }
    };
  }

  async acquireMediaStream(constraints) {
    try {
      return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (err) {
      console.warn('Primary camera constraints failed, trying fallback...', err);
      try {
        const isBack = this.currentFacingMode === 'environment';
        return await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: isBack ? 'environment' : 'user'
          }
        });
      } catch (err2) {
        console.warn('FacingMode fallback failed, trying basic video...', err2);
        return await navigator.mediaDevices.getUserMedia({ audio: false, video: true });
      }
    }
  }

  async startStream() {
    this.setStatus('Requesting Camera...', 'warning');
    this.log('Requesting webcam access...', 'info');

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const isHttp = window.location.protocol === 'http:' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1';
      const msg = isHttp
        ? 'iOS Safari disables camera on insecure HTTP IP. Use HTTPS (GitHub Pages or local HTTPS tunnel).'
        : 'Camera API (getUserMedia) not supported in this browser context.';
      this.log(msg, 'error');
      this.setStatus('Requires HTTPS', 'error');
      return;
    }

    // Explicit iOS video inline playback flags
    this.video.setAttribute('playsinline', 'true');
    this.video.setAttribute('webkit-playsinline', 'true');
    this.video.setAttribute('muted', 'true');
    this.video.muted = true;
    this.video.playsInline = true;

    const constraints = this.getMediaConstraints();

    try {
      this.stream = await this.acquireMediaStream(constraints);
      this.video.srcObject = this.stream;
      await this.video.play();

      this.isRunning = true;
      this.dsp.reset();
      this.detector.resetTracking();
      this.lastSampleTime = 0;

      if (this.placeholder) {
        this.placeholder.classList.add('hidden');
      }

      this.updateBtnState(true);
      this.setStatus('Tracking Face', 'active');
      this.log(`Camera active (${this.currentFacingMode === 'environment' ? 'Back' : 'Front'}). 30 FPS rPPG locked.`, 'success');

      // Refresh device labels once permissions are granted (iOS Safari compatibility)
      await this.loadCameraDevices();

      this.syncCanvasSize();
      this.startLoop();
    } catch (err) {
      this.log(`Camera error: ${err.name || 'Error'} - ${err.message}`, 'error');
      this.setStatus('Camera Denied', 'error');
    }
  }

  async stopStream() {
    this.isRunning = false;
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }

    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }

    this.video.srcObject = null;
    this.renderer.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    if (this.placeholder) {
      this.placeholder.classList.remove('hidden');
    }

    this.updateBtnState(false);
    this.setStatus('Stopped', 'idle');
    this.log('Session stopped.', 'info');
  }

  updateBtnState(running) {
    if (running) {
      this.startBtn.classList.replace('bg-emerald-600', 'bg-rose-600');
      this.startBtn.classList.replace('hover:bg-emerald-500', 'hover:bg-rose-500');
      this.btnText.textContent = 'Stop Measurement';
      this.btnIcon.innerHTML = `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />`;
    } else {
      this.startBtn.classList.replace('bg-rose-600', 'bg-emerald-600');
      this.startBtn.classList.replace('hover:bg-rose-500', 'hover:bg-emerald-500');
      this.btnText.textContent = 'Start Measurement';
      this.btnIcon.innerHTML = `<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />`;
    }
  }

  syncCanvasSize() {
    if (this.video && this.video.videoWidth > 0) {
      this.canvas.width = this.video.videoWidth;
      this.canvas.height = this.video.videoHeight;
    }
  }

  startLoop() {
    const processFrame = async (now) => {
      if (!this.isRunning) return;

      this.syncCanvasSize();

      // Trigger face detection periodically
      if (!this.isDetecting && (now - this.lastDetectionTime >= this.detectionInterval)) {
        this.isDetecting = true;
        this.lastDetectionTime = now;
        this.detector.detect(this.video).then(detection => {
          this.currentDetection = detection;
          this.isDetecting = false;
        }).catch(err => {
          this.isDetecting = false;
        });
      }

      // Standardize DSP sampling rate to 30 FPS across 60Hz, 120Hz ProMotion iPad, and high-refresh displays
      const sampleElapsed = now - this.lastSampleTime;
      if (sampleElapsed >= this.targetSampleInterval - 3) {
        this.lastSampleTime = now;

        if (this.currentDetection && this.currentDetection.roi) {
          const roi = this.currentDetection.roi;

          // Sample mean RGB in forehead ROI
          const meanRGB = this.sampleROIMean(this.video, roi);
          if (meanRGB) {
            this.lastDspResult = this.dsp.processFrame(meanRGB, now, false);

            if (this.lastDspResult.isReady && this.lastDspResult.meanBpm > 0) {
              this.bpmDisplay.textContent = this.lastDspResult.meanBpm.toFixed(1);
            } else {
              this.bpmDisplay.textContent = '--';
            }

            this.fpsDisplay.textContent = this.lastDspResult.fps.toFixed(1);
          }
        }
      }

      // Render canvas overlay smoothly on every visual frame
      this.renderer.render(this.video, this.currentDetection, this.lastDspResult, this.isMirrored);

      if ('requestVideoFrameCallback' in this.video) {
        this.video.requestVideoFrameCallback(processFrame);
      } else {
        this.animationFrameId = requestAnimationFrame(processFrame);
      }
    };

    if ('requestVideoFrameCallback' in this.video) {
      this.video.requestVideoFrameCallback(processFrame);
    } else {
      this.animationFrameId = requestAnimationFrame(processFrame);
    }
  }

  /**
   * Extract average R, G, B color values in the forehead ROI
   */
  sampleROIMean(video, roi) {
    const rw = Math.max(2, Math.round(roi.width));
    const rh = Math.max(2, Math.round(roi.height));
    const rx = Math.max(0, Math.min(video.videoWidth - rw, Math.round(roi.x)));
    const ry = Math.max(0, Math.min(video.videoHeight - rh, Math.round(roi.y)));

    this.roiCanvas.width = rw;
    this.roiCanvas.height = rh;

    this.roiCtx.drawImage(video, rx, ry, rw, rh, 0, 0, rw, rh);
    const imgData = this.roiCtx.getImageData(0, 0, rw, rh);
    const data = imgData.data;
    const len = data.length;

    let rSum = 0;
    let gSum = 0;
    let bSum = 0;
    const pixelCount = len / 4;

    for (let i = 0; i < len; i += 4) {
      rSum += data[i];
      gSum += data[i + 1];
      bSum += data[i + 2];
    }

    return {
      r: rSum / pixelCount,
      g: gSum / pixelCount,
      b: bSum / pixelCount
    };
  }

  setStatus(text, type = 'idle') {
    if (!this.statusBadge) return;
    this.statusBadge.textContent = text;
    const colorMap = {
      idle: 'bg-slate-800 text-slate-400 border-slate-700',
      warning: 'bg-amber-950/70 text-amber-400 border-amber-800/80',
      ready: 'bg-emerald-950/70 text-emerald-400 border-emerald-800/80',
      active: 'bg-indigo-950/70 text-indigo-400 border-indigo-800/80 animate-pulse',
      error: 'bg-rose-950/70 text-rose-400 border-rose-800/80'
    };
    this.statusBadge.className = `px-3 py-1 text-xs font-mono font-medium rounded-full border transition-all ${colorMap[type] || colorMap.idle}`;
  }

  log(msg, type = 'info') {
    if (!this.terminalLogs) return;
    const line = document.createElement('div');
    const time = new Date().toLocaleTimeString();
    const colors = {
      info: 'text-slate-400',
      success: 'text-emerald-400',
      error: 'text-rose-400',
      warning: 'text-amber-400'
    };
    line.className = `font-mono text-xs ${colors[type] || colors.info}`;
    line.textContent = `[${time}] ${msg}`;
    this.terminalLogs.appendChild(line);
    this.terminalLogs.scrollTop = this.terminalLogs.scrollHeight;
  }
}

// Auto-boot on DOM ready
window.addEventListener('DOMContentLoaded', () => {
  window.app = new HeartbeatApp();
});
