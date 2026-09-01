/**
 * renderer.js
 * High-performance canvas rendering replicating the visual style of heartbeat-master GUI.
 */

export class RPPG_Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
  }

  /**
   * Clear and draw all visual elements
   * @param {HTMLVideoElement} videoElement
   * @param {Object|null} detection - { box, roi, corners }
   * @param {Object} dspData - Output from RPPG_DSP.processFrame
   */
  render(videoElement, detection, dspData) {
    const ctx = this.ctx;
    const width = this.canvas.width;
    const height = this.canvas.height;

    ctx.clearRect(0, 0, width, height);

    if (!detection || !detection.box) {
      // Draw subtle searching overlay if no face is detected
      this.drawSearchingState(width, height);
      return;
    }

    const { box, roi, corners } = detection;
    const { fps, meanBpm, isReady, filteredSignal, powerSpectrum, bandLowIdx, bandHighIdx, bufferProgress } = dspData;

    // 1. Draw Red Bounding Box around Face
    ctx.save();
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    ctx.strokeRect(box.x, box.y, box.width, box.height);

    // 2. Draw Green Forehead ROI
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 2;
    ctx.strokeRect(roi.x, roi.y, roi.width, roi.height);

    // 3. Draw Green Tracking Crosshairs
    if (corners && corners.length > 0) {
      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = 1.5;
      for (const pt of corners) {
        ctx.beginPath();
        ctx.moveTo(pt.x - 5, pt.y);
        ctx.lineTo(pt.x + 5, pt.y);
        ctx.moveTo(pt.x, pt.y - 5);
        ctx.lineTo(pt.x, pt.y + 5);
        ctx.stroke();
      }
    }

    // 4. Draw BPM Text (Red, top-left of face box)
    ctx.shadowColor = 'rgba(0, 0, 0, 0.85)';
    ctx.shadowBlur = 4;
    ctx.fillStyle = '#ef4444';
    ctx.font = 'bold 22px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';

    if (isReady && meanBpm > 0) {
      ctx.fillText(`${meanBpm.toFixed(1)} bpm`, box.x, Math.max(26, box.y - 12));
    } else {
      const pct = Math.round((bufferProgress || 0) * 100);
      ctx.fillText(`Buffering (${pct}%)`, box.x, Math.max(26, box.y - 12));
    }

    // 5. Draw FPS Text (Green, bottom-left of face box)
    ctx.fillStyle = '#22c55e';
    ctx.font = 'bold 18px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
    ctx.fillText(`${fps.toFixed(1)} fps`, box.x, Math.min(height - 15, box.y + box.height + 28));

    // 6. Draw Dual Red Waveforms (Time-domain signal & Power Spectrum)
    this.drawWaveforms(box, filteredSignal, powerSpectrum, bandLowIdx, bandHighIdx, width, height);

    ctx.restore();
  }

  /**
   * Draw dual red waveforms matching heartbeat-master layout
   */
  drawWaveforms(box, signal, spectrum, lowIdx, highIdx, canvasWidth, canvasHeight) {
    if (!signal || signal.length < 2 || !spectrum || spectrum.length < 2) return;

    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2.0;
    ctx.shadowColor = 'rgba(0, 0, 0, 0.9)';
    ctx.shadowBlur = 3;

    // Calculate display dimensions based on face box
    let displayHeight = box.height / 2.0;
    let displayWidth = box.width * 0.85;

    // Determine horizontal start: to the right of face box if room permits, otherwise anchored on right edge
    let drawAreaTlX = box.x + box.width + 25;
    if (drawAreaTlX + displayWidth > canvasWidth - 10) {
      drawAreaTlX = Math.max(10, canvasWidth - displayWidth - 20);
    }
    let drawAreaTlY = Math.max(20, Math.min(canvasHeight - displayHeight * 2 - 20, box.y));

    // A. Top Waveform: Time-domain filtered pulse signal (s_f)
    let minSig = Infinity;
    let maxSig = -Infinity;
    for (let i = 0; i < signal.length; i++) {
      if (signal[i] < minSig) minSig = signal[i];
      if (signal[i] > maxSig) maxSig = signal[i];
    }
    const sigRange = maxSig - minSig || 1e-6;
    const sigHeightMult = (displayHeight * 0.85) / sigRange;
    const sigWidthMult = displayWidth / (signal.length - 1);

    ctx.beginPath();
    for (let i = 0; i < signal.length; i++) {
      const px = drawAreaTlX + i * sigWidthMult;
      const py = drawAreaTlY + (maxSig - signal[i]) * sigHeightMult;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // B. Bottom Waveform: Frequency-domain power spectrum
    const totalBand = Math.max(1, highIdx - lowIdx);
    let minSpec = Infinity;
    let maxSpec = -Infinity;
    for (let i = lowIdx; i <= highIdx && i < spectrum.length; i++) {
      if (spectrum[i] < minSpec) minSpec = spectrum[i];
      if (spectrum[i] > maxSpec) maxSpec = spectrum[i];
    }
    const specRange = maxSpec - minSpec || 1e-6;
    const specHeightMult = (displayHeight * 0.85) / specRange;
    const specWidthMult = displayWidth / totalBand;

    const specBaseY = drawAreaTlY + displayHeight + 10;
    ctx.beginPath();
    for (let i = lowIdx; i <= highIdx && i < spectrum.length; i++) {
      const px = drawAreaTlX + (i - lowIdx) * specWidthMult;
      const py = specBaseY + (maxSpec - spectrum[i]) * specHeightMult;
      if (i === lowIdx) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    ctx.restore();
  }

  drawSearchingState(width, height) {
    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = 'rgba(239, 68, 68, 0.8)';
    ctx.font = '16px ui-monospace, monospace';
    ctx.shadowColor = 'rgba(0, 0, 0, 0.9)';
    ctx.shadowBlur = 4;
    ctx.fillText('Scanning for face...', 20, 35);
    ctx.restore();
  }
}
