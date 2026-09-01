/**
 * rppg-dsp.js
 * Implementation of Remote Photoplethysmography (rPPG) signal processing
 * ported and optimized from heartbeat-master (Philipp Rouast).
 */

export class RPPG_DSP {
  constructor(options = {}) {
    this.minSignalSeconds = options.minSignalSeconds || 5.0;
    this.maxSignalSeconds = options.maxSignalSeconds || 5.0;
    this.samplingFrequency = options.samplingFrequency || 1.0; // Update mean BPM every 1s
    this.lowBPM = options.lowBPM || 42;
    this.highBPM = options.highBPM || 240;

    this.reset();
  }

  reset() {
    // Buffers
    this.s = [];       // Raw RGB means: [{r, g, b}]
    this.t = [];       // Timestamps in milliseconds
    this.re = [];      // Rescan/jump flags (boolean)
    this.bpms = [];    // BPM samples since last sampling interval

    this.s_f = [];           // Filtered time-domain signal
    this.powerSpectrum = []; // Frequency spectrum magnitudes
    this.bandLowIdx = 0;
    this.bandHighIdx = 0;

    this.fps = 30.0;
    this.bpm = 0.0;
    this.meanBpm = 0.0;
    this.minBpm = 0.0;
    this.maxBpm = 0.0;
    this.lastSamplingTime = 0;
    this.isReady = false;
  }

  /**
   * Process a single video frame with extracted mean RGB values from face ROI.
   * @param {Object} meanRGB - { r: number, g: number, b: number }
   * @param {number} timestamp - Performance timestamp in ms
   * @param {boolean} rescanFlag - True if face was re-detected or tracking reset
   * @returns {Object} Current DSP status and metrics
   */
  processFrame(meanRGB, timestamp, rescanFlag = false) {
    this.s.push({ r: meanRGB.r, g: meanRGB.g, b: meanRGB.b });
    this.t.push(timestamp);
    this.re.push(rescanFlag);

    // Calculate dynamic FPS from timestamps buffer
    this.fps = this.calculateFPS();

    // Trim buffers to maxSignalSeconds
    const maxSamples = Math.round(this.fps * this.maxSignalSeconds);
    while (this.s.length > maxSamples && this.s.length > 2) {
      this.s.shift();
      this.t.shift();
      this.re.shift();
    }

    const minSamples = Math.round(this.fps * this.minSignalSeconds);
    if (this.s.length >= minSamples && minSamples >= 10) {
      this.isReady = true;

      // Extract and filter green channel signal (Algorithm 0 in heartbeat-master)
      this.extractSignal_g();

      // Estimate heart rate from power spectrum
      this.estimateHeartrate(timestamp);
    } else {
      this.isReady = false;
    }

    return {
      isReady: this.isReady,
      fps: this.fps,
      bpm: this.bpm,
      meanBpm: this.meanBpm,
      minBpm: this.minBpm,
      maxBpm: this.maxBpm,
      filteredSignal: this.s_f,
      powerSpectrum: this.powerSpectrum,
      bandLowIdx: this.bandLowIdx,
      bandHighIdx: this.bandHighIdx,
      bufferProgress: Math.min(1.0, this.s.length / Math.max(1, minSamples))
    };
  }

  /**
   * Calculate effective FPS from timestamp buffer
   */
  calculateFPS() {
    if (this.t.length < 2) return 30.0;
    const durationSec = (this.t[this.t.length - 1] - this.t[0]) / 1000.0;
    if (durationSec <= 0) return 30.0;
    const computedFps = (this.t.length - 1) / durationSec;
    return Math.max(5.0, Math.min(120.0, computedFps));
  }

  /**
   * Extract green channel signal, denoise, normalize, detrend, and smooth
   */
  extractSignal_g() {
    const N = this.s.length;
    const rawG = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      rawG[i] = this.s[i].g;
    }

    // 1. Denoise jumps caused by face rescans
    const denoised = this.denoise(rawG, this.re);

    // 2. Normalization (Z-score)
    const normalized = this.normalization(denoised);

    // 3. Smoothness priors detrending (High-pass equivalent, lambda = fps)
    const lambda = Math.max(5, Math.round(this.fps));
    const detrended = this.detrend(normalized, lambda);

    // 4. Moving average filter (Low-pass equivalent, 3 iterations)
    const kernelSize = Math.max(2, Math.floor(this.fps / 6));
    const smoothed = this.movingAverage(detrended, 3, kernelSize);

    this.s_f = Array.from(smoothed);
  }

  /**
   * Eliminate jump discontinuities when face tracking resets / rescans
   */
  denoise(signal, jumps) {
    const N = signal.length;
    const result = new Float64Array(signal);

    for (let i = 1; i < N; i++) {
      if (jumps[i]) {
        const diff = signal[i] - signal[i - 1];
        for (let j = i; j < N; j++) {
          result[j] -= diff;
        }
      }
    }
    return result;
  }

  /**
   * Subtract mean and divide by standard deviation
   */
  normalization(signal) {
    const N = signal.length;
    let sum = 0.0;
    for (let i = 0; i < N; i++) sum += signal[i];
    const mean = sum / N;

    let varSum = 0.0;
    for (let i = 0; i < N; i++) {
      const diff = signal[i] - mean;
      varSum += diff * diff;
    }
    const stdDev = Math.sqrt(varSum / N) || 1e-6;

    const out = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      out[i] = (signal[i] - mean) / stdDev;
    }
    return out;
  }

  /**
   * Detrending using Smoothness Priors approach (Tarvainen et al., 2002)
   * Solves: (I + lambda^2 * D2^T * D2) * z = s, then detrended = s - z
   * Solved efficiently via pentadiagonal banded linear solver in O(N).
   */
  detrend(signal, lambda) {
    const N = signal.length;
    if (N < 4) return new Float64Array(signal);

    const lambdaSq = lambda * lambda;

    // D2 has rows [1, -2, 1].
    // Q = lambda^2 * D2^T * D2 is pentadiagonal symmetric matrix.
    // Matrix A = I + Q has diagonals:
    // main diag: d[i]
    // sub/super diag 1: e[i]
    // sub/super diag 2: f[i]
    const d = new Float64Array(N);
    const e = new Float64Array(N - 1);
    const f = new Float64Array(N - 2);

    for (let i = 0; i < N; i++) {
      // Contributions to D2^T * D2 on main diagonal
      let c = 0;
      if (i >= 0 && i < N - 2) c += 1; // row i of D2
      if (i >= 1 && i < N - 1) c += 4; // row i-1 of D2 (-2)^2
      if (i >= 2 && i < N) c += 1;     // row i-2 of D2 (1)^2
      d[i] = 1.0 + lambdaSq * c;
    }

    for (let i = 0; i < N - 1; i++) {
      let c = 0;
      if (i < N - 2) c += -2; // row i: 1 * (-2)
      if (i >= 1 && i < N - 1) c += -2; // row i-1: (-2) * 1
      e[i] = lambdaSq * c;
    }

    for (let i = 0; i < N - 2; i++) {
      f[i] = lambdaSq * 1.0; // row i: 1 * 1
    }

    // Solve pentadiagonal system A * z = signal via Cholesky decomposition A = L * L^T
    const z = this.solvePentadiagonal(N, d, e, f, signal);

    const out = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      out[i] = signal[i] - z[i];
    }
    return out;
  }

  /**
   * Fast O(N) symmetric positive-definite pentadiagonal solver
   */
  solvePentadiagonal(n, d, e, f, rhs) {
    const ld = new Float64Array(n);
    const le = new Float64Array(n - 1);
    const lf = new Float64Array(n - 2);

    // Cholesky factorization: A = L * D_diag * L^T (or standard L * L^T)
    for (let i = 0; i < n; i++) {
      let val = d[i];
      if (i >= 1) val -= le[i - 1] * le[i - 1];
      if (i >= 2) val -= lf[i - 2] * lf[i - 2];
      ld[i] = Math.sqrt(Math.max(1e-12, val));

      if (i < n - 1) {
        let eval_ = e[i];
        if (i >= 1) eval_ -= le[i - 1] * lf[i - 1];
        le[i] = eval_ / ld[i];
      }

      if (i < n - 2) {
        lf[i] = f[i] / ld[i];
      }
    }

    // Forward substitution: L * y = rhs
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let val = rhs[i];
      if (i >= 1) val -= le[i - 1] * y[i - 1];
      if (i >= 2) val -= lf[i - 2] * y[i - 2];
      y[i] = val / ld[i];
    }

    // Backward substitution: L^T * z = y
    const z = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let val = y[i];
      if (i < n - 1) val -= le[i] * z[i + 1];
      if (i < n - 2) val -= lf[i] * z[i + 2];
      z[i] = val / ld[i];
    }

    return z;
  }

  /**
   * 1D moving average filter repeated n iterations
   */
  movingAverage(signal, iterations, windowSize) {
    let cur = new Float64Array(signal);
    const N = cur.length;
    const half = Math.floor(windowSize / 2);

    for (let it = 0; it < iterations; it++) {
      const next = new Float64Array(N);
      for (let i = 0; i < N; i++) {
        let sum = 0;
        let count = 0;
        for (let w = -half; w <= half; w++) {
          const idx = i + w;
          if (idx >= 0 && idx < N) {
            sum += cur[idx];
            count++;
          }
        }
        next[i] = sum / (count || 1);
      }
      cur = next;
    }
    return cur;
  }

  /**
   * Estimate heart rate via discrete Fourier transform / FFT power spectrum
   */
  estimateHeartrate(timestamp) {
    const N = this.s_f.length;
    if (N < 4) return;

    // Real DFT magnitude power spectrum
    const spectrum = this.computePowerSpectrum(this.s_f);
    this.powerSpectrum = spectrum;

    // Determine band spectrum indices matching heartbeat-master:
    // low = (int)(s.rows * LOW_BPM / SEC_PER_MIN / fps)
    // high = (int)(s.rows * HIGH_BPM / SEC_PER_MIN / fps) + 1
    const low = Math.max(0, Math.floor((N * this.lowBPM) / (60.0 * this.fps)));
    const high = Math.min(Math.floor(N / 2), Math.ceil((N * this.highBPM) / (60.0 * this.fps)) + 1);

    this.bandLowIdx = low;
    this.bandHighIdx = high;

    // Find peak in frequency band [low, high]
    let maxVal = -1;
    let peakIdx = low;

    for (let i = low; i <= high && i < spectrum.length; i++) {
      if (spectrum[i] > maxVal) {
        maxVal = spectrum[i];
        peakIdx = i;
      }
    }

    // Parabolic sub-bin interpolation for smooth, accurate peak estimation
    let refinedPeak = peakIdx;
    if (peakIdx > low && peakIdx < high && peakIdx < spectrum.length - 1) {
      const a = spectrum[peakIdx - 1];
      const b = spectrum[peakIdx];
      const c = spectrum[peakIdx + 1];
      const denom = a - 2 * b + c;
      if (Math.abs(denom) > 1e-6) {
        const delta = 0.5 * (a - c) / denom;
        refinedPeak = peakIdx + Math.max(-0.5, Math.min(0.5, delta));
      }
    }

    // Calculate instantaneous BPM
    this.bpm = (refinedPeak * this.fps / N) * 60.0;
    this.bpms.push(this.bpm);

    // Update running mean BPM periodically (every 1 / samplingFrequency sec)
    if ((timestamp - this.lastSamplingTime) >= (1000.0 / this.samplingFrequency) || this.lastSamplingTime === 0) {
      this.lastSamplingTime = timestamp;

      if (this.bpms.length > 0) {
        const sorted = [...this.bpms].sort((a, b) => a - b);
        let sum = 0;
        for (const val of sorted) sum += val;
        this.meanBpm = sum / sorted.length;
        this.minBpm = sorted[0];
        this.maxBpm = sorted[sorted.length - 1];

        this.bpms = [];
      }
    }
  }

  /**
   * Real DFT power spectrum computation
   */
  computePowerSpectrum(signal) {
    const N = signal.length;
    const numFreqs = Math.floor(N / 2) + 1;
    const spectrum = new Float64Array(numFreqs);

    for (let k = 0; k < numFreqs; k++) {
      let real = 0.0;
      let imag = 0.0;
      const angleStep = (2.0 * Math.PI * k) / N;

      for (let n = 0; n < N; n++) {
        const angle = angleStep * n;
        real += signal[n] * Math.cos(angle);
        imag -= signal[n] * Math.sin(angle);
      }
      spectrum[k] = Math.sqrt(real * real + imag * imag);
    }

    return Array.from(spectrum);
  }
}
