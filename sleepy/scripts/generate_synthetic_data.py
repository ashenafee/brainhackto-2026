import numpy as np
import pyedflib
from scipy import signal


def generate_pink_noise(n_samples):
    """Generate pink (1/f) noise using the Voss-McCartney algorithm."""
    # Use FFT method for pink noise
    white = np.fft.rfft(np.random.randn(n_samples))
    freqs = np.fft.rfftfreq(n_samples)
    # Avoid division by zero
    freqs[0] = 1
    pink = white / np.sqrt(freqs)
    pink[0] = 0  # Remove DC component
    return np.fft.irfft(pink, n=n_samples)


def generate_k_complex(fs):
    """Generate a K-complex waveform (sharp negative followed by positive deflection)."""
    duration = 0.5  # seconds
    n_points = int(duration * fs)
    t = np.linspace(0, duration, n_points)

    # Sharp negative wave followed by slower positive wave
    negative_wave = -80 * np.exp(-((t - 0.15) ** 2) / (2 * 0.03**2))
    positive_wave = 40 * np.exp(-((t - 0.35) ** 2) / (2 * 0.05**2))

    return negative_wave + positive_wave


def generate_realistic_qrs(fs):
    """Generate a realistic PQRST complex using sum of Gaussians."""
    duration = 0.6  # seconds for full beat
    n_points = int(duration * fs)
    t = np.linspace(0, duration, n_points)

    # P wave
    p_wave = 0.15 * np.exp(-((t - 0.1) ** 2) / (2 * 0.02**2))

    # Q wave (small negative)
    q_wave = -0.1 * np.exp(-((t - 0.2) ** 2) / (2 * 0.008**2))

    # R wave (tall positive)
    r_wave = 1.2 * np.exp(-((t - 0.22) ** 2) / (2 * 0.008**2))

    # S wave (small negative)
    s_wave = -0.2 * np.exp(-((t - 0.24) ** 2) / (2 * 0.01**2))

    # T wave
    t_wave = 0.3 * np.exp(-((t - 0.4) ** 2) / (2 * 0.04**2))

    return p_wave + q_wave + r_wave + s_wave + t_wave


def generate_synthetic_edf(filename="synthetic_sleep.edf", duration_sec=600, fs=256):
    """
    Generates a synthetic EDF file with EEG, EKG, EMG, and Respiration signals.

    Args:
        filename (str): Output filename.
        duration_sec (int): Duration of the recording in seconds.
        fs (int): Sampling frequency in Hz.
    """
    n_samples = duration_sec * fs
    time = np.linspace(0, duration_sec, n_samples)

    print(f"Generating {duration_sec} seconds of data at {fs}Hz...")

    # ---------------------------------------------------------
    # 1. EEG Generation (Simulating Wake -> N2 transition)
    # ---------------------------------------------------------
    # Background activity using pink noise (1/f) instead of white noise
    pink_noise = generate_pink_noise(n_samples)
    eeg_background = pink_noise * 15  # Scale to appropriate amplitude

    # Alpha waves (8-12 Hz) - Dominant in Wake (First 30% of data)
    # Time-varying frequency and amplitude
    alpha_freq = 10 + 0.5 * np.sin(2 * np.pi * 0.1 * time)  # Slight frequency variation
    alpha_amp = 20 + 5 * np.sin(2 * np.pi * 0.05 * time)  # Amplitude variation
    alpha_phase = np.cumsum(2 * np.pi * alpha_freq / fs)
    alpha_wave = alpha_amp * np.sin(alpha_phase)

    # Smooth envelope to fade alpha out (gradual transition)
    alpha_envelope = np.ones(n_samples)
    transition_start = int(n_samples * 0.25)
    transition_end = int(n_samples * 0.35)
    alpha_envelope[transition_start:transition_end] = np.linspace(
        1, 0, transition_end - transition_start
    )
    alpha_envelope[transition_end:] = 0
    eeg_alpha = alpha_wave * alpha_envelope

    # Theta waves (4-8 Hz) - Dominant in N1/N2 (Last 70% of data)
    # Time-varying frequency and amplitude
    theta_freq = 6 + 0.3 * np.sin(2 * np.pi * 0.08 * time)
    theta_amp = 15 + 3 * np.sin(2 * np.pi * 0.03 * time)
    theta_phase = np.cumsum(2 * np.pi * theta_freq / fs)
    theta_wave = theta_amp * np.sin(theta_phase)

    # Smooth envelope for theta onset
    theta_envelope = np.zeros(n_samples)
    theta_envelope[transition_start:transition_end] = np.linspace(
        0, 1, transition_end - transition_start
    )
    theta_envelope[transition_end:] = 1
    eeg_theta = theta_wave * theta_envelope

    # Sleep Spindles (12-15 Hz bursts) - Characteristic of N2
    # Randomized frequency, duration, and amplitude
    spindle_signal = np.zeros(n_samples)
    n_spindles = np.random.randint(5, 10)  # Random number of spindles

    for _ in range(n_spindles):
        spindle_freq = np.random.uniform(12, 15)  # Random frequency
        spindle_duration = np.random.uniform(0.5, 2.0)  # Random duration (0.5-2s)
        spindle_amp = np.random.uniform(20, 35)  # Random amplitude

        duration_samples = int(spindle_duration * fs)
        start = np.random.randint(
            int(n_samples * 0.35), n_samples - duration_samples - 1
        )

        # Gaussian window for spindle envelope
        window = signal.windows.gaussian(duration_samples, std=duration_samples / 5)
        spindle_t = np.linspace(0, spindle_duration, duration_samples)
        spindle_wave = (
            spindle_amp * np.sin(2 * np.pi * spindle_freq * spindle_t) * window
        )
        spindle_signal[start : start + duration_samples] += spindle_wave

    # K-complexes in N2 (sharp negative followed by positive deflection)
    k_complex_signal = np.zeros(n_samples)
    k_complex_template = generate_k_complex(fs)
    n_k_complexes = np.random.randint(3, 8)

    for _ in range(n_k_complexes):
        start = np.random.randint(
            int(n_samples * 0.4), n_samples - len(k_complex_template) - 1
        )
        # Vary amplitude slightly
        amp_scale = np.random.uniform(0.7, 1.3)
        k_complex_signal[start : start + len(k_complex_template)] += (
            k_complex_template * amp_scale
        )

    # Combine EEG components
    eeg_signal = (
        eeg_background + eeg_alpha + eeg_theta + spindle_signal + k_complex_signal
    )

    # Clip to realistic physical limits (uV)
    eeg_signal = np.clip(eeg_signal, -150, 150)

    # ---------------------------------------------------------
    # 2. EKG Generation (Simulating Heartbeat)
    # ---------------------------------------------------------
    # Use realistic PQRST template
    qrs_template = generate_realistic_qrs(fs)

    # Simulate ~70 BPM with respiratory sinus arrhythmia
    base_bpm = 70

    # Low-frequency heart rate variability (respiratory sinus arrhythmia ~0.15-0.4 Hz)
    # Plus very low frequency component (~0.04 Hz)
    hrv_signal = (
        4 * np.sin(2 * np.pi * 0.25 * time)  # RSA component
        + 2 * np.sin(2 * np.pi * 0.04 * time)  # VLF component
    )

    # Baseline wander (low frequency drift, ~0.05-0.5 Hz)
    baseline_wander = 0.1 * np.sin(2 * np.pi * 0.1 * time) + 0.05 * np.sin(
        2 * np.pi * 0.3 * time
    )

    # Muscle artifact (high frequency noise bursts)
    muscle_artifact = np.zeros(n_samples)
    n_artifacts = np.random.randint(3, 8)
    for _ in range(n_artifacts):
        start = np.random.randint(0, n_samples - fs)
        duration = np.random.randint(int(fs * 0.1), int(fs * 0.5))
        artifact = np.random.normal(0, 0.05, duration)
        b, a = signal.butter(4, [20, 100], btype="band", fs=fs)
        artifact = signal.lfilter(b, a, artifact)
        muscle_artifact[start : start + duration] += artifact * np.random.uniform(
            0.5, 1.5
        )

    ekg_signal = (
        baseline_wander + np.random.normal(0, 0.02, n_samples) + muscle_artifact
    )

    # Place beats with HRV
    current_time = 0
    sample_idx = 0
    while sample_idx < n_samples - len(qrs_template):
        # Get current BPM with HRV
        hrv_idx = min(sample_idx, n_samples - 1)
        current_bpm = base_bpm + hrv_signal[hrv_idx]
        beat_interval = 60 / current_bpm

        # Add beat
        ekg_signal[sample_idx : sample_idx + len(qrs_template)] += qrs_template

        # Next beat with slight randomness
        variation = np.random.uniform(-0.02, 0.02)  # Small random variation
        current_time += beat_interval + variation
        sample_idx = int(current_time * fs)

    # ---------------------------------------------------------
    # 3. EMG Generation (Muscle Tone)
    # ---------------------------------------------------------
    # Use Rayleigh distribution for amplitude (more realistic than Gaussian)
    emg_base = np.random.rayleigh(scale=1, size=n_samples)

    # High pass filter to remove low freq drift (EMG is usually > 10Hz)
    b, a = signal.butter(4, 10, btype="high", fs=fs)
    emg_filtered = signal.lfilter(b, a, emg_base - np.mean(emg_base))

    # Gradual amplitude decrease from Wake to Sleep (smooth transition)
    emg_envelope = np.ones(n_samples)
    wake_end = int(n_samples * 0.3)
    transition_length = int(n_samples * 0.15)

    # Exponential decay during transition
    emg_envelope[:wake_end] = 25
    emg_envelope[wake_end : wake_end + transition_length] = 25 * np.exp(
        -3 * np.linspace(0, 1, transition_length)
    )
    emg_envelope[wake_end + transition_length :] = 25 * np.exp(-3)  # ~1.24

    # Add random bursts of activity (especially in wake)
    burst_signal = np.zeros(n_samples)
    n_bursts_wake = np.random.randint(8, 15)
    n_bursts_sleep = np.random.randint(2, 5)

    # Wake bursts
    for _ in range(n_bursts_wake):
        start = np.random.randint(0, wake_end - int(fs * 0.3))
        duration = int(np.random.uniform(0.1, 0.3) * fs)
        burst = np.random.rayleigh(scale=2, size=duration)
        b_filt, a_filt = signal.butter(4, 10, btype="high", fs=fs)
        burst = signal.lfilter(b_filt, a_filt, burst - np.mean(burst))
        burst_signal[start : start + duration] += burst * np.random.uniform(10, 30)

    # Sleep bursts (less frequent, lower amplitude)
    for _ in range(n_bursts_sleep):
        start = np.random.randint(wake_end, n_samples - int(fs * 0.2))
        duration = int(np.random.uniform(0.05, 0.15) * fs)
        burst = np.random.rayleigh(scale=1, size=duration)
        b_filt, a_filt = signal.butter(4, 10, btype="high", fs=fs)
        burst = signal.lfilter(b_filt, a_filt, burst - np.mean(burst))
        burst_signal[start : start + duration] += burst * np.random.uniform(3, 8)

    emg_signal = emg_filtered * emg_envelope + burst_signal

    # ---------------------------------------------------------
    # 4. Respiration Generation (Breathing)
    # ---------------------------------------------------------
    # Base respiratory rate with variability (~12-18 breaths/min)
    base_resp_rate = 15 / 60  # breaths per second (0.25 Hz)

    # Slow rate and amplitude variability
    rate_variation = 0.05 * np.sin(2 * np.pi * 0.01 * time)  # Very slow rate changes
    amp_variation = 1 + 0.2 * np.sin(2 * np.pi * 0.02 * time)  # Amplitude changes

    # Asymmetric waveform (faster inspiration, slower expiration)
    # Using sawtooth-like function
    resp_signal = np.zeros(n_samples)
    resp_phase = np.zeros(n_samples)

    current_phase = 0
    for i in range(n_samples):
        current_rate = base_resp_rate + rate_variation[i]
        current_phase += 2 * np.pi * current_rate / fs
        resp_phase[i] = current_phase

    # Asymmetric sine: inspiration (0 to pi) faster than expiration (pi to 2pi)
    # Using modified sine with different rates for each half
    for i in range(n_samples):
        phase_mod = resp_phase[i] % (2 * np.pi)
        if phase_mod < np.pi:
            # Inspiration (steeper rise) - compress to 40% of cycle
            resp_signal[i] = np.sin(phase_mod * 1.25)
        else:
            # Expiration (slower fall) - expand to 60% of cycle
            resp_signal[i] = np.sin(np.pi + (phase_mod - np.pi) * 0.833)

    resp_signal = resp_signal * 10 * amp_variation

    # Add some natural irregularity
    resp_noise = np.random.normal(0, 0.3, n_samples)
    b, a = signal.butter(2, 0.5, btype="low", fs=fs)
    resp_drift = signal.lfilter(b, a, resp_noise)
    resp_signal += resp_drift

    # Optional: Insert apnea events (short periods of flatline)
    n_apneas = np.random.randint(0, 3)  # 0-2 apnea events
    for _ in range(n_apneas):
        apnea_duration = np.random.uniform(5, 15)  # 5-15 seconds
        apnea_samples = int(apnea_duration * fs)
        start = np.random.randint(int(n_samples * 0.4), n_samples - apnea_samples - 1)

        # Create smooth transition into and out of apnea
        fade_samples = int(fs * 1)  # 1 second fade
        fade_in = np.linspace(1, 0, fade_samples)
        fade_out = np.linspace(0, 1, fade_samples)

        # Apply apnea with smooth transitions
        resp_signal[start : start + fade_samples] *= fade_in
        resp_signal[start + fade_samples : start + apnea_samples - fade_samples] = 0
        resp_signal[start + apnea_samples - fade_samples : start + apnea_samples] *= (
            fade_out
        )

    # ---------------------------------------------------------
    # Write to EDF
    # ---------------------------------------------------------
    n_channels = 4
    channel_info = []
    data_list = []

    # Define Channel Headers
    # Note: Physical dims are important for visualization scaling

    # 1. EEG
    channel_info.append(
        {
            "label": "EEG Fpz-Cz",
            "dimension": "uV",
            "sample_frequency": fs,
            "physical_max": 200,
            "physical_min": -200,
            "digital_max": 32767,
            "digital_min": -32768,
            "transducer": "AgAgCl",
            "prefilter": "",
        }
    )
    data_list.append(eeg_signal)

    # 2. Respiration (Flow)
    channel_info.append(
        {
            "label": "Resp Oro-Nasal",
            "dimension": "uV",  # Often arbitrary or uV in EDF
            "sample_frequency": fs,
            "physical_max": 100,
            "physical_min": -100,
            "digital_max": 32767,
            "digital_min": -32768,
            "transducer": "Thermistor",
            "prefilter": "",
        }
    )
    data_list.append(resp_signal)

    # 3. EKG
    channel_info.append(
        {
            "label": "ECG",
            "dimension": "mV",
            "sample_frequency": fs,
            "physical_max": 5,
            "physical_min": -5,
            "digital_max": 32767,
            "digital_min": -32768,
            "transducer": "AgAgCl",
            "prefilter": "",
        }
    )
    data_list.append(ekg_signal)

    # 4. EMG
    channel_info.append(
        {
            "label": "EMG Submental",
            "dimension": "uV",
            "sample_frequency": fs,
            "physical_max": 100,
            "physical_min": -100,
            "digital_max": 32767,
            "digital_min": -32768,
            "transducer": "AgAgCl",
            "prefilter": "",
        }
    )
    data_list.append(emg_signal)

    # Write file
    try:
        f = pyedflib.EdfWriter(
            filename, n_channels, file_type=pyedflib.FILETYPE_EDFPLUS
        )
        f.setSignalHeaders(channel_info)
        f.writeSamples(data_list)
        f.close()
        print(f"Successfully created '{filename}'")
    except Exception as e:
        print(f"Error writing EDF: {e}")


if __name__ == "__main__":
    generate_synthetic_edf()
