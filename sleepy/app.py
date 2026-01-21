import json
import os
import tempfile
from collections import OrderedDict

import matplotlib.pyplot as plt
import altair as alt
import h5py
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sleepfm.models.models import SetTransformer, SleepEventLSTMClassifier, DiagnosisFinetuneFullLSTMCOXPHWithDemo
from sleepfm.preprocessing.preprocessing import EDFToHDF5Converter

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

FOUNDATION_MODEL_PATH = "../sleepfm/checkpoints/model_base"
STAGING_MODEL_PATH = "../sleepfm/checkpoints/model_sleep_staging"
DIAGNOSIS_MODEL_PATH = "../sleepfm/checkpoints/model_diagnosis"

def load_foundation_model(path, device):
    """Loads the SleepFM Foundation Model (SetTransformer)."""
    config_path = os.path.join(path, "config.json")
    checkpoint_path = os.path.join(path, "best.pt")

    if not os.path.exists(config_path) or not os.path.exists(checkpoint_path):
        st.error(f"Model config {config_path} or checkpoint {checkpoint_path} not found. Please check the path.")
        return None, None

    with open(config_path, "r") as f:
        config = json.load(f)

    model = SetTransformer(
        in_channels=config["in_channels"],
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        pooling_head=config.get("pooling_head", 4),
        dropout=0.0,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Remove 'module.' prefix if trained with DataParallel
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model, config


def load_staging_model(path, device):
    """Loads the Sleep Staging Classifier."""
    config_path = os.path.join(path, "config.json")
    checkpoint_path = os.path.join(path, "best.pt")

    if not os.path.exists(config_path) or not os.path.exists(checkpoint_path):
        # st.error("Staging model config or checkpoint not found.")
        st.error(f"Model config {config_path} or checkpoint {checkpoint_path} not found. Please check the path.")
        
        return None

    with open(config_path, "r") as f:
        config = json.load(f)

    params = config["model_params"]
    model = SleepEventLSTMClassifier(**params)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Remove 'module.' prefix
    state_dict = checkpoint
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


def load_diagnosis_model(path, device):
    """Loads the Sleep Staging Classifier."""
    config_path = os.path.join(path, "config.json")
    checkpoint_path = os.path.join(path, "best.pt")

    if not os.path.exists(config_path) or not os.path.exists(checkpoint_path):
        # st.error("Staging model config or checkpoint not found.")
        st.error(f"Model config {config_path} or checkpoint {checkpoint_path} not found. Please check the path.")
        
        return None

    with open(config_path, "r") as f:
        config = json.load(f)

    params = config["model_params"]
    model = DiagnosisFinetuneFullLSTMCOXPHWithDemo(**params)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Remove 'module.' prefix
    state_dict = checkpoint
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


def extract_epochs(hdf5_path, epoch_sec=30, fs=128):
    """
    Simple data loader that extracts 30s epochs from HDF5.
    In a real scenario, this would use channel_groups.json to map specific channels.
    Here, we use a simple heuristic to find 1 channel per modality.
    """
    data_map = {"bas": None, "resp": None, "ekg": None, "emg": None}

    with h5py.File(hdf5_path, "r") as f:
        keys = list(f.keys())

        # Simple heuristic to map channels (Case insensitive)
        for k in keys:
            kl = k.lower()
            if "eeg" in kl and data_map["bas"] is None:
                data_map["bas"] = f[k][:]
            elif "resp" in kl or "flow" in kl and data_map["resp"] is None:
                data_map["resp"] = f[k][:]
            elif "ecg" in kl or "ekg" in kl and data_map["ekg"] is None:
                data_map["ekg"] = f[k][:]
            elif "emg" in kl and data_map["emg"] is None:
                data_map["emg"] = f[k][:]

        # Fallback: if missing, use zeros or first available channel
        first_signal = f[keys[0]][:]
        total_len = len(first_signal)

        for mod in data_map:
            if data_map[mod] is None:
                data_map[mod] = np.zeros(total_len)  # Pad missing modalities with zeros

        # Chunk into epochs
        samples_per_epoch = epoch_sec * fs
        num_epochs = total_len // samples_per_epoch

        batched_data = {k: [] for k in data_map}

        for i in range(num_epochs):
            start = i * samples_per_epoch
            end = start + samples_per_epoch
            for mod in data_map:
                batched_data[mod].append(data_map[mod][start:end])

        # Convert to numpy arrays (N, Samples)
        for mod in batched_data:
            batched_data[mod] = np.array(batched_data[mod])

    return batched_data, num_epochs


def get_psg_metadata(hdf5_path, epoch_data, num_epochs, fs=128, epoch_sec=30):
    """
    Extract metadata and statistics from the PSG recording.
    """
    metadata = {
        "sampling_rate": fs,
        "epoch_duration": epoch_sec,
        "num_epochs": num_epochs,
        "total_duration_sec": num_epochs * epoch_sec,
        "total_duration_min": num_epochs * epoch_sec / 60,
        "total_duration_hours": num_epochs * epoch_sec / 3600,
        "channels": {},
    }

    # Get channel names from HDF5
    with h5py.File(hdf5_path, "r") as f:
        metadata["original_channels"] = list(f.keys())
        metadata["num_channels"] = len(f.keys())

    # Compute per-channel statistics
    modality_names = {
        "bas": "EEG (Brain Activity)",
        "resp": "Respiration",
        "ekg": "EKG (Heart)",
        "emg": "EMG (Muscle)",
    }

    for mod, name in modality_names.items():
        signal = epoch_data[mod].flatten()
        is_present = not np.allclose(signal, 0)
        metadata["channels"][mod] = {
            "name": name,
            "present": is_present,
            "mean": float(np.mean(signal)) if is_present else None,
            "std": float(np.std(signal)) if is_present else None,
            "min": float(np.min(signal)) if is_present else None,
            "max": float(np.max(signal)) if is_present else None,
            "samples": len(signal),
        }

    return metadata


def display_psg_info_section(metadata):
    """
    Display recording information and metadata.
    """
    st.header("📋 Recording Information")

    # Recording overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Duration", f"{metadata['total_duration_hours']:.2f} hrs")
    with col2:
        st.metric("Total Epochs", metadata["num_epochs"])
    with col3:
        st.metric("Sampling Rate", f"{metadata['sampling_rate']} Hz")
    with col4:
        st.metric("Epoch Length", f"{metadata['epoch_duration']} sec")

    # Channel availability
    st.subheader("Channel Availability")

    channel_status = []
    for mod, info in metadata["channels"].items():
        channel_status.append(
            {
                "Channel": info["name"],
                "Status": "✅ Detected" if info["present"] else "⚠️ Not Found",
                "Samples": f"{info['samples']:,}" if info["present"] else "N/A",
            }
        )

    df_channels = pd.DataFrame(channel_status)
    st.dataframe(df_channels, use_container_width=True, hide_index=True)

    # Original channel names from file
    with st.expander("📁 Original Channel Names from File"):
        st.write("The following channels were found in the uploaded EDF file:")
        for i, ch in enumerate(metadata["original_channels"], 1):
            st.text(f"  {i}. {ch}")


def display_psg_signals_section(epoch_data, num_epochs, fs=128, epoch_sec=30):
    """
    Display the PSG signals visualization section.
    """
    st.header("📊 PSG Signal Visualization")

    st.markdown(
        """
    Explore the polysomnography signals captured during the sleep recording. 
    Use the controls below to navigate through different epochs and view individual channel details.
    """
    )

    # Modality info
    modality_info = {
        "bas": {
            "name": "EEG (Brain Activity)",
            "color": "#1f77b4",
            "description": "Electrical activity of the brain",
        },
        "resp": {
            "name": "Respiration",
            "color": "#2ca02c",
            "description": "Breathing patterns and airflow",
        },
        "ekg": {
            "name": "EKG (Heart)",
            "color": "#d62728",
            "description": "Cardiac electrical activity",
        },
        "emg": {
            "name": "EMG (Muscle)",
            "color": "#9467bd",
            "description": "Muscle tone and movement",
        },
    }

    # Navigation controls
    st.subheader("🎛️ Navigation Controls")
    col1, col2, col3 = st.columns(3)

    with col1:
        start_epoch = st.number_input(
            "Start Epoch",
            min_value=0,
            max_value=max(0, num_epochs - 1),
            value=0,
            help="Select the starting epoch to visualize",
            key="psg_start_epoch",
        )
    with col2:
        num_display_epochs = st.slider(
            "Epochs to Display",
            min_value=1,
            max_value=min(20, num_epochs),
            value=min(5, num_epochs),
            help="Number of consecutive epochs to display",
            key="psg_num_epochs",
        )
    with col3:
        selected_channels = st.multiselect(
            "Channels",
            options=list(modality_info.keys()),
            default=list(modality_info.keys()),
            format_func=lambda x: modality_info[x]["name"],
            key="psg_channels",
        )

    if not selected_channels:
        st.warning("Please select at least one channel to visualize.")
        return

    end_epoch = min(start_epoch + num_display_epochs, num_epochs)

    # Time calculations
    samples_per_epoch = epoch_sec * fs
    total_samples = (end_epoch - start_epoch) * samples_per_epoch
    time_axis = np.arange(total_samples) / fs + (start_epoch * epoch_sec)

    # Time range display
    start_time_min = start_epoch * epoch_sec / 60
    end_time_min = end_epoch * epoch_sec / 60
    st.caption(
        f"Displaying: Epoch {start_epoch} to {end_epoch - 1} | Time: {start_time_min:.1f} min to {end_time_min:.1f} min"
    )

    # Signal visualization tabs
    tab_overview, tab_detailed, tab_spectrum = st.tabs(
        ["📈 Stacked View", "🔍 Detailed View", "📉 Signal Statistics"]
    )

    with tab_overview:
        st.markdown(
            "**Multi-channel stacked view** - All selected channels displayed together with normalized amplitudes."
        )

        # Combined stacked plot
        combined_data = []
        for mod in selected_channels:
            signal = epoch_data[mod][start_epoch:end_epoch].flatten()
            # Normalize for display
            signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            # Downsample for performance if needed
            step = max(1, len(signal_norm) // 5000)
            for t, v in zip(time_axis[::step], signal_norm[::step]):
                combined_data.append(
                    {
                        "Time (s)": t,
                        "Amplitude": v,
                        "Channel": modality_info[mod]["name"],
                    }
                )

        df_combined = pd.DataFrame(combined_data)

        chart = (
            alt.Chart(df_combined)
            .mark_line(strokeWidth=0.5)
            .encode(
                x=alt.X("Time (s):Q", title="Time (seconds)"),
                y=alt.Y("Amplitude:Q", title="Normalized Amplitude"),
                color=alt.Color("Channel:N", legend=alt.Legend(orient="bottom")),
                row=alt.Row(
                    "Channel:N",
                    header=alt.Header(labelAngle=0, labelAlign="left"),
                    sort=[modality_info[m]["name"] for m in selected_channels],
                ),
            )
            .properties(width=700, height=80)
            .resolve_scale(y="independent")
        )

        st.altair_chart(chart, use_container_width=True)

    with tab_detailed:
        st.markdown(
            "**Individual channel analysis** - Explore each channel with raw amplitude values and statistics."
        )

        for mod in selected_channels:
            signal = epoch_data[mod][start_epoch:end_epoch].flatten()
            is_present = not np.allclose(signal, 0)

            with st.expander(
                f"{'🟢' if is_present else '🔴'} {modality_info[mod]['name']}",
                expanded=(mod == selected_channels[0]),
            ):
                st.caption(modality_info[mod]["description"])

                if not is_present:
                    st.warning(
                        "This channel was not detected in the recording. Displaying zeros."
                    )

                # Statistics row
                col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
                with col_s1:
                    st.metric("Mean", f"{np.mean(signal):.3f}")
                with col_s2:
                    st.metric("Std Dev", f"{np.std(signal):.3f}")
                with col_s3:
                    st.metric("Min", f"{np.min(signal):.3f}")
                with col_s4:
                    st.metric("Max", f"{np.max(signal):.3f}")
                with col_s5:
                    st.metric("Range", f"{np.ptp(signal):.3f}")

                # Signal plot (downsampled for performance)
                step = max(1, len(signal) // 3000)
                df_signal = pd.DataFrame(
                    {
                        "Time (s)": time_axis[::step],
                        "Amplitude (μV)": signal[::step],
                    }
                )

                chart = (
                    alt.Chart(df_signal)
                    .mark_line(color=modality_info[mod]["color"], strokeWidth=0.5)
                    .encode(
                        x=alt.X("Time (s):Q", title="Time (seconds)"),
                        y=alt.Y("Amplitude (μV):Q", title="Amplitude (μV)"),
                    )
                    .properties(height=150)
                    .interactive()
                )

                st.altair_chart(chart, use_container_width=True)

    with tab_spectrum:
        st.markdown(
            "**Signal statistics summary** - Overview of signal quality and characteristics across the entire recording."
        )

        # Full recording statistics
        stats_data = []
        for mod in modality_info:
            signal = epoch_data[mod].flatten()
            is_present = not np.allclose(signal, 0)

            stats_data.append(
                {
                    "Channel": modality_info[mod]["name"],
                    "Status": "Present" if is_present else "Missing",
                    "Mean": f"{np.mean(signal):.4f}" if is_present else "N/A",
                    "Std Dev": f"{np.std(signal):.4f}" if is_present else "N/A",
                    "Min": f"{np.min(signal):.4f}" if is_present else "N/A",
                    "Max": f"{np.max(signal):.4f}" if is_present else "N/A",
                    "Range": f"{np.ptp(signal):.4f}" if is_present else "N/A",
                }
            )

        df_stats = pd.DataFrame(stats_data)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)

        # Distribution visualization
        st.subheader("Amplitude Distribution")

        dist_channel = st.selectbox(
            "Select channel for distribution",
            options=[
                m
                for m in selected_channels
                if not np.allclose(epoch_data[m].flatten(), 0)
            ],
            format_func=lambda x: modality_info[x]["name"],
            key="dist_channel",
        )

        if dist_channel:
            signal = epoch_data[dist_channel].flatten()
            # Sample for histogram
            sample_size = min(50000, len(signal))
            sample_idx = np.random.choice(len(signal), sample_size, replace=False)
            df_hist = pd.DataFrame({"Amplitude": signal[sample_idx]})

            hist_chart = (
                alt.Chart(df_hist)
                .mark_bar(color=modality_info[dist_channel]["color"], opacity=0.7)
                .encode(
                    x=alt.X(
                        "Amplitude:Q", bin=alt.Bin(maxbins=50), title="Amplitude (μV)"
                    ),
                    y=alt.Y("count()", title="Frequency"),
                )
                .properties(height=200)
            )

            st.altair_chart(hist_chart, use_container_width=True)


# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------

st.set_page_config(page_title="SleepFM Auto-Hypnogram", layout="wide")

st.title("🌙 SleepFM: Automated Hypnogram Generator")
st.markdown(
    "Upload a raw PSG file to generate a sleep stage hypnogram using the SleepFM foundation model."
)

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    # Default paths based on notebook context
    base_model_path = st.text_input(
        "Foundation Model Path", "../sleepfm/checkpoints/model_base"
    )
    staging_model_path = st.text_input(
        "Staging Model Path", "../sleepfm/checkpoints/model_sleep_staging"
    )
    diagnostic_model_path = st.text_input(
        "Diagnostic Model Path", "../sleepfm/checkpoints/model_diagnosis"
    )

    device_opt = st.selectbox("Device", ["cuda", "cpu", "mps"])
    if device_opt == "cuda" and not torch.cuda.is_available():
        st.warning("CUDA not available, falling back to CPU")
        device_opt = "cpu"
    if device_opt == "mps" and not torch.mps.is_available():
        st.warning("MPS not available, falling back to CPU")
        device_opt = "cpu"

    device = torch.device(device_opt)

uploaded_file = st.file_uploader("Upload PSG File (.edf)", type=["edf"])

if uploaded_file is not None:
    st.info("File uploaded successfully. Processing...")

    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Save temp file
        temp_edf_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_edf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 2. Preprocessing (EDF -> HDF5)
        st.text("Converting to HDF5 (Resampling to 128Hz)...")
        hdf5_filename = os.path.splitext(uploaded_file.name)[0] + ".hdf5"
        temp_hdf5_path = os.path.join(temp_dir, hdf5_filename)

        # Initialize converter with 128Hz as per demo notebook requirements
        converter = EDFToHDF5Converter(
            root_dir=temp_dir, target_dir=temp_dir, resample_rate=128, num_files=1
        )

        try:
            # We call convert directly on the file
            converter.convert(temp_edf_path, temp_hdf5_path)
        except Exception as e:
            st.error(f"Error during conversion: {e}")
            st.stop()

        # Extract data from HDF5
        epoch_data, num_epochs = extract_epochs(temp_hdf5_path, fs=128)

        # Get metadata
        metadata = get_psg_metadata(temp_hdf5_path, epoch_data, num_epochs, fs=128)

        # ---------------------------------------------------------------------
        # Section 1: Recording Information
        # ---------------------------------------------------------------------
        display_psg_info_section(metadata)

        st.divider()

        # ---------------------------------------------------------------------
        # Section 2: PSG Signal Visualization
        # ---------------------------------------------------------------------
        display_psg_signals_section(epoch_data, num_epochs, fs=128)

        st.divider()

        # ---------------------------------------------------------------------
        # Section 3: Sleep Stage Analysis
        # ---------------------------------------------------------------------
        st.header("🧠 Sleep Stage Analysis")

        sleep_staging_tab, diagnostic_tab = st.tabs(["Sleep staging", "Diagnostic"])
        
        with sleep_staging_tab:

            # 3. Embedding Generation
            st.text("Loading Foundation Model & Generating Embeddings...")

            foundation_model, fm_config = load_foundation_model(base_model_path, device)

            if foundation_model:
                # Prepare tensors
                # Shape: (Batch, 1, Samples) - Model expects (B, C, T)
                t_bas = (
                    torch.tensor(epoch_data["bas"], dtype=torch.float)
                    .unsqueeze(1)
                    .to(device)
                )
                t_resp = (
                    torch.tensor(epoch_data["resp"], dtype=torch.float)
                    .unsqueeze(1)
                    .to(device)
                )
                t_ekg = (
                    torch.tensor(epoch_data["ekg"], dtype=torch.float)
                    .unsqueeze(1)
                    .to(device)
                )
                t_emg = (
                    torch.tensor(epoch_data["emg"], dtype=torch.float)
                    .unsqueeze(1)
                    .to(device)
                )

                # Create dummy masks (assuming all data is valid for this demo)
                # Mask shape logic from generate_embeddings.py seems to imply boolean mask
                # Here we create a mask of False (no padding)
                mask = torch.zeros((num_epochs, 1), dtype=torch.bool).to(device)

                batch_size = 16
                embeddings_list = []

                progress_bar = st.progress(0)

                with torch.no_grad():
                    for i in range(0, num_epochs, batch_size):
                        end = min(i + batch_size, num_epochs)

                        # Get batch
                        b_bas = t_bas[i:end]
                        b_resp = t_resp[i:end]
                        b_ekg = t_ekg[i:end]
                        b_emg = t_emg[i:end]
                        b_mask = mask[i:end]

                        # Forward pass for each modality
                        # Model returns (x, embedding). We want the embedding (index 1)
                        # Note: generate_embeddings.py uses index 1 for 5-min agg, index 0 for granular
                        # Staging usually uses the granular sequence or aggregated.
                        # Based on notebook: "embeddings_new = [e[0].unsqueeze(1) for e in embeddings]" for granular

                        emb_bas = foundation_model(b_bas, b_mask)[1]
                        emb_resp = foundation_model(b_resp, b_mask)[1]
                        emb_ekg = foundation_model(b_ekg, b_mask)[1]
                        emb_emg = foundation_model(b_emg, b_mask)[1]

                        # Stack modalities: (Batch, Seq, Embed) -> (Batch, Modalities*Seq, Embed) ??
                        # Or simply concatenate. The Staging model expects a specific input structure.
                        # In the notebook, it saves separate HDF5 datasets per modality.
                        # The staging dataset loader usually concatenates them.

                        # For simplicity in this app, we concatenate along the sequence dimension
                        # or feature dimension depending on how the staging model was trained.
                        # Assuming Staging Model takes (Batch, Modalities, Seq, Embed) or similar.
                        # Let's look at SleepEventLSTMClassifier forward: input x is (B, C, S, E).

                        batch_emb = torch.stack(
                            [emb_bas, emb_resp, emb_ekg, emb_emg], dim=1
                        )
                        embeddings_list.append(batch_emb)

                        progress_bar.progress(end / num_epochs)

                full_embeddings = torch.cat(
                    embeddings_list, dim=0
                )  # (Total_Epochs, 4, Seq, Embed)

                # 4. Staging Inference
                st.text("Predicting Sleep Stages...")
                staging_model = load_staging_model(staging_model_path, device)

                if staging_model:
                    predictions = []

                    # Create padded matrix mask for staging model
                    # Shape: (Batch, Modalities, Seq)
                    # Since we have no padding, it's all False (valid)
                    B, C, S, E = full_embeddings.shape
                    padded_matrix = torch.zeros((B, C, S), dtype=torch.bool).to(device)

                    with torch.no_grad():
                        # Process in batches
                        for i in range(0, B, batch_size):
                            end = min(i + batch_size, B)
                            batch_x = full_embeddings[i:end]
                            batch_pad = padded_matrix[i:end]

                            outputs, _ = staging_model(batch_x, batch_pad)
                            # outputs shape: (Batch, Seq, Num_Classes) -> We likely want the classification for the epoch
                            # Usually sleep staging is one label per 30s epoch.
                            # If the model returns sequence, we might average or take last.
                            # However, standard SleepFM staging usually outputs (Batch, Num_Classes) if configured for epoch classification
                            # OR (Batch, Seq, Num_Classes) if dense.
                            # Let's assume the model outputs (Batch, Num_Classes) based on the `fc` layer in `model.py`
                            # Wait, `model.py` LSTM returns (B, S, num_classes).
                            # We will take the mean over the sequence dimension S for the epoch label.

                            logits = outputs.mean(dim=1)
                            preds = torch.argmax(logits, dim=1).cpu().numpy()
                            predictions.extend(preds)

                    st.success("Analysis Complete!")

                    # Map predictions to labels
                    stage_map = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
                    mapped_stages = [stage_map.get(p, "Unknown") for p in predictions]

                    df_hypno = pd.DataFrame(
                        {"Epoch": range(len(mapped_stages)), "Stage": mapped_stages}
                    )

                    # Visualization
                    st.subheader("Patient Hypnogram")

                    # Custom sorting for Y-axis to make Hypnogram look standard (Wake at top)
                    stage_order = ["Wake", "REM", "N1", "N2", "N3"]

                    chart = (
                        alt.Chart(df_hypno)
                        .mark_point()
                        .encode(
                            x=alt.X("Epoch:Q", title="Epoch"),
                            y=alt.Y("Stage:O", sort=stage_order, title="Stage"),
                        )
                        .properties(height=300)
                    )

                    st.altair_chart(chart, use_container_width=True)

                    # Download
                    csv = df_hypno.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Hypnogram CSV",
                        csv,
                        "hypnogram.csv",
                        "text/csv",
                        key="download-csv",
                    )

        with diagnostic_tab:

            # 3. Embedding Generation
            st.text("Loading Foundation Model & Generating Embeddings...")

            foundation_model, fm_config = load_foundation_model(base_model_path, device)

            if foundation_model:
                # Prepare tensors
                # Shape: (Batch, 1, Samples) - Model expects (B, C, T)
                t_bas = (
                    torch.tensor(epoch_data["bas"], dtype=torch.float)
                    .unsqueeze(1)
                    .to(device)
                )
                t_resp = (
                    torch.tensor(epoch_data["resp"], dtype=torch.float)
                    .unsqueeze(1)
                    .to(device)
                )
                t_ekg = (
                    torch.tensor(epoch_data["ekg"], dtype=torch.float)
                    .unsqueeze(1)
                    .to(device)
                )
                t_emg = (
                    torch.tensor(epoch_data["emg"], dtype=torch.float)
                    .unsqueeze(1)
                    .to(device)
                )

                # Create dummy masks (assuming all data is valid for this demo)
                # Mask shape logic from generate_embeddings.py seems to imply boolean mask
                # Here we create a mask of False (no padding)
                mask = torch.zeros((num_epochs, 1), dtype=torch.bool).to(device)

                batch_size = 16
                embeddings_list = []

                progress_bar = st.progress(0)

                with torch.no_grad():
                    for i in range(0, num_epochs, batch_size):
                        end = min(i + batch_size, num_epochs)

                        # Get batch
                        b_bas = t_bas[i:end]
                        b_resp = t_resp[i:end]
                        b_ekg = t_ekg[i:end]
                        b_emg = t_emg[i:end]
                        b_mask = mask[i:end]

                        # Forward pass for each modality
                        # Model returns (x, embedding). We want the embedding (index 1)
                        # Note: generate_embeddings.py uses index 1 for 5-min agg, index 0 for granular
                        # Staging usually uses the granular sequence or aggregated.
                        # Based on notebook: "embeddings_new = [e[0].unsqueeze(1) for e in embeddings]" for granular

                        emb_bas = foundation_model(b_bas, b_mask)[1]
                        emb_resp = foundation_model(b_resp, b_mask)[1]
                        emb_ekg = foundation_model(b_ekg, b_mask)[1]
                        emb_emg = foundation_model(b_emg, b_mask)[1]

                        # Stack modalities: (Batch, Seq, Embed) -> (Batch, Modalities*Seq, Embed) ??
                        # Or simply concatenate. The Staging model expects a specific input structure.
                        # In the notebook, it saves separate HDF5 datasets per modality.
                        # The staging dataset loader usually concatenates them.

                        # For simplicity in this app, we concatenate along the sequence dimension
                        # or feature dimension depending on how the staging model was trained.
                        # Assuming Staging Model takes (Batch, Modalities, Seq, Embed) or similar.
                        # Let's look at SleepEventLSTMClassifier forward: input x is (B, C, S, E).

                        batch_emb = torch.stack(
                            [emb_bas, emb_resp, emb_ekg, emb_emg], dim=1
                        )
                        embeddings_list.append(batch_emb)

                        progress_bar.progress(end / num_epochs)

                full_embeddings = torch.cat(
                    embeddings_list, dim=0
                )  # (Total_Epochs, 4, Seq, Embed)

                # 4. Diagnosis Inference
                diagnosis_model = load_diagnosis_model(diagnostic_model_path, device)

                if diagnosis_model:
                    st.text("Producing diagnosis...")
                    predictions = []

                    # Create padded matrix mask for staging model
                    # Shape: (Batch, Modalities, Seq)
                    # Since we have no padding, it's all False (valid)
                    B, C, S, E = full_embeddings.shape
                    padded_matrix = torch.zeros((B, C, S), dtype=torch.bool).to(device)
                    
                    mean_age = 30 
                    std_age = 2
                    sex_info = torch.randint(0, 2, size=())
                    age_info = torch.normal(mean=mean_age, std=std_age, size=())
                    demo_info = torch.tensor([[sex_info, age_info]]*B).to(device)

                    preds = []
                    labels = []
                    with torch.no_grad():
                        # Process in batches
                        for i in range(0, B, batch_size):

                            end = min(i + batch_size, B)
                            batch_x = full_embeddings[i:end]
                            batch_pad = padded_matrix[i:end]
                            demo_batch = demo_info[i:end]

                            outputs = diagnosis_model(batch_x, batch_pad, demo_batch)

                            pred, label = torch.max(outputs, 1)

                            preds.extend(pred.tolist())
                            labels.extend(label.tolist())

            st.success("Analysis Complete!")
            
            mapping = pd.read_csv('../sleepfm/configs/label_mapping.csv')
            phenotypes = [mapping[mapping.label_idx == label].phenotype.iloc[0] for label in labels]

            df = pd.DataFrame({
                "x": np.arange(len(preds)),
                "height": preds,
                "phenotypes": phenotypes
            })

            bars = alt.Chart(df).mark_bar().encode(
                x=alt.X("x:O", title="Index"),
                y=alt.Y("height:Q", title="Height")
            )

            st.altair_chart(bars, use_container_width=True)
