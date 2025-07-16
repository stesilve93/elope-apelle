
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import torch
import torch.nn as nn

from pathlib import Path
from typing import Dict, Tuple, Optional

from sklearn.manifold import TSNE 

from elope.datasets import SequenceLoader, ElopeDataLoader, ElopeDataset
from elope.trainers import LunarTrainer
from elope.models.emmnetVelGru import MultiModalVelocityEstimator
from elope.utils import LOGGER, load_yaml


def run_realtime_prediction_and_extract_features(
    model: nn.Module,
    dataset_cfg: dict,
    seq_loader: SequenceLoader,
    sequence_id: str,
    dt: float = 0.05,
    start_offset_s: float = 0.5,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: # Added event_features_all
    """
    Runs "real-time" sequential prediction and extracts event features for analysis.
    """
    model.eval() # Set model to evaluation mode
    model.to(device)
    
    # Retrieve all the dataset config values 
    H = int(dataset_cfg["events"]["height"])
    W = int(dataset_cfg["events"]["width"])
    T = int(dataset_cfg["events"]["time_bins"])
    
    # Retrieve event parsing options
    event_encoder_method = dataset_cfg["events"]["encoder_method"]
    event_integration_window = float(dataset_cfg["events"]["integration_window"])
    
    imu_seq_len = int(dataset_cfg["imu_sequence_length"])

    # Load the sequence
    LOGGER.info(f"Loading test sequence: {sequence_id}.")
    seq_loader.load_sequence(sequence_id)
    
    # Recover initial trajectory time
    t_beg = seq_loader.timestamps_full[0]
    
    avg_imu_interval = np.mean(np.diff(seq_loader.timestamps_full[:imu_seq_len+5]))
    
    required_min_offset_s = imu_seq_len * avg_imu_interval
    if start_offset_s < required_min_offset_s:
        LOGGER.warning(
            f"Warning: start_offset_s ({start_offset_s:.4f}s) is less than the recommended "
            f"minimum for IMU sequence length ({required_min_offset_s:.4f}s). Adjusting."
        )
        start_offset_s = required_min_offset_s * 1.1

    # Compute the initial prediction time and index
    t_idx = np.searchsorted(seq_loader.timestamps_full, t_beg + start_offset_s, side='left')
    t0 = seq_loader.timestamps_full[t_idx]
    
    if t_idx >= len(seq_loader.timestamps_full):
        LOGGER.warning("No valid timestamps found for inference after initial offset.")
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # These store event embeddings and sequence of events for visualization
    event_features, event_sequence = [], []
    predictions, targets, times = [], [], []
    
    LOGGER.info(f"Starting predictions from timestamp {t0:.4f}s")
    while t_idx < len(seq_loader.timestamps_full):
        
        # Retrieve the current trajectory time
        tk = seq_loader.timestamps_full[t_idx]
        
        # Retrieve the current trajectory data
        data_point = seq_loader.get_data_at_time(
            tk, event_integration_window, event_encoder_method, imu_seq_len, H, W, T
        )

        if data_point is None:
            LOGGER.warning(f"Skipping prediction at {tk:.4f}s due to insufficient data.")
            t_idx += 1
            continue
        
        # visualize_event_data(data_point['events_tensor'].cpu(), current_time=t_current_s)

        event_t = data_point['events_tensor'].unsqueeze(0).to(device)
        imu_s   = data_point['imu_sequence'].unsqueeze(0).to(device)
        range_s = data_point['rangemeter_sequence'].unsqueeze(0).to(device)
        gt_pv   = data_point['ground_truth'].unsqueeze(0).to(device)
        
        # Store the event tensor and timestamp
        event_sequence.append((data_point['events_tensor'].cpu(), f"Time_{tk}")) 

        with torch.no_grad():
            # Run inference 
            outputs = model(event_t, imu_s, range_s)
            # Retrieve the prediction
            pred_k = outputs['prediction']
            # Extract event features
            event_k = outputs['event_features'] 

        # Store all the data
        predictions.append(pred_k.cpu().numpy().squeeze())
        targets.append(gt_pv.cpu().numpy().squeeze())
        times.append(tk)
        event_features.append(event_k.cpu().numpy().squeeze())

        # Compute the next time index
        t_idx = np.searchsorted(seq_loader.timestamps_full, tk + dt, side='left')
        
    LOGGER.info(
        f"Finished predictions for sequence {sequence_id}. "
        f"Total predictions: {len(predictions)}"
    )
    
    return (np.array(predictions), np.array(targets),
            np.array(times), np.array(event_features), event_sequence)


def run_realtime_prediction(
    model: nn.Module,
    dataset_cfg: dict,
    seq_loader: SequenceLoader,
    sequence_id: str,
    dt: float = 0.5, # How often to make a prediction (e.g., every 50ms)
    start_offset_s: float = 0.5, # Time to wait before first prediction (to fill LSTM buffers)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs "real-time" sequential prediction on a single test trajectory.

    Args:
        model (nn.Module): The trained MultiModalVelocityEstimator model.
        data_loader_instance (DataLoader): An instantiated DataLoader object.
        sequence_id (str): The ID of the test sequence (e.g., '0040').
        event_integration_window (float): Time window for events in microseconds.
        imu_seq_len (int): Sequence length for IMU and rangemeter data.
        H, W, T: Dimensions for the event tensor.
        prediction_interval_s (float): Interval in seconds at which to generate a prediction.
        start_offset_s (float): Initial offset in seconds to ensure enough data for seq_len.
                                This should be at least (imu_seq_len * trajectory_timestamp_interval).
        device (torch.device): The device to run inference on (cuda or cpu).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - predicted_states: Array of predicted (x,y,z,vx,vy,vz) at each prediction timestamp.
            - ground_truth_states: Array of corresponding ground truth (x,y,z,vx,vy,vz).
            - prediction_times_s: Array of timestamps at which predictions were made.
    """
    model.eval() # Set model to evaluation mode
    model.to(device)
    
    # Retrieve all the dataset config values 
    H = int(dataset_cfg["events"]["height"])
    W = int(dataset_cfg["events"]["width"])
    T = int(dataset_cfg["events"]["time_bins"])
    
    # Retrieve event parsing options
    event_encoder_method = dataset_cfg["events"]["encoder_method"]
    event_integration_window = float(dataset_cfg["events"]["integration_window"])
    
    imu_seq_len = int(dataset_cfg["imu_sequence_length"])

    # Load the sequence
    LOGGER.info(f"Loading test sequence: {sequence_id}.")
    seq_loader.load_sequence(sequence_id)
    
    # Recover initial trajectory time. This assumes trajectory timestamps are sorted 
    # and representative of the sequence
    t_beg = seq_loader.timestamps_full[0]
    

    # Determine the actual start time for predictions
    # This ensures that for the first prediction, we have enough history for the LSTMs.
    # The minimum required history time is roughly imu_seq_len * (avg_time_between_imu_samples)
    # A simple way to approximate avg_time_between_imu_samples:
    avg_imu_interval = np.mean(np.diff(seq_loader.timestamps_full[:imu_seq_len+5])) # Small sample for avg
    
    # Ensure start_offset_s covers at least the LSTM sequence length
    required_min_offset_s = imu_seq_len * avg_imu_interval
    if start_offset_s < required_min_offset_s:
        LOGGER.warning(
            f"Warning: start_offset_s ({start_offset_s:.4f}s) is less than the recommended "
            f"minimum for IMU sequence length ({required_min_offset_s:.4f}s). Adjusting."
        )
        start_offset_s = required_min_offset_s * 1.1 # Add a small buffer


    # Compute the initial prediction time and index
    t_idx = np.searchsorted(seq_loader.timestamps_full, t_beg + start_offset_s, side='left')
    t0 = seq_loader.timestamps_full[t_idx]
    
    # Ensure we don't go out of bounds
    if t_idx >= len(seq_loader.timestamps_full):
        LOGGER.warning("No valid timestamps found for inference after initial offset.")
        return np.array([]), np.array([]), np.array([])
    
    # Store the results
    predictions, targets, times = [], [], []
    
    LOGGER.info(f"Starting predictions from timestamp {t0:.4f}s")
    while t_idx < len(seq_loader.timestamps_full):
        
        # Retrieve the current trajectory time
        tk = seq_loader.timestamps_full[t_idx]
        
        # Retrieve the current trajectory data
        data_point = seq_loader.get_data_at_time(
            tk, event_integration_window, event_encoder_method, imu_seq_len, H, W, T
        )

        if data_point is None:
            LOGGER.warning(f"Skipping prediction at {tk:.4f}s due to insufficient data.")
            t_idx += 1
            continue
        
        # Unpack and move to device after adding batch dimension
        event_t = data_point['events_tensor'].unsqueeze(0).to(device) 
        imu_s   = data_point['imu_sequence'].unsqueeze(0).to(device)
        range_s = data_point['rangemeter_sequence'].unsqueeze(0).to(device)
        gt_pv   = data_point['ground_truth'].unsqueeze(0).to(device)
        
        with torch.no_grad(): 
            # Run inference and retrieve predictions
            outputs = model(event_t, imu_s, range_s)
            pred_k = outputs['prediction']

        # Store the results
        times.append(tk)
        predictions.append(pred_k.cpu().numpy().squeeze()) # Remove batch dim
        LOGGER.info(f"Prediction at {tk:.4f}s: {predictions[-1]}")
        
        targets.append(gt_pv.cpu().numpy().squeeze())
        LOGGER.info(f"Ground truth at {tk:.4f}s: {targets[-1]}")
        
        # Compute the next time index
        t_idx = np.searchsorted(seq_loader.timestamps_full, tk + dt, side='left')
        
    LOGGER.info(
        f"Finished predictions for sequence {sequence_id}. "
        f"Total predictions: {len(predictions)}")
    
    return np.array(predictions), np.array(targets), np.array(times)
    

def visualize_activation_maps(
    model: nn.Module,
    data_point: Dict[str, torch.Tensor],
    layer_name: str,
    feature_map_idx: int = 0, # Index of the feature map (channel) to plot
    device: torch.device = torch.device("cpu")
):
    """
    Visualizes the activation maps (feature maps) of a specified Conv3d layer
    for a given input event tensor.

    Args:
        model (nn.Module): The trained model.
        data_point (Dict[str, torch.Tensor]): A single data point from the DataLoader,
                                               containing 'events_tensor'.
        layer_name (str): The name of the attribute in the model that holds the Conv3d layer.
                          e.g., 'initial_block.0', 'res_block1.conv1'.
        num_maps_to_plot (int): How many feature maps (channels) to visualize from the layer.
        device (torch.device): The device the model is on.
    """
    
    # Get the specific layer using a forward hook
    activations = None
    def hook_fn(module, input, output):
        nonlocal activations # Allow modification of activations variable
        activations = output.cpu().numpy() # Store output and move to CPU

    # Register the hook to the specified layer
    try:
        parts = layer_name.split('.')
        target_layer = model.event_encoder # Start from event_encoder
        for part in parts:
            if part.isdigit():
                target_layer = target_layer[int(part)]
            else:
                target_layer = getattr(target_layer, part)
        
        # Ensure it's a convolutional layer whose output we want to capture
        if not isinstance(target_layer, (nn.Conv3d, nn.ReLU, nn.BatchNorm3d, nn.MaxPool3d)):
             print(f"Warning: Layer '{layer_name}' is not a typical processing layer. "
                   f"Hook might not capture expected activations. Found: {type(target_layer)}")
        
        hook = target_layer.register_forward_hook(hook_fn)

    except AttributeError:
        print(f"Error: Layer '{layer_name}' not found in the EventEncoder of the model.")
        return

    # Prepare input for inference
    event_t = data_point['events_tensor'].unsqueeze(0).to(device) # Add batch dim

    # Perform a forward pass to trigger the hook
    model.eval() # Ensure model is in eval mode
    with torch.no_grad():
        # We only need the forward pass through the relevant part,
        # but calling the full model is simplest for hook to trigger.
        _ = model(event_t, 
                  data_point['imu_sequence'].unsqueeze(0).to(device),
                  data_point['rangemeter_sequence'].unsqueeze(0).to(device))

    # Remove the hook after use to prevent memory leaks
    hook.remove()

    if activations is None:
        print(f"Could not retrieve activations for layer '{layer_name}'.")
        return

    # --- Prepare Input Event Data for Plotting ---
    # Input event tensor: (Batch, C, T, H, W) -> (1, 2, T, H, W)
    input_events_cpu = event_t[0].cpu().numpy() # Remove batch dim (2, T, H, W)
    input_t_bins = input_events_cpu.shape[1] # T from input
    input_h, input_w = input_events_cpu.shape[2], input_events_cpu.shape[3]

    # Option 1: Sum ON and OFF events for a combined visualization
    # We might want to abs() or just sum, depending on how we encoded polarity
    # Assuming channel 0 is ON, channel 1 is OFF (or vice versa),
    # a simple sum might cancel out, so let's try sum of absolute values or sum if positive events are 1 and negative are -1.
    # For now, let's sum them as is, assuming event accumulation results in meaningful values.
    # If events are +1 and -1, then summing works like event count difference.
    input_event_frames = input_events_cpu[0, :, :, :] + input_events_cpu[1, :, :, :] # Sum ON and OFF channels
    # input_event_frames will be (T, H, W)

    # --- Prepare Activation Map Data for Plotting ---
    # Activations shape: (Batch, Channels, Temporal_Depth_prime, Height_prime, Width_prime)
    num_output_channels, act_t_depth, act_h, act_w = activations[0].shape
    
    if feature_map_idx >= num_output_channels or feature_map_idx < 0:
        print(f"Error: feature_map_idx {feature_map_idx} is out of bounds for "
              f"layer '{layer_name}' which has {num_output_channels} channels.")
        return

    # Select the specific feature map (channel) to plot
    selected_activation_map = activations[0, feature_map_idx, :, :, :] # (Temporal_Depth_prime, Height_prime, Width_prime)

    # --- Plotting ---
    # Number of temporal slices for input and activation might be different
    num_temporal_slices = max(input_t_bins, act_t_depth)

    fig, axes = plt.subplots(2, num_temporal_slices, figsize=(2.5 * num_temporal_slices, 5)) # 2 rows: Input, Activation

    # Ensure axes is always a 2D array for consistent indexing
    if num_temporal_slices == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    elif axes.ndim == 1: # For subplots(2, N) where N=1, axes is 1D array of 2 elements
        axes = axes.reshape(2, -1)


    # Row 1: Input Integrated Event Frames
    for k in range(input_t_bins):
        ax = axes[0, k]
        frame_slice = input_event_frames[k, :, :]
        
        # Normalize for visualization. Use a divergent colormap for signed event values.
        max_abs_val = np.max(np.abs(frame_slice))
        if max_abs_val > 1e-6:
            im = ax.imshow(frame_slice, cmap='RdBu', vmin=-max_abs_val, vmax=max_abs_val)
        else:
            im = ax.imshow(frame_slice, cmap='gray', vmin=0, vmax=1) # If all zeros, show as black

        ax.axis('on')
        if k == 0:
            ax.set_title(f'Input (T={k})', fontsize=9)
            ax.text(-0.5, 0.5, 'Input Events', transform=ax.transAxes,
                    fontsize=10, va='center', ha='right')
        else:
            ax.set_title(f'T={k}', fontsize=9)
    
    # Fill remaining columns if activation maps have more temporal slices
    for k in range(input_t_bins, num_temporal_slices):
        axes[0, k].axis('on') # Hide unused axes

    # Row 2: Selected Activation Map Slices
    for k in range(act_t_depth):
        ax = axes[1, k]
        map_slice = selected_activation_map[k, :, :]
        
        # Normalize for visualization. Activations are typically non-negative after ReLU.
        min_val = np.min(map_slice)
        max_val = np.max(map_slice)
        if max_val - min_val > 1e-6:
            norm_map_slice = (map_slice - min_val) / (max_val - min_val)
        else:
            norm_map_slice = np.zeros_like(map_slice)

        im = ax.imshow(norm_map_slice, cmap='viridis', vmin=0, vmax=1)
        ax.axis('on')
        if k == 0:
            ax.set_title(f'Activation (T\'={k})', fontsize=9)
            ax.text(-0.5, 0.5, f'Map {feature_map_idx}', transform=ax.transAxes,
                    fontsize=10, va='center', ha='right')
        else:
            ax.set_title(f'T\'={k}', fontsize=9)

    # Fill remaining columns if input has more temporal slices
    for k in range(act_t_depth, num_temporal_slices):
        axes[1, k].axis('off') # Hide unused axes


    plt.suptitle(f'Input Events vs. Activation Map (Layer: {layer_name}, Map Index: {feature_map_idx})', fontsize=14, y=1.05)
    plt.tight_layout(rect=[0.02, 0.03, 1, 0.98])
    plt.savefig(f"./plots/input_vs_activation_{layer_name.replace('.', '_')}_map{feature_map_idx}_{TEST_SEQUENCE_ID}.png", dpi=300)
    
def visualize_event_data(events_tensor, current_time):
    """
    Visualizes the events_tensor for specified timestamps.

    Args:
        events_tensor (np.ndarray): The 4D events tensor polarity-features-h-w (2, 2, H, W).
        timestamp_indices (list): A list of indices for the timestamps to visualize.
    """
    feature_types = ["Normalized Timestamps", "Freshness Values"]
    polarities = ["Positive Events", "Negative Events"]
    
    # Extract H and W from the tensor
    _, H, W, _ = events_tensor.shape

    # For each feature type (timestamps, freshness)
    for feature_channel_idx in range(events_tensor.shape[1]):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6)) # One row, two columns for polarities
        fig.suptitle(f"{feature_types[feature_channel_idx]} at Timestamp {current_time}", fontsize=16)

        # For each polarity (positive, negative)
        for polarity_channel_idx in range(events_tensor.shape[0]):
            ax = axes[polarity_channel_idx]
            
            # Extract the 2D data slice
            data_slice = events_tensor[polarity_channel_idx, feature_channel_idx, :, :]
            
            # Use imshow to render the feature values as an image
            # 'viridis' is a good perceptually uniform colormap
            # vmin/vmax ensure consistent scaling (features are 0-1)
            im = ax.imshow(data_slice, cmap='Greys', origin='lower', vmin=0, vmax=1)
            
            ax.set_title(f"{polarities[polarity_channel_idx]}")
            ax.set_xlabel("Width")
            ax.set_ylabel("Height")
            
            # Add a colorbar for interpretation
            fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.savefig(f"./plots/event_tensor_{current_time}_{feature_types[feature_channel_idx]}_{TEST_SEQUENCE_ID}.png", dpi=300) 

def animate_event_data(events_data_sequence):
    """
    Creates an animation of event data over time.

    Args:
        events_data_sequence (list): A list of (events_tensor, current_time) tuples,
                                     where events_tensor is (2, 2, H, W).
    """
    feature_types = ["Normalized Timestamps", "Freshness Values"]
    polarities = ["Positive Events", "Negative Events"]

    # Initialize figures for each feature type
    fig_timestamps, axes_timestamps = plt.subplots(1, 2, figsize=(15, 6))
    fig_freshness, axes_freshness = plt.subplots(1, 2, figsize=(15, 6))

    ims_timestamps = [] # To store the image artists for timestamps figure
    ims_freshness = []  # To store the image artists for freshness figure
    # Set up initial plots for timestamps
    for polarity_channel_idx in range(2):
        ax = axes_timestamps[polarity_channel_idx]
        # Use a dummy initial image, it will be updated
        im = ax.imshow(np.zeros_like(events_data_sequence[0][0][polarity_channel_idx, 0, :, :]),
                        cmap='Greys', origin='lower', vmin=0, vmax=1)
        ims_timestamps.append(im)
        ax.set_title(f"{polarities[polarity_channel_idx]}")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        fig_timestamps.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    fig_timestamps.suptitle(f"{feature_types[0]}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Set up initial plots for freshness
    for polarity_channel_idx in range(2):
        ax = axes_freshness[polarity_channel_idx]
        im = ax.imshow(np.zeros_like(events_data_sequence[0][0][polarity_channel_idx, 1, :, :]),
                        cmap='Greys', origin='lower', vmin=0, vmax=1)
        ims_freshness.append(im)
        ax.set_title(f"{polarities[polarity_channel_idx]}")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        fig_freshness.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    fig_freshness.suptitle(f"{feature_types[1]}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


    def update(frame):
        events_tensor, current_time = events_data_sequence[frame]

        # Update timestamps figure
        fig_timestamps.suptitle(f"{feature_types[0]} at Timestamp {current_time}", fontsize=16)
        for polarity_channel_idx in range(2):
            data_slice = events_tensor[polarity_channel_idx, 0, :, :]
            ims_timestamps[polarity_channel_idx].set_array(data_slice)

        # Update freshness figure
        fig_freshness.suptitle(f"{feature_types[1]} at Timestamp {current_time}", fontsize=16)
        for polarity_channel_idx in range(2):
            data_slice = events_tensor[polarity_channel_idx, 1, :, :]
            ims_freshness[polarity_channel_idx].set_array(data_slice)

        return ims_timestamps + ims_freshness # Return all updated artists

    # Create animations
    num_frames = len(events_data_sequence)
    ani_timestamps = animation.FuncAnimation(fig_timestamps, update, frames=num_frames, blit=True, repeat=False)
    ani_freshness = animation.FuncAnimation(fig_freshness, update, frames=num_frames, blit=True, repeat=False)

    # Save the animations
    ani_timestamps.save(f'./plots/event_tensor_timestamps_animation.mp4', writer='ffmpeg', fps=5)
    ani_freshness.save(f'./plots/event_tensor_freshness_animation.mp4', writer='ffmpeg', fps=5)

    plt.close(fig_timestamps) # Close figures to prevent them from displaying
    plt.close(fig_freshness)

def animate_event_data_with_combined(events_data_sequence):
    """
    Creates an animation of event data over time, including combined visualization.

    Parameters
    ----------
    events_data_sequence : list of tuples
        A list of tuples containing the event tensors and corresponding timestamps.
    """
    feature_types = ["Normalized Timestamps", "Freshness Values"]
    polarities = ["Positive Events", "Negative Events"]

    # Initialize figures for each feature type (individual and combined)
    fig_timestamps, axes_timestamps = plt.subplots(1, 2, figsize=(15, 6))
    fig_freshness, axes_freshness = plt.subplots(1, 2, figsize=(15, 6))
    fig_combined_timestamps, ax_combined_timestamps = plt.subplots(figsize=(10, 8))
    fig_combined_freshness, ax_combined_freshness = plt.subplots(figsize=(10, 8))

    ims_timestamps = []
    ims_freshness = []
    im_combined_timestamps = None
    im_combined_freshness = None

    # Get H, W from the first tensor in the sequence
    _, _, H, W = events_data_sequence[0][0].shape

    # Set up initial plots for timestamps (individual)
    for polarity_channel_idx in range(2):
        ax = axes_timestamps[polarity_channel_idx]
        im = ax.imshow(np.zeros((H,W)), cmap='Greys', origin='lower', vmin=0, vmax=1)
        ims_timestamps.append(im)
        ax.set_title(f"{polarities[polarity_channel_idx]}")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        fig_timestamps.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    fig_timestamps.suptitle(f"{feature_types[0]}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Set up initial plots for freshness (individual)
    for polarity_channel_idx in range(2):
        ax = axes_freshness[polarity_channel_idx]
        im = ax.imshow(np.zeros((H,W)), cmap='Greys', origin='lower', vmin=0, vmax=1)
        ims_freshness.append(im)
        ax.set_title(f"{polarities[polarity_channel_idx]}")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        fig_freshness.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    fig_freshness.suptitle(f"{feature_types[1]}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Set up initial plot for Combined Timestamps
    im_combined_timestamps = ax_combined_timestamps.imshow(np.zeros((H,W)), cmap='RdBu_r', origin='lower', vmin=-1, vmax=1)
    ax_combined_timestamps.set_title(f"Positive (Blue) and Negative (Red) {feature_types[0]}")
    ax_combined_timestamps.set_xlabel("Width")
    ax_combined_timestamps.set_ylabel("Height")
    cbar_ts = fig_combined_timestamps.colorbar(im_combined_timestamps, ax=ax_combined_timestamps, orientation='vertical', fraction=0.046, pad=0.04)
    cbar_ts.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar_ts.set_ticklabels(['Strong Negative', 'Weak Negative', 'No Event', 'Weak Positive', 'Strong Positive'])
    fig_combined_timestamps.suptitle(f"Combined {feature_types[0]}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Set up initial plot for Combined Freshness
    im_combined_freshness = ax_combined_freshness.imshow(np.zeros((H,W)), cmap='RdBu_r', origin='lower', vmin=-1, vmax=1)
    ax_combined_freshness.set_title(f"Positive (Blue) and Negative (Red) {feature_types[1]}")
    ax_combined_freshness.set_xlabel("Width")
    ax_combined_freshness.set_ylabel("Height")
    cbar_fr = fig_combined_freshness.colorbar(im_combined_freshness, ax=ax_combined_freshness, orientation='vertical', fraction=0.046, pad=0.04)
    cbar_fr.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar_fr.set_ticklabels(['Strong Negative', 'Weak Negative', 'No Event', 'Weak Positive', 'Strong Positive'])
    fig_combined_freshness.suptitle(f"Combined {feature_types[1]}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


    def update(frame):
        """
        Update function to be called for each frame of the animation.

        Parameters
        ----------
        frame : int
            The current frame index.
        """
        events_tensor, current_time = events_data_sequence[frame]

        # Update timestamps figure (individual)
        fig_timestamps.suptitle(f"{feature_types[0]} at Timestamp {current_time}", fontsize=16)
        for polarity_channel_idx in range(2):
            data_slice = events_tensor[polarity_channel_idx, 0, :, :]
            ims_timestamps[polarity_channel_idx].set_array(data_slice)

        # Update freshness figure (individual)
        fig_freshness.suptitle(f"{feature_types[1]} at Timestamp {current_time}", fontsize=16)
        for polarity_channel_idx in range(2):
            data_slice = events_tensor[polarity_channel_idx, 1, :, :]
            ims_freshness[polarity_channel_idx].set_array(data_slice)

        # Update Combined Timestamps
        positive_ts = events_tensor[0, 0, :, :]
        negative_ts = events_tensor[1, 0, :, :]
        combined_ts_map = np.zeros_like(positive_ts)
        combined_ts_map[positive_ts > 0] = positive_ts[positive_ts > 0]
        combined_ts_map[negative_ts > 0] = -negative_ts[negative_ts > 0]
        im_combined_timestamps.set_array(combined_ts_map)
        fig_combined_timestamps.suptitle(f"Combined {feature_types[0]} at Timestamp {current_time}", fontsize=16)

        # Update Combined Freshness
        positive_fr = events_tensor[0, 1, :, :]
        negative_fr = events_tensor[1, 1, :, :]
        combined_fr_map = np.zeros_like(positive_fr)
        combined_fr_map[positive_fr > 0] = positive_fr[positive_fr > 0]
        combined_fr_map[negative_fr > 0] = -negative_fr[negative_fr > 0]
        im_combined_freshness.set_array(combined_fr_map)
        fig_combined_freshness.suptitle(f"Combined {feature_types[1]} at Timestamp {current_time}", fontsize=16)

        return ims_timestamps + ims_freshness + [im_combined_timestamps, im_combined_freshness]


    # Create animations
    num_frames = len(events_data_sequence)
    ani_timestamps = animation.FuncAnimation(fig_timestamps, update, frames=num_frames, blit=True, repeat=False)
    ani_freshness = animation.FuncAnimation(fig_freshness, update, frames=num_frames, blit=True, repeat=False)
    ani_combined_timestamps = animation.FuncAnimation(fig_combined_timestamps, update, frames=num_frames, blit=True, repeat=False)
    ani_combined_freshness = animation.FuncAnimation(fig_combined_freshness, update, frames=num_frames, blit=True, repeat=False)


    # Save the animations
    # Use a consistent writer for all
    writer = animation.FFMpegWriter(fps=5) 
    ani_timestamps.save(f'./plots/event_tensor_timestamps_animation.mp4', writer=writer)
    ani_freshness.save(f'./plots/event_tensor_freshness_animation.mp4', writer=writer)
    ani_combined_timestamps.save(f'./plots/event_tensor_combined_timestamps_animation.mp4', writer=writer)
    ani_combined_freshness.save(f'./plots/event_tensor_combined_freshness_animation.mp4', writer=writer)


    plt.close(fig_timestamps)
    plt.close(fig_freshness)
    plt.close(fig_combined_timestamps)
    plt.close(fig_combined_freshness)

# --- Main execution block for testing ---
if __name__ == "__main__":
    
    # Use physics aware IMU encoder
    USE_PHYSICS_AWARE = False 
    
    # Path to the folder from which to retrieve the weights 
    WEIGHTS_PATH = Path("weights") 
    
    # Name of the file in which the weights are stored
    WEIGHTS_NAME = "elope-emmnet-v1_20250715_225530.pth"
    
    # Path to the yaml file containing the dataset settings
    DATASET_CONFIG = "cfg/v1-rnd-cfg.yml"
    
    # Path in which the sequence data is stored
    DATAPATH = Path("elope_data") / "train"
    
    # True if the network should only provide the velocity as output
    VELOCITY_ONLY = True
        
    # A test sequence not used in training (e.g., the first test trajectory)
    TEST_SEQUENCE_ID = '0010' 
    
    # Set to True if we want to extract event features
    EXTRACT_INTERMEDIATE_FEATURES = False 
    
    PREDICTION_INTERVAL = 0.1
    
    # Device configuration 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    LOGGER.info(f"Using device: {device}")
    
    # Create the sequence loader 
    seq_loader = SequenceLoader(datapath=DATAPATH)
    
    # Create the model 
    model = MultiModalVelocityEstimator.create_model(
        use_attention=True, device=device, use_physics_aware=USE_PHYSICS_AWARE
    )
    
    # Load the model weights
    weights_fullpath = WEIGHTS_PATH / WEIGHTS_NAME
    if weights_fullpath.exists(): 
        LOGGER.info(f"Loading weights from: {weights_fullpath}")
        data = torch.load(str(weights_fullpath), map_location=device) 
        model.load_state_dict(data, strict=False)
        
    else: 
        raise ValueError(f"Weights file {weights_fullpath} does not exist.")

    
    # Retrieve the settings from the YAML config 
    dataset_cfg = load_yaml(DATASET_CONFIG)
    
    # --- 3. Run Real-Time Prediction ---

    if EXTRACT_INTERMEDIATE_FEATURES:
        print("\nStarting real-time prediction and feature extraction simulation...")
        predictions, targets, prediction_times, event_features, event_sequence = \
        run_realtime_prediction_and_extract_features(
            model=model,
            dataset_cfg=dataset_cfg,
            seq_loader=seq_loader,
            sequence_id=TEST_SEQUENCE_ID,
            dt=PREDICTION_INTERVAL,
            start_offset_s=0.5,
            device=device,
        )
        
        #animate_event_data(event_sequence)
        animate_event_data_with_combined(event_sequence)

        print("\nVisualizing Event Feature Embeddings with t-SNE...")

        velocity_magnitudes = np.linalg.norm(targets[:, 3:6], axis=1)

        if len(event_features) < 30:
            print(f"Not enough event features ({len(event_features)}) for t-SNE visualization. Need at least 30.")
        else:
            perplexity_val = min(30, max(5, int(len(event_features) * 0.1)))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, learning_rate='auto', init='random')
            event_features_2d = tsne.fit_transform(event_features)

            plt.figure(figsize=(10, 8))
            
            # Capture the seaborn Axes object
            ax = sns.scatterplot(
                x=event_features_2d[:, 0], y=event_features_2d[:, 1],
                hue=velocity_magnitudes,
                palette="viridis", # Changed to string, seaborn will create the colormap internally
                s=50, alpha=0.8,
                legend='full'
            )
            
            # Get the mappable object from the Axes for the colorbar
            # This is typically the first child of type ScalarMappable (or related)
            norm = plt.Normalize(vmin=velocity_magnitudes.min(), vmax=velocity_magnitudes.max())
            sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis") # Create a ScalarMappable explicitly
            sm.set_array([]) # Needs a dummy array to avoid a warning

            plt.colorbar(sm, ax=ax, label='Ground Truth Velocity Magnitude (m/s)') # Pass the mappable and the axes
            
            plt.title(f't-SNE Visualization of Event Features for Sequence {TEST_SEQUENCE_ID}')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.grid(True)
            plt.savefig(f"./plots/event_features_tsne_{TEST_SEQUENCE_ID}.png", dpi=300)
            plt.show()

            print("t-SNE plot saved.")
    else:
        
        print("\nStarting real-time prediction simulation...")
        
        predictions, targets, prediction_times = run_realtime_prediction(
            model=model,
            dataset_cfg=dataset_cfg,
            seq_loader=seq_loader,
            sequence_id=TEST_SEQUENCE_ID,
            dt=PREDICTION_INTERVAL, # Make a prediction every 50ms
            start_offset_s=0.5, # Ensure at least 0.5s of data for LSTM warm-up
            device=device,
        )
        
        # Compute test metrics
        test_metrics = LunarTrainer.compute_metrics(
            torch.tensor(predictions), 
            torch.tensor(targets), 
            velocity_only=VELOCITY_ONLY
        )
        
        print(test_metrics)
                    
        if VELOCITY_ONLY:
            
            print(f"Test Metrics - Vel Error: {test_metrics['velocity_error']:.2f}m/s", f"elope_score: {test_metrics['elope_score']:.4f}")
            
        else:
            
            print(f"Test Metrics - Pos Error: {test_metrics['position_error']:.2f}m, "
                    f"Vel Error: {test_metrics['velocity_error']:.2f}m/s", f"elope_score: {test_metrics['elope_score']:.4f}")

    if len(predictions) > 0:
        
        print("\nPrediction Results:")
        print(f"Total predictions: {len(predictions)}")
        print(f"Shape of predictions: {predictions.shape}")
        print(f"Shape of targets: {targets.shape}")

        if VELOCITY_ONLY:
            
            # Calculate individual MSE for each component
            mse_labels = ['Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)']
            
            # State index offset
            offset = 3
               
            fig, axes = plt.subplots(1, 3, figsize=(15, 15)) 
            plot_info = [
                {'idx': 0, 'label': 'Vx', 'ylabel': 'Velocity (m/s)'},
                {'idx': 1, 'label': 'Vy', 'ylabel': 'Velocity (m/s)'},
                {'idx': 2, 'label': 'Vz', 'ylabel': 'Velocity (m/s)'}
            ]
            
        else: 
            
            # Calculate individual MSE for each component
            mse_labels = ['X (m)', 'Y (m)', 'Z (m)', 'Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)']
            
            # State index offset
            offset = 0
            
            # Create a 2x3 grid of subplots for position (x, y, z) and velocity (vx, vy, vz)
            fig, axes = plt.subplots(2, 3, figsize=(15, 15)) 
            plot_info = [
                {'idx': 0, 'label': 'X', 'ylabel': 'Position (m)'},
                {'idx': 1, 'label': 'Y', 'ylabel': 'Position (m)'},
                {'idx': 2, 'label': 'Z', 'ylabel': 'Altitude (m)'},
                {'idx': 3, 'label': 'Vx', 'ylabel': 'Velocity (m/s)'},
                {'idx': 4, 'label': 'Vy', 'ylabel': 'Velocity (m/s)'},
                {'idx': 5, 'label': 'Vz', 'ylabel': 'Velocity (m/s)'}
            ]

        # Calculate Mean Squared Error (MSE) on the test sequence for the whole vector
        mse_total = np.mean((predictions - targets[:, offset:])**2)
        print(f"Overall Mean Squared Error on test sequence {TEST_SEQUENCE_ID}: {mse_total:.6f}")
        
        # Compute the MSE errors 
        for i, label in enumerate(mse_labels): 
            mse_comp = np.mean((predictions[:,i] - targets[:, i+offset])**2)
            print(f"MSE for {label}: {mse_comp:.6f}")
        
        # Flatten the 2D array of axes for easier iteration
        axes = axes.flatten()
        
        # Display the data 
        for i, info in enumerate(plot_info):
            ax = axes[i]
            idx = info['idx']
            label = info['label']
            ylabel = info['ylabel']

            # Diplay groundtruth value
            ax.plot(
                prediction_times, targets[:, idx+offset], label=f'GT {label}', 
                color='blue', linewidth=2, marker='o', markersize=3
            )
            
            # Display network predictions
            ax.plot(
                prediction_times, predictions[:, idx], label=f'Pred {label}', 
                color='red', linestyle='--', linewidth=1.5, marker='x', markersize=3
            )
            
            ax.set_title(f'{label} Prediction vs. Ground Truth')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True)

        plt.suptitle(f'6-DOF State Prediction vs. Ground Truth for Sequence {TEST_SEQUENCE_ID}', fontsize=16, y=1.02) # Add a main title
        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to make space for suptitle
        plt.savefig(f"./plots/lunar_descent_predictions_{TEST_SEQUENCE_ID}.png", dpi=300) # Save with sequence ID in filename and higher DPI
        
    else:
        print("No predictions were made. Check sequence ID, data path, or time offsets.")

    # --- Add Activation Map Visualization Here ---
    print("\nAttempting to visualize activation maps.")
    # We need a single data point to feed into the network for activations.
    # Let's take one from the beginning of the prediction sequence.
    # Make sure you have at least one prediction point
    if len(prediction_times) > 0:
        # Get one data point from the data loader at a specific time
        # This will be the first valid prediction time
                
        # Retrieve all the dataset config values 
        H = int(dataset_cfg["events"]["height"])
        W = int(dataset_cfg["events"]["width"])
        T = int(dataset_cfg["events"]["time_bins"])
        
        INT_WINDOW_US = float(dataset_cfg["events"]["integration_window"])
        
        SEQ_LEN = int(dataset_cfg["imu_sequence_length"])

        first_prediction_time_s = prediction_times[0]
        sample_data_point = seq_loader.get_data_at_time(
            first_prediction_time_s,
            event_integration_window=INT_WINDOW_US,
            imu_seq_len=SEQ_LEN,
            H=H, W=W, T=T
        )
        
        if sample_data_point:
            # Example: Visualize activations from the output of the first ResNet3DBlock (after conv1 and ReLU)
            # The name refers to the layer within the model where we want to hook.
            # In the EventEncoder, the `res_block1` is a ResNet3DBlock.
            # If we want the activation *after* the first ReLU in `res_block1`:
            def print_model_layers(model: nn.Module):
                """
                Prints all named modules (layers) in a PyTorch model and their types.
                This helps in identifying the correct layer_name for hooks or access.
                """
                print("\n--- Model Layer Names and Types ---")
                for name, module in model.named_modules():
                    # Filter out the top-level module itself and potentially empty Sequential modules
                    if name: # Only print non-empty names
                        print(f"Name: {name:<50} Type: {type(module)}")
                print("-----------------------------------\n")

            #print_model_layers(model)
            visualize_activation_maps(model, sample_data_point, 'layer1.1.conv1', feature_map_idx=3, device=device)
            # Or the activation *after* the initial block's MaxPool3d:
            visualize_activation_maps(model, sample_data_point, 'layer3.1.conv1', feature_map_idx=4, device=device) # Index 3 is MaxPool3d if initial_block is Sequential
            # Or the output of an early convolution (e.g., the first one in res_block1):
            visualize_activation_maps(model, sample_data_point, 'conv1', feature_map_idx=3, device=device)

        else:
            print("Could not retrieve a sample data point for activation visualization.")
    else:
        print("No predictions were made, so no sample data point available for activation visualization.")
