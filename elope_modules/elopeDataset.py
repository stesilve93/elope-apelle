import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import os
import numpy as np
from elope_modules.dataloader import DataLoader

class LunarDescentDataset(Dataset):
    def __init__(self, data_loader_instance: DataLoader,
                 sequence_ids: list,
                 event_integration_window_us: float = 1e5,
                 imu_seq_len: int = 50,
                 H: int = 200, W: int = 200, T: int = 10,
                 sample_interval: int = 10):
        """
        Custom PyTorch Dataset for lunar descent data.

        Args:
            data_loader_instance (DataLoader): An instantiated DataLoader object.
            sequence_ids (list): List of sequence IDs (e.g., ['0000', '0001', ...]) to load.
            event_integration_window_us (float): Time window for events in microseconds.
            imu_seq_len (int): Sequence length for IMU and rangemeter data.
            H, W, T: Dimensions for the event tensor.
            sample_interval (int): How often to sample a timestamp from the trajectory
                                   (e.g., 10 means every 10th timestamp).
        """
        self.data_loader = data_loader_instance
        self.samples = []
        self.event_integration_window_us = event_integration_window_us
        self.imu_seq_len = imu_seq_len
        self.H = H
        self.W = W
        self.T = T

        print("Preparing dataset samples...")
        for seq_id in sequence_ids:
            print(f"Loading sequence {seq_id}...")
            # Load the full sequence data into the DataLoader instance
            self.data_loader.load_sequence(seq_id)

            # Generate samples from this loaded sequence
            # Iterate through timestamps from the loaded trajectory
            # Use data_loader.timestamps_full which was populated by load_sequence
            if self.data_loader.timestamps_full is not None:
                for i in range(sample_interval, len(self.data_loader.timestamps_full), sample_interval):
                    t_curr_s = self.data_loader.timestamps_full[i]
                    data_point = self.data_loader.get_data_at_time(
                        t_curr_s,
                        event_integration_window_us=self.event_integration_window_us,
                        imu_seq_len=self.imu_seq_len,
                        H=self.H, W=self.W, T=self.T
                    )
                    if data_point:
                        self.samples.append(data_point)
            print(f"  -> Added {len(self.samples)} samples so far.")
        print(f"Finished preparing dataset. Total samples: {len(self.samples)}")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Return the pre-processed tensors and ground truth
        sample = self.samples[idx]
        return (sample['events_tensor'], sample['imu_sequence'],
                sample['rangemeter_sequence'], sample['ground_truth'])

###########     testing code    ###########
if __name__ == "__main__":
    datapath = './elope_data' # Adjust as needed
    data_loader = DataLoader(datapath=datapath)

    # Generate a list of your 40 train trajectory IDs
    # Assuming they are '0000.npz' to '0039.npz'
    train_sequence_ids = [str(i).zfill(4) for i in range(28)]

    # Create the dataset
    train_dataset = LunarDescentDataset(
        data_loader_instance=data_loader,
        sequence_ids=train_sequence_ids,
        event_integration_window_us=100000, # 100ms window
        imu_seq_len=50,
        H=200, W=200, T=10,
        sample_interval=5 # Adjust sampling frequency
    )

    # Create a PyTorch DataLoader
    batch_size = 16 # Or whatever fits your GPU memory
    train_dataloader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4) # num_workers for parallel data loading