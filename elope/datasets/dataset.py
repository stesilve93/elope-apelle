
import numpy as np 
import torch

from pathlib import Path 

from torch.utils.data import Dataset

from elope.utils import LOGGER, load_yaml

from .sequence import SequenceLoader

class ElopeDataset(Dataset): 
    
    def __init__(
        self, cfg: dict | str | Path, sequence_ids: list, mode: str="train",
        **kwargs
    ): 
            
        # The final configuration can be overloaded by keyword arguments
        cfg = {**cfg, **kwargs}
        
        # TODO: sequence list should be in the configuration file
            
        # Store the configuration settings 
        self.cfg = cfg 
        self.seq_ids = sequence_ids
            
        # Retrieve all the events dimensions
        cfg_events = cfg["events"]
        
        self.event_H = int(cfg_events["height"])
        self.event_W = int(cfg_events["width"])
        self.event_T = int(cfg_events["time_bins"])
        self.event_integration_window = float(cfg_events["integration_window"])
        self.event_encoder_method = str(cfg_events["encoder_method"])
        
        # Retrieve the length of the IMU and altimeter sequence 
        self.imu_seq_len = int(cfg["imu_sequence_length"])
        
        # Store whether samples should be retrieved sequentially during training 
        self.sequential_samples = bool(cfg["sequential_samples"])
        
        self.batch_size = int(cfg["batch_size"])
        self.batch_interval = int(cfg["batch_interval"])
        
        self.sample_interval = int(cfg["sample_interval"])
        
        # Create a Sequence loader instance 
        self.seq_loader = SequenceLoader(cfg["datapath"])
        
        # Store the lenght of each sequence 
        self.seq_lengths = []
        self.samples = []
        
        LOGGER.info("Dataset creation initialized.")
        for seq_id in self.seq_ids: 
        
            # Retrieve the sequence subsamples and store their number 
            subsamples = self.get_sequence_samples(seq_id) 
            
            seq_len = len(subsamples)
            if seq_len == 0: 
                # No sequence data is available. 
                continue 
            
            LOGGER.info(f"Populating dataset with sequence samples.")
            
            # Store the length of the sequence
            self.seq_lengths.append(seq_len)
            
            if not self.sequential_samples: 
                self.samples.extend(subsamples)
                LOGGER.info(f"Added {len(self)} samples so far.\n")
                continue
            
            if seq_len < self.batch_size: 
                # If the sequence is shorter than the batch-size we can't really use it
                # TODO: maybe we can pad the data?? 
                continue 
            
            for i in range(0, seq_len-self.batch_size+1, self.batch_interval): 
                
                # Extract the subsamples for this sample 
                samples_i = subsamples[i:i+self.batch_size]
                
                times_i, states_i, events_i, imu_i, rangemeter_i = [], [], [], [], []
                for s in samples_i: 
                    times_i.append(s['time'])
                    events_i.append(s['events_tensor'])
                    states_i.append(s['ground_truth'])
                    imu_i.append(s['imu_sequence'])
                    rangemeter_i.append(s['rangemeter_sequence'])
                    
                # Create the dictionary for the i-th sample by stacking all the information
                # of the sub-sequence on the batch-dimension (i.e., 0).
                self.samples.append({
                    'time': torch.stack(times_i, dim=0),
                    'events_tensor': torch.stack(events_i, dim=0), 
                    'imu_sequence': torch.stack(imu_i, dim=0), 
                    'rangemeter_sequence': torch.stack(rangemeter_i, dim=0), 
                    'ground_truth': torch.stack(states_i, dim=0)
                })
                
            
            LOGGER.info(f"Added {len(self)} samples so far.\n")
        
        # Compute the minimum and maximum sequence lengths 
        self.seq_len_min = np.min(self.seq_lengths)
        self.seq_len_max = np.max(self.seq_lengths)
        
        LOGGER.info(f"Finished preparing the dataset. Total samples: {len(self)}.")
        
        # Check whether the dataset should be saved 
        cache = bool(cfg["save_cache"])
        if cache:     
            
            # Create the cache directory
            cache_path = Path(cfg["output_path"])
            cache_path.mkdir(parents=True, exist_ok=True)
            
            cache_name = cfg["output_name"] + "-" + mode + ".pth"
            fullpath = cache_path / cache_name
            
            torch.save(self, fullpath)
            LOGGER.info(f"Cached dataset at: \033[33m{fullpath}\033[0m")
    
    @staticmethod 
    def from_yaml(cfg: str | Path, sequence_ids: list=[], mode: str="train", **kwargs): 
        # DOCME 
        
        # Retrieve the dataset configuration
        cfg = load_yaml(cfg)
        
        # Create the dataset from a dictionary config
        return ElopeDataset.from_dict(cfg, sequence_ids, mode, **kwargs)
    
    @staticmethod 
    def from_dict(cfg: dict, sequence_ids: list=[], mode: str="train", **kwargs): 
        # DOCME
        
        # The final configuration can be overloaded by keyword arguments
        cfg = {**cfg, **kwargs}
        
        if cfg["use_cached"]: 
            # Check whether a cached model is available 
            dataset_name = cfg["output_name"] + "-" + mode + ".pth"
            dataset_path = Path(cfg["output_path"]) / dataset_name
            
            if dataset_path.exists() and dataset_path.suffix == ".pth": 
                LOGGER.info(f"Retrieving dataset from cache: \033[33m{dataset_path}\033[0m.")
                return torch.load(dataset_path, weights_only=False)

        return ElopeDataset(cfg, sequence_ids, mode=mode)
        
    def get_sequence_samples(self, seq_id: str) -> list: 
        """Return a sequential list of samples from the latest loaded trajectory sequence.
        
        Parameters
        ----------
        seq_id : str 
            String identifier of the sequence file name.
        
        Returns 
        -------
        subsamples : list
        
        """
        
        subsamples = []
                
        # Load the full sequence data 
        self.seq_loader.load_sequence(seq_id)
            
        if self.seq_loader.seq_len == 0: 
            return subsamples    
        
        # Iterate over the entire trajectory and collect samples
        seq_len = len(self.seq_loader.timestamps_full)
        for i in range(0, seq_len, self.sample_interval):
            
            data_i = self.seq_loader.get_data_at_time(
                self.seq_loader.timestamps_full[i], 
                event_integration_window=self.event_integration_window, 
                imu_seq_len=self.imu_seq_len, 
                H=self.event_H, 
                W=self.event_W, 
                T=self.event_T, 
                event_encoder_method=self.event_encoder_method
            )
            
            if data_i: 
                subsamples.append(data_i) 
        
        return subsamples
    
    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx: int): 
        
        # Return the pre-processed tensors and groundtruth data 
        sample = self.samples[idx]
        
        return (
            sample['events_tensor'], 
            sample['imu_sequence'], 
            sample['rangemeter_sequence'],
            sample['ground_truth'], 
            sample['time']
        )