
import numpy as np 
import torch

from pathlib import Path 

from torch.utils.data import Dataset

from elope.utils import (
    LOGGER, 
    dict2hash,
    getfiles,
    load_pickle,
    load_yaml, 
    save_pickle, 
)

from .events import EventProcessor
from .sequence import SequenceLoader

class ElopeDataset(Dataset): 
    
    default_cfg = {
        'save_cache': False, 
        'use_cached': False, 
    }
    
    KEYS_DATASET = [
        "sample_interval", "imu_sequence_length", "imu_padding", "events", "side"
    ]
    
    def __init__(
        self, cfg: dict | str | Path, sequence_ids: list, **kwargs
    ): 
        
        if isinstance(cfg, (str, Path)): 
            # Retrieve the configuration from the yaml file.
            cfg = load_yaml(cfg)
            
        # The final configuration can be overloaded by keyword arguments
        cfg = {**self.default_cfg, **cfg, **kwargs}
        
        # Split the configuration between those related to the dataset structure and 
        # those independent of that (e.g., caching, augmentation, etc...)
        self.cfg_dataset = {}
        for key in self.KEYS_DATASET: 
            self.cfg_dataset[key] = cfg[key]
            
        # Retrieve all the events dimensions
        cfg_events = self.cfg_dataset["events"]
            
        # Create a Sequence loader instance 
        self.seq_loader = SequenceLoader(
            cfg["datapath"], 
            event_integration_window=cfg_events["integration_window"],
            event_encoder_method=cfg_events["encoder_method"],
            event_clamp=cfg_events.get("clamp", -1),
            event_H=cfg_events["height"],
            event_W=cfg_events["width"],
            event_T=cfg_events.get("channels", 1), 
            imu_seq_len=self.cfg_dataset["imu_sequence_length"], 
            imu_padding=self.cfg_dataset["imu_padding"]
        )
        
        # Store the dataset configuration for the sequences as a hash key.
        self.hash = dict2hash(self.cfg_dataset)
        
        # Retrieve the type of event normalization, if given 
        self.event_normalization = cfg.get("event_normalization", "null")
        assert self.event_normalization in ("null", "standard", "minmax")
        
        # Check whether cached data is available, and try loading
        has_cache = False 
        if cfg["use_cached"]: 
            path, name = cfg["output_path"], cfg["output_name"]
            has_cache, seq_samples = self.load_cache(path, name)

        # If we haven't got cached data, parse the entire sequence files and build it.
        if not has_cache: 
        
            # Parse all the files within the directory
            seq_files = getfiles(cfg["datapath"], ".npz")
            seq_files.sort() 
            
            # Retrieve the IDs of all the target sequences
            seq_names = [s.stem for s in seq_files]
            
            seq_samples = {}
            for seq_id in seq_names:
                seq_samples[int(seq_id)] = self.parse_sequence(seq_id)
                
            # Save the cached data 
            if cfg["save_cache"]:
                path, name = cfg["output_path"], cfg["output_name"] 
                self.save_cache(path, name, seq_samples)
                
        # At this point we have a list of subsamples for each sequence trajectory. 
        # Thus, we build a dataset from only the desired sequences.
        self.seq_ids = sequence_ids
        
        self.seq_lengths = []
        self.samples = []
        
        LOGGER.info("Dataset creation initialized.")
        for seq_id in self.seq_ids: 
            subsamples = seq_samples[(int(seq_id))]
            self.samples.extend(subsamples)
            self.seq_lengths.append(len(subsamples))
            LOGGER.info(f"Added {len(subsamples)} samples from Sequence `{seq_id}`.")
            
        LOGGER.info(f"Dataset created. Total samples: {len(self)}")
        
        # Compute the minimum and maximum sequence lengths 
        self.seq_len_min = np.min(self.seq_lengths)
        self.seq_len_max = np.max(self.seq_lengths)
        
        # Store whether the data should be augment 
        self.augment = cfg.get("augment", False)
    
        # Retrieve the rangemeter noise setting. This is expressed as a percentage 
        # of the actual rangemeter value (from 0.0 to 1.0)
        self.rangemeter_noise = np.clip(float(cfg.get("rangemeter_noise", 0.0)), 0, 1)
        
        # Retrieve the angle noise setting. Expressed as a percentage
        self.angles_noise = np.clip(float(cfg.get("angles_noise", 0.0)), 0, 1)
        
        # Retrieve the angular velocity noise, expressed as a percentage 
        self.angles_vel_noise = np.clip(float(cfg.get("angles_vel_noise", 0.0)), 0, 1)
            
    def load_cache(self, path: str, name: str) -> tuple: 
        """Retrieve the dataset subsamples from a cache file.
        
        Parameters
        ----------
        path : str 
            Path to the folder in which to store the cached sequence samples.
        name : str
            Name of the cache file.
             
        Returns 
        -------
        flag : bool
            True if cached data is available 
        subsamples : dict 
            Dictionary of sequence IDs and subsamples.
        """
        
        # Check whether a cached dataset is available 
        cache_name = name + ".ape"
        cache_path = Path(path) / cache_name
        
        if not (cache_path.exists() and cache_path.suffix == ".ape"):
            return False, []
         
        LOGGER.info(f"Retrieving dataset samples from cache: \033[33m{cache_path}\033[0m.")          
        cache_data = load_pickle(cache_path, compressed=True)

        # Check if the dataset settings are coherent
        if cache_data[0] != self.hash: 
            LOGGER.warning("Incompatible cache hash found. Re-generating dataset.")
            return False, []
        
        return True, cache_data[1]
    
    def save_cache(self, path: str, name: str, seq_samples: dict):
        """Save the dataset subsamples and configuration hash string.
        
        Parameters
        ----------
        path : str 
            Path to the folder in which to store the cached sequence samples.
        name : str
            Name of the cache file.
        seq_samples : dict 
            Dictionary of sequence IDs and subsamples.        
        """

        # Compute the path in which to store the cached data 
        cache_name = name + ".ape"
        cache_path = Path(path) / cache_name
        
        # Ensure the output folder exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Store the data in a binary pickle file
        save_pickle(cache_path, (self.hash, seq_samples), compress=True)
        LOGGER.info(f"Cached dataset at: \033[33m{cache_path}\033[0m")   
         
    def parse_sequence(self, seq_id: str) -> list:
        """Extract trajectory samples from a sequence trajectory.
        
        Parameters
        ----------
        seq_id : str
            Sequence string identifier.
            
        Returns
        -------
        subsamples : list 
            List of sequence samples.
        """
        
        # Load the full sequence data and get its length
        self.seq_loader.load_sequence(seq_id)
        seq_len = len(self.seq_loader.timestamps_full)
        if seq_len == 0:  
            return []
            
        # Retrieve the sequence sampling interval 
        sample_interval = int(self.cfg_dataset["sample_interval"])
        
        # Get the directions from which to create the dataset
        cfg_side = self.cfg_dataset["side"]
        if cfg_side in ("left", "right"): 
            sides = [cfg_side]
        elif cfg_side == "both": 
            sides = ["left", "right"]
        else: 
            raise ValueError(f"`{cfg_side}` is not a supported side value.")
        
        subsamples = []
        for side in sides: 
            
            # Check whether we should flip the data 
            flip = side == "right"
            
            # Pre-load the sequence events
            self.seq_loader.preprocess_events(side=side)
            
            # Iterate over the entire trajectory and collect samples 
            for k in range(0, seq_len, sample_interval): 
                # Retrieve the data points at this time
                data_k = self.seq_loader.get_data_at_time(
                    self.seq_loader.timestamps_full[k], flip=flip
                )
                
                if data_k: 
                    subsamples.append(data_k)

        return subsamples
    
    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx: int): 
        
        # Return the pre-processed tensors and groundtruth data 
        sample = self.samples[idx]
        
        # Retrieve the different sequences 
        events     = sample['events_tensor']
        imu_seq    = sample['imu_sequence'].clone()
        rangemeter = sample['rangemeter_sequence'].clone()
        targets    = sample['ground_truth']
        times      = sample['times'] 
        
        # Normalize the event tensor, if requested 
        if self.event_normalization != "null": 
            
            event_clamp = self.cfg_dataset["events"].get("clamp", -1)
            max_val = event_clamp if event_clamp > 0 else None       
            
            for k in range(events.shape[0]): 
                events[k] = EventProcessor.normalize_tensor(
                    events[k], method=self.event_normalization, max_val=max_val
                )
        
        if not self.augment: 
            # No augmentation is performed
            return events, imu_seq, rangemeter, targets, times
        
        if self.rangemeter_noise > 0.0: 
            # Add rangemeter noise on the sequence 
            noise_fct = 2*torch.rand(rangemeter.shape) - 1 
            rangemeter = rangemeter*(1 + self.rangemeter_noise*noise_fct)
            
        if self.angles_noise > 0.0: 
            # Add noise on the Euler angles 
            noise_fct = 2*torch.rand(imu_seq[..., 0:3].shape) - 1 
            imu_seq[..., 0:3] = imu_seq[..., 0:3]*(1 + self.angles_noise*noise_fct)
        
        if self.angles_vel_noise > 0.0: 
            # Add noise on the angular velocities 
            noise_fct = 2*torch.rand(imu_seq[..., 3:6].shape) - 1
            imu_seq[..., 3:6] = imu_seq[..., 3:6]*(1 + self.angles_vel_noise*noise_fct)

        # TODO: EVENT NOISE???

        return events, imu_seq, rangemeter, targets, times