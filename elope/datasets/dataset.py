
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
from .sequence import FixedSequenceLoader, VariableSequenceLoader

class ElopeDataset(Dataset): 
    
    default_cfg = {
        'save_cache': False, 
        'use_cached': False, 
        'time_step': -1, 
    }
    
    KEYS_DATASET = ["sequence_type", "time_step", "events"]
    
    def __init__(
        self, 
        cfg: dict | str | Path, 
        sequence_ids: list, 
        sample_len: int, 
        verbose: bool=True, 
        event_integration_window: float=None,
        **kwargs
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
            
        # Store the length of the IMU sequence 
        self.sample_len = sample_len
        
        # Retrieve the interval between sub-sequences 
        self.sample_interval = int(cfg.get("sample_interval", 1))
        
        # Retrieve the type of data padding 
        self.padding = cfg.get("padding", "static")
        assert self.padding == "static"
        
        # Retrieve the type of sequence 
        seq_type = self.cfg_dataset["sequence_type"]
        assert seq_type in ("fixed", "variable")
        seq_cls = FixedSequenceLoader if seq_type == "fixed" else VariableSequenceLoader
        
        # Create a Sequence loader instance 
        self.seq_loader = seq_cls(
            cfg["datapath"], 
            time_step=float(self.cfg_dataset["time_step"]),
            event_integration_window=float(cfg_events["integration_window"]),
            event_encoder_method=cfg_events["encoder_method"],
            event_clamp=float(cfg_events.get("clamp", -1)),
            event_H=int(cfg_events["height"]),
            event_W=int(cfg_events["width"]),
            event_T=int(cfg_events.get("channels", 1)), 
            sequence_len=self.sample_len,
            sequence_pad=self.padding
        )
        
        # Store the dataset configuration for the sequences as a hash key.
        self.hash = dict2hash(self.cfg_dataset)
        
        # Retrieve the type of event normalization, if given 
        self.event_normalization = cfg.get("event_normalization", "null")
        assert self.event_normalization in ("null", "standard", "minmax")
        
        self.new_integration_window = None
        if event_integration_window is not None: 
            # This makes sense only if the event encoder is last_timestamp
            assert self.cfg_dataset["events"]["encoder_method"] == "last_timestamp"
            
            original_window = float(cfg["events"]["integration_window"])
            if event_integration_window != original_window: 
                LOGGER.info(f"Event window reset to: {event_integration_window}")
                self.new_integration_window = event_integration_window
                
        # Check whether cached data is available, and try loading
        has_cache = False 
        if cfg["use_cached"]: 
            path, name = cfg["output_path"], cfg["output_name"]
            has_cache, seq_samples = self.load_cache(path, name)

        # If we haven't got cached data, parse the entire sequence files and build it.
        if not has_cache: 
            
            if cfg["save_cache"]: 

                # Parse all the files within the directory
                seq_files = getfiles(cfg["datapath"], ".npz")
                seq_files.sort() 
                
                seq_names = [s.stem for s in seq_files]
            
            else: 
                # Since we are not caching the data, we don't need to process all seqs.    
                seq_names = sequence_ids
                
            # Parse the target sequences.
            seq_samples = {}
            for seq_id in seq_names:
                # Retrieve the sequence data
                seq_data = self.parse_sequence(seq_id)
                if seq_data is not None:
                    seq_samples[int(seq_id)] = seq_data
                
            # Save the cached data 
            if cfg["save_cache"]:
                path, name = cfg["output_path"], cfg["output_name"] 
                self.save_cache(path, name, seq_samples)
                
        # At this point we have a list of subsamples for each sequence trajectory. 
        # Thus, we build a dataset from only the desired sequences.
        self.seq_ids = sequence_ids
        
        self.seq_lengths = []
        self.seq_indexes_beg = []
        self.seq_indexes_end = []
        self.samples = []
         
        LOGGER.info("Dataset creation initialized.")
        for seq_id in self.seq_ids: 
            
            # Retrieve the sequence options
            if not int(seq_id) in seq_samples.keys(): 
                LOGGER.warning(f"Unable to find data for sequence: {seq_id}")
                continue 
            
            # Convert the sequence arrays into dataset subsamples
            subsamples = self._seq2samples(seq_samples[(int(seq_id))])
            
            # Store the initial sample index of each seq.
            self.seq_indexes_beg.append(len(self.samples)) 
            
            self.samples.extend(subsamples)
            self.seq_lengths.append(len(subsamples))
            
            # Store the final sample index of each seq
            self.seq_indexes_end.append((len(self.samples)-1))
            
            if verbose:
                LOGGER.info(f"Added {len(subsamples)} samples from Sequence `{seq_id}`.")
                
        self.seq_indexes_beg = np.array(self.seq_indexes_beg, dtype=int)
        self.seq_indexes_end = np.array(self.seq_indexes_end, dtype=int)
            
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
        
        # Retrieve the flipping probability 
        self.flip_prob = float(cfg.get("flip", 0.0))
            
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
        LOGGER.info(f"Saving dataset cache at: \033[33m{cache_path}\033[0m")
        save_pickle(cache_path, (self.hash, seq_samples), compress=True)
        LOGGER.info(f"Dataset cache saving completed.")
         
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
        self.seq_loader.load_sequence(seq_id, events_side="both")
        if len(self.seq_loader) == 0: 
            return None

        # Retrieve all the arrays of the sequence
        return {
            "imu": torch.from_numpy(self.seq_loader.seq_imu .astype(np.float32)), 
            "times": torch.from_numpy(self.seq_loader.seq_times .astype(np.float32)), 
            "targets": torch.from_numpy(self.seq_loader.seq_states.astype(np.float32)), 
            "rangemeter": torch.from_numpy(self.seq_loader.seq_ranges.astype(np.float32)), 
            "events_left": torch.from_numpy(
                self.seq_loader.seq_events_left.astype(np.float32)
            ),
            "events_right": torch.from_numpy(
                self.seq_loader.seq_events_right.astype(np.float32)
            )
        }
        
    def _seq2samples(self, seq: dict) -> list:
        """Convert a dictionary with sequence data into dataset samples.""" 
        
        # Retrieve the sequence states at the given interval
        imu          = seq["imu"][::self.sample_interval]
        times        = seq["times"][::self.sample_interval]
        targets      = seq["targets"][::self.sample_interval]
        rangemeter   = seq["rangemeter"][::self.sample_interval]
        events_left  = seq["events_left"][::self.sample_interval]
        events_right = seq["events_right"][::self.sample_interval]
        
        ns = len(times)
        samples = []
        for k in range(ns): 
            
            # Retrieve the events
            event_lk = events_left[k] 
            event_rk = events_right[k]
            
            # Check whether they have to be resized to a different window
            if self.new_integration_window is not None: 
                event_lk = EventProcessor.reduce_window(
                    event_lk, 
                    float(self.cfg_dataset["events"]["integration_window"]), 
                    self.new_integration_window
                )
                
                event_rk = EventProcessor.reduce_window(
                    event_rk, 
                    float(self.cfg_dataset["events"]["integration_window"]), 
                    self.new_integration_window
                )
            
            samples.append((
                times[k], 
                targets[k], 
                imu[k], 
                rangemeter[k], 
                event_lk,
                event_rk,
            ))
            
        return samples
    
    def __len__(self) -> int: 
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple: 
        
        # Check whether to flip the sequence 
        flip = self.augment and torch.rand(1).item() < self.flip_prob

        if flip: 
            return self._getseq_right(idx)
        else: 
            return self._getseq_left(idx)
        
    def _getseq_left(self, idx: int) -> tuple: 
        """Retrieve a forward sequence at at given dataset index."""
    
        # Retrieve tensor sizes 
        ni = self.sample_len 
        T, H, W = self.seq_loader.T, self.seq_loader.H, self.seq_loader.W
        
        # Initialize all the arrays 
        times   = torch.empty(ni, dtype=torch.float32)
        targets = torch.empty(ni, 6, dtype=torch.float32) 
        imus    = torch.empty(ni, 6, dtype=torch.float32)
        ranges  = torch.empty(ni, 1, dtype=torch.float32)
        events  = torch.empty(ni, 2, T, H, W, dtype=torch.float32)
        
        # Retrieve the starting index of the sequence in which we are
        idx_offset = idx - self.seq_indexes_beg
        mask = idx_offset >= 0
        
        idx_beg = int(self.seq_indexes_beg[mask][np.argmin(idx_offset[mask])])
        
        # Retrieve the index relative to the beginning of the trajectory
        idx_rel = idx - idx_beg
        
        # Compute the number of padding values we need to add at the beginning    
        npads = max(0, ni - 1 - idx_rel)
        
        # Fill the part of the sequence that is available
        for k in range(npads, ni): 
            
            sk = self.samples[idx-ni+k+1]
            
            times[k] = sk[0]
            targets[k, :] = sk[1]
            imus[k, :] = sk[2]
            ranges[k, :] = sk[3]
            events[k, :] = sk[4]
        
        # Add the initial padding values for the remaining data
        if npads > 0:
            
            # Retrieve the initial point of that sequence and get the time step
            s0 = self.samples[idx_beg]
            dt = self.samples[idx_beg+1][0] - s0[0]
            
            times[:npads] = s0[0] + torch.arange(-npads, 0, 1)*dt
            targets[:npads, :] = torch.hstack((s0[1][:3], torch.zeros(3)))
            
            imus[:npads, :] = torch.hstack((s0[2][:3], torch.zeros(3)))
            ranges[:npads, :] = s0[3]
            
            # What happens here depends on the type of encoding
            events[:npads, :] = events[npads]
            
        # Normalize the event tensor, if requested 
        if self.event_normalization != "null": 
            
            event_clamp = self.cfg_dataset["events"].get("clamp", -1)
            max_val = event_clamp if event_clamp > 0 else None       
            
            for k in range(events.shape[0]): 
                events[k] = EventProcessor.normalize_tensor(
                    events[k], method=self.event_normalization, max_val=max_val
                )
                
        # Normalize time w.r.t. the beginning of the sequence 
        times[:] = times[:] - times[0]
        
        if not self.augment: 
            return events, imus, ranges, targets, times

        # Add augmentation 
        return self._augment(events, imus, ranges, targets, times)
        
    def _getseq_right(self, idx: int) -> tuple: 
        """Retrieve a backward sequence at a given dataset index."""
        
        # Retrieve tensor sizes 
        ni = self.sample_len 
        T, H, W = self.seq_loader.T, self.seq_loader.H, self.seq_loader.W
        
        # Initialize all the arrays 
        times   = torch.empty(ni, dtype=torch.float32)
        targets = torch.empty(ni, 6, dtype=torch.float32) 
        imus    = torch.empty(ni, 6, dtype=torch.float32)
        ranges  = torch.empty(ni, 1, dtype=torch.float32)
        events  = torch.empty(ni, 2, T, H, W, dtype=torch.float32)
        
        # Retrieve the starting index of the sequence in which we are 
        idx_offset = idx - self.seq_indexes_end
        mask = idx_offset <= 0
        
        idx_beg = int(self.seq_indexes_end[mask][np.argmax(idx_offset[mask])])
        
        # Retrieve the index relative to the beginning of the trajectory 
        idx_rel = idx_beg - idx
        
        # Compute the number of padding values we need to add at the beginning    
        npads = max(0, ni - 1 - idx_rel)
        
        # Fill the part of the sequence that is available
        for k in range(npads, ni): 
            
            sk = self.samples[idx+ni-k-1]
        
            times[k] = sk[0]
            targets[k, :] = sk[1]
            imus[k, :] = sk[2]
            ranges[k, :] = sk[3]
            events[k, :] = sk[5]  
        
        # Add the initial padding values for the remaining data 
        if npads > 0: 
            
            # Retrieve the initial point of that sequence and get the time step
            s0 = self.samples[idx_beg]
            dt = s0[0] - self.samples[idx_beg-1][0]
            
            times[:npads] = s0[0] + torch.arange(0, npads, 1)*dt
            targets[:npads, :] = torch.hstack((s0[1][:3], torch.zeros(3)))
            
            imus[:npads, :] = torch.hstack((s0[2][:3], torch.zeros(3)))
            ranges[:npads, :] = s0[3]
            
            # What we place here depends on the type of event-encoding 
            events[:npads] = 0.0 
            
            # Update the first timestamps values to 1.0 (event never happend)
            if self.seq_loader.event_encoder_method == "hybrid": 
                events[:npads, :, 1] = 1.0 
            
            elif self.seq_loader.event_encoder_method == "first_timestamp": 
                events[:npads] = 1.0
        
        # Invert the timings (otherwise derivatives are not coherent in time) 
        times[:] = times.flip(0)
        
        # Invert the sign of the velocity quantities 
        imus[:, 3:6]    = -imus[:, 3:6]
        targets[:, 3:6] = -targets[:, 3:6]
        
        # Normalize the event tensor, if requested 
        if self.event_normalization != "null": 
            
            event_clamp = self.cfg_dataset["events"].get("clamp", -1)
            max_val = event_clamp if event_clamp > 0 else None       
            
            for k in range(events.shape[0]): 
                events[k] = EventProcessor.normalize_tensor(
                    events[k], method=self.event_normalization, max_val=max_val
                )
                
        # Normalize time w.r.t. the beginning of the sequence 
        times[:] = times[:] - times[0]
        
        if not self.augment: 
            return events, imus, ranges, targets, times

        # Add augmentation 
        return self._augment(events, imus, ranges, targets, times)
    
    def _augment(
        self, 
        events: torch.Tensor, 
        imus: torch.Tensor, 
        ranges: torch.Tensor, 
        targets: torch.Tensor, 
        times: torch.Tensor
    ) -> tuple: 
        """Function to add noise on dataset samples."""

        if self.rangemeter_noise > 0.0: 
            # Add rangemeter noise on the sequence 
            noise_fct = 2*torch.rand(ranges.shape) - 1 
            ranges[:] = ranges*(1 + self.rangemeter_noise*noise_fct)
            
        if self.angles_noise > 0.0: 
            # Add noise on the Euler angles 
            noise_fct = 2*torch.rand(imus[..., 0:3].shape) - 1 
            imus[..., 0:3] = imus[..., 0:3]*(1 + self.angles_noise*noise_fct)
        
        if self.angles_vel_noise > 0.0: 
            # Add noise on the angular velocities 
            noise_fct = 2*torch.rand(imus[..., 3:6].shape) - 1
            imus[..., 3:6] = imus[..., 3:6]*(1 + self.angles_vel_noise*noise_fct)

        # TODO: EVENT NOISE???

        return events, imus, ranges, targets, times