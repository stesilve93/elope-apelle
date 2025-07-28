
from pathlib import Path

from torch.utils.data import DataLoader

from .dataset import ElopeDataset


class ElopeDataLoader(DataLoader): 
    
    def __init__(
        self, 
        cfg_dataset: dict | str | Path, 
        sequence_ids: list, 
        sample_len: int,
        sample_interval: int=1,
        augment: bool=False, 
        padding: str="static",
        verbose: bool=True,
        event_normalization: str="null",
        event_integration_window: float=None,
        flip: float=0.0,
        rangemeter_noise: float=0.0,
        angles_noise: float=0.0,
        angles_vel_noise: float=0.0,
        **kwargs
    ):
        # DOCME
        
        # Create the dataset from the sequence IDs
        dataset = ElopeDataset(
            cfg_dataset, 
            sequence_ids, 
            sample_len,
            sample_interval=sample_interval,
            padding=padding,
            event_normalization=event_normalization,
            event_integration_window=event_integration_window,
            augment=augment,
            verbose=verbose, 
            flip=flip,
            rangemeter_noise=rangemeter_noise,
            angles_noise=angles_noise, 
            angles_vel_noise=angles_vel_noise, 
        )
        
        # Initialize the baseline class
        super().__init__(dataset, **kwargs)
         
        