
from pathlib import Path

from torch.utils.data import DataLoader

from .dataset import ElopeDataset


class ElopeDataLoader(DataLoader): 
    
    def __init__(
        self, 
        cfg_dataset: dict | str | Path, 
        sequence_ids: list, 
        augment: bool=False, 
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
            augment=augment, 
            rangemeter_noise=rangemeter_noise,
            angles_noise=angles_noise, 
            angles_vel_noise=angles_vel_noise
        )
        
        # Initialize the baseline class
        super().__init__(dataset, **kwargs)
         
        