
from torch.utils.data import DataLoader

from .dataset import ElopeDataset


class ElopeDataLoader(DataLoader): 
    
    def __init__(self, dataset: ElopeDataset, **kwargs):
        
        # Retrieve the batch-size and make sure its coherent with the dataset settings
        batch_size = kwargs.pop("batch_size")
        
        if dataset.sequential_samples:
            
            if dataset.batch_size != batch_size: 
                raise RuntimeError(
                     "Dataset is set to sequential mode but the specified "
                     "batch size is different from the dataset one. "
                    f"Found {batch_size}, expected {dataset.batch_size}"
                )
                
            else: 
                # The batch-size is embedded in the dataset samples to ensure the points 
                # are sequential.
                batch_size=1
                
        # Store whether samples are sequential
        self.sequential_samples = dataset.sequential_samples
        
        # Initialize the baseline class
        super().__init__(dataset, batch_size=batch_size, **kwargs)
         