import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from elope_modules.landing_pipeline import LunarDescentPipeline
from elope_modules.dataloader import DataLoader

# Main execution function
def run_lunar_descent_pipeline():
    """Run the complete pipeline"""
    
    # Load data
    datapath = '/home/stesilve/Documents/github/elope/elope_data'
    fn = os.path.join(datapath, 'train', '0000.npz')
    
    try:
        events, timestamps, trajectory, rangemeter = \
            DataLoader().load_sequence(sequence_id='0000')
        
        # Initialize and run pipeline
        pipeline = LunarDescentPipeline(use_gpu=True, acc_time=1e5)  # Enable GPU if available
                
        results = pipeline.process_sequence(events, timestamps, trajectory, rangemeter)
        
        if results is not None:
            print("\nPipeline execution completed successfully!")
            print(f"Estimated velocities for {len(results['estimated_velocities'])} time steps")
            
            # Visualize results
            pipeline.visualize_results(results, trajectory, timestamps)
            
            # Print some statistics
            if len(results['estimated_velocities']) > 0:
                print("\nVelocity Estimation Statistics:")
                print(f"Mean estimated velocity: {np.mean(results['estimated_velocities'], axis=0)}")
                print(f"Std estimated velocity: {np.std(results['estimated_velocities'], axis=0)}")
        else:
            print("Pipeline failed to generate results")
            
    except FileNotFoundError:
        print(f"Data file not found at {fn}")
        print("Please ensure the data file exists at the specified path")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    run_lunar_descent_pipeline()