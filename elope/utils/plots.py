
import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
import torch 

from matplotlib import ticker

def gridminor(ax, xlog: bool=False, ylog: bool=False): 
    """Add a grid on a set of axis. 
    
    Parameters
    ----------
    ax
        Matplotlib's Axis instance. 
    xlog : bool, optional 
        True if the x-axis is in logarithmic scale. Defaults to False. 
    ylog : bool, optional 
        True if the y-axis is in a logarithmic scale. Defaults to False.
    """

    # Add the major grid lines
    ax.grid(True, which='major', linestyle='-', linewidth=0.8, color='#dedede', zorder=1)  
    # Add the minor grid lines
    ax.grid(True, which='minor', linestyle='--', linewidth=0.5, color='#ececec', zorder=1) 
    
    if xlog:
        # Handle logarithmic grid minor positions
        ax.xaxis.set_minor_locator(
            ticker.LogLocator(10, subs=0.1*np.arange(2, 10), numticks=10)
        )
    else: 
        # Standard linear grid minor ticks
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    if ylog: 
        ax.yaxis.set_minor_locator(
            ticker.LogLocator(10, subs=0.1*np.arange(2, 10), numticks=10)
        ) 
    else: 
        # Standard linear grid minor ticks
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())    

    