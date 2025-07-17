
import glob 
import re 

import yaml 

from pathlib import Path 

def check_suffix(file: str | Path, suffix): 
    """Check file(s) for acceptable suffixes.
    
    Parameters
    ----------
    file : list, str
        String or list of files to be checked. 
    suffix: str, tuple 
        Allowed file suffices.
    """
    
    if isinstance(suffix, str): 
        suffix = (suffix,)
    
    # Check for each file that the suffix is correct
    for f in file if isinstance(file, (list, tuple)) else [file]: 
        # Retrieve the file suffix
        s = Path(file).suffix.lower().strip() 
        assert s in suffix, f"`{file}` acceptable suffices are {suffix}."    
        

def load_yaml(filename: str) -> dict: 
    """Read the content of a .yml file.
    
    Parameters
    ----------
    filename : str 
        Path to the .YAML file.
        
    Returns 
    -------
    data : dict 
        YAML file content.
    """
    
    # Check the file suffix 
    check_suffix(filename, (".yml", ".yaml"))
    
    # Read the file content
    with open(filename, 'r') as f: 
        data = yaml.safe_load(f)
        
    return data


def save_yaml(filepath: str | Path, data: dict): 
    """Store content into a YAML file.
    
    Parameters
    ----------
    filepath : str or Path 
        Filepath to the .json file. 
    data : dict 
        Data to be saved. 
    """
    
    # Check the file suffix 
    check_suffix(filepath, (".yml", ".yaml"))
    
    # Write the file content 
    with open(filepath, 'w') as f: 
        yaml.dump(data, f)


def increment_path(path: str | Path, exist_ok: bool=True, sep: str = '-') -> Path:
    # DOCME 
    
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return path
    
    # Search for all the paths with similar names
    dirs = glob.glob(f"{path}{sep}*") 
    matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
    
    # Retrieve the highest existing index
    i = [int(m.groups()[0]) for m in matches if m] 
    n = max(i) + 1 if i else 2  
    return Path(f"{path}{sep}{n}")


def getfiles(path: str | Path, ext: str=None) -> list: 
    """Return a list of all the files in a directory.
    
    Parameters
    ----------
    path : str or Path
        Search directory.
    ext : str or tuple, optional 
        Desired file extension(s), including the dot (e.g., .png). Defaults to None.
    """
    path = Path(path) 
    if ext is None: 
        return [f for f in path.iterdir() if f.is_file()]
    else: 
        
        if isinstance(ext, str): 
            ext = (ext,)
    
        return [f for f in path.iterdir() if f.suffix in ext]
