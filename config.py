from pathlib import Path
from typing import Dict, Optional, Union, Any

def get_config() -> Dict[str, Any]:
    """
    Load the configuration file and return the config dictionary.
    
    Returns:
        Dict[str, Any]: Dictionary containing configuration parameters
    """

    # Default configuration parameters
    config = {
        "batch_size": 8,
        "seq_len": 350,
        "num_epochs": 20,
        "lr": 0.0001,
        "d_model": 512,
        "datasource": "opus_books",
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"   
    }

    return config

def get_weights_file_path(config: Dict[str, Any], epoch: str) -> str:
    """
    Return the path to the weights file for a given epoch.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        epoch (str): Epoch identifier
        
    Returns:
        str: Path to the weights file
    """
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config: Dict[str, Any]) -> Optional[str]:
    """
    Return the path to the latest weights file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        Optional[str]: Path to the latest weights file or None if no files found
    """
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    
    # Check if folder exists before searching for weights
    folder_path = Path(model_folder)
    if not folder_path.exists():
        print(f"Warning: Model folder '{model_folder}' does not exist")
        return None
        
    try:
        # Get all weight files and sort them by modification time (newest last)
        weights_files = list(folder_path.glob(model_filename))
        if len(weights_files) == 0:
            return None
            
        # More reliable sorting using numerical values in filenames
        # Extracts epoch number from filename for better sorting
        weights_files.sort(key=lambda x: int(x.stem.split('_')[-1]) if x.stem.split('_')[-1].isdigit() else 0)
        return str(weights_files[-1])
    except Exception as e:
        print(f"Error finding latest weights file: {e}")
        return None