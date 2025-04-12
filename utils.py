import pandas as pd
import os
import logging
import uuid
import json

logger = logging.getLogger(__name__)

def save_csv(data, dataset_type, suffix=""):
    """
    Save dataset to a CSV file.
    
    Args:
        data: List of dictionaries or DataFrame containing dataset
        dataset_type: Type of dataset (used for filename)
        suffix: Optional suffix for the filename
        
    Returns:
        str: Path to the saved CSV file
    """
    if not data:
        logger.warning("Empty data provided to save_csv")
        return None
    
    # Ensure data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        try:
            df = pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Failed to convert data to DataFrame: {str(e)}")
            raise ValueError(f"Invalid data format: {str(e)}")
    else:
        df = data
    
    # Generate a unique filename
    unique_id = uuid.uuid4().hex[:8]
    if suffix:
        file_name = f"{dataset_type}_{suffix}_{unique_id}.csv"
    else:
        file_name = f"{dataset_type}_{unique_id}.csv"
    
    # Save DataFrame to CSV
    df.to_csv(file_name, index=False)
    logger.info(f"Saved dataset with {len(df)} rows to {file_name}")
    
    return file_name

def merge_csv_files(file_list, output_filename):
    """
    Merge multiple CSV files into a single CSV file.
    
    Args:
        file_list: List of CSV file paths to merge
        output_filename: Path for the output file
        
    Returns:
        str: Path to the merged CSV file
    """
    if not file_list:
        logger.warning("No files provided to merge")
        return None
    
    # Read and combine all CSV files
    all_dfs = []
    for file in file_list:
        try:
            df = pd.read_csv(file)
            all_dfs.append(df)
            logger.info(f"Read {len(df)} rows from {file}")
        except Exception as e:
            logger.error(f"Error reading file {file}: {str(e)}")
    
    # Combine all DataFrames
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(output_filename, index=False)
        logger.info(f"Merged {len(all_dfs)} files into {output_filename} with {len(combined_df)} total rows")
        return output_filename
    else:
        logger.error("No valid DataFrames to merge")
        return None

def save_processing_stats(stats, filename):
    """
    Save processing statistics to a JSON file.
    
    Args:
        stats: Dictionary of statistics
        filename: Output filename
        
    Returns:
        str: Path to the saved file
    """
    try:
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved processing stats to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving stats to {filename}: {str(e)}")
        return None
