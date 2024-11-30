# parsing function that will parse data
import pandas as pd
import numpy as np
import gzip
from pathlib import Path
import json
import requests
import os
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")


def download_review_data(url: str, save_path: str) -> None:
    """
    Downloads the review dataset and saves it to a specified file path.

    Args:
        url (str): The URL of the dataset to be downloaded.
        save_path (str): The file path where the downloaded dataset will be saved.

    Returns:
        None
    """
    try:
        # Ensure the directory exist or create one
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_path):
            os.makedirs(save_dir, exist_ok=True)

        # Send request to url
        session = requests.Session()
        response = session.get(url, stream=True)
        response.raise_for_status()

        # Write the content to a file (assuming it's gzipped)
        with open(save_path, "wb") as f:
            # Use stream=True to download in chunks and save it
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Data downloaded successfully and saved to: {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")


def parseData(file_path):
    """
    Parses a gzipped JSON file

    Args:
        file_path (str): The file path

    Returns:
        list[dict]: A list of dictionaries
    """

    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def clean_gmap_meta_data(df):
    """
    Cleans and processes a DataFrame containing review metadata.

    This function performs the following operations:
    - Renames the 'name' column to 'gmap_name'
    - Reorders the columns
    - Removes the 'state' and 'url' column
    - Handles missing values and fills other missing values with 'NaN'.
    - Filters out entries based on frequency of 'category'

    Args:
        df (DataFrame): A DataFrame containing review metadata

    Returns:
        DataFrame: The cleaned DataFrame
    """
    df = df.rename(columns={"name": "gmap_name"})
    ## reorder column to make them more readable
    df = df[
        [
            "gmap_id",
            "gmap_name",
            "address",
            "latitude",
            "longitude",
            "description",
            "category",
            "avg_rating",
            "num_of_reviews",
            "price",
            "hours",
            "state",
            "MISC",
            "relative_results",
            "url",
        ]
    ]

    ## remove state col: avoid multicolumnity
    ## remove url col: inrevelant col
    df = df.drop(columns=["state", "url"])

    ## missingness handlation
    df = df.dropna(subset=["address", "category", "gmap_name", "relative_results"])
    df = df.fillna(np.nan)

    ## remove gmap_id that has categories that appear less than 20 times
    categories = df["category"].explode().value_counts()
    spare_categories = categories[categories <= 10].index

    df = df[
        df["category"].apply(
            lambda categories: not any(
                [cate in spare_categories for cate in categories]
            )
        )
    ]

    return df


def clean_review_data(df, meta_df):
    """
    Cleans and processes a DataFrame containing review data.

    This function performs the following operations:
    - Removes the 'pics' column
    - Adds the 'has_rep' column
    - Merges the DataFrame with metadata

    Args:
        df (DataFrame): A DataFrame containing review data

    Returns:
        DataFrame: The cleaned DataFrame
    """
    ## remove pics column: since we don't want to deal with image
    df = df.drop(columns=["pics"])

    ## adding feature
    df = df.assign(has_rep=df["resp"].notna())

    # merging
    df = df.merge(meta_df, how="inner", right_on="gmap_id", left_on="gmap_id")

    return df


def get_clean_review_data(url: str, meta_url: str, chunk_size=100000, export=False):
    """
    take in data url and export the clean data set

    Args:
        url: url of the review data of a single state
        meta_url: url of the meta data of a single state
        chunk_size: specify the chunk size of the processing data
        export: whether we need to export the data or not (in progress)

    Returns:
        DataFrame: cleaned review data set

    """
    ## Set up paths
    base_path = Path.cwd()
    file_path = base_path / "data" / "california_clean_data.json.gz"
    meta_file_path = base_path / "data" / "california_clean_metadata.json.gz"
    

    ## Download data if not available
    # if not file_path.exists():
    #     print(f"Review file not found. Downloading...")
    #     download_review_data(url, file_path)

    # if not meta_file_path.exists():
    #     print(f"Metadata file not found. Downloading...")
    #     download_review_data(meta_url, meta_file_path)

    ## Process meta data
    if meta_file_path.exists():
        print(f"Loading metadata from: {meta_file_path}")
        metadata = parseData(meta_file_path)
        metadata_df = pd.DataFrame(metadata)
        metadata_df = clean_gmap_meta_data(metadata_df)
        print(f"Loaded {len(metadata_df)} metadata entries.")
    else:
        print(
            f"Metadata file not found: {meta_file_path}. Please ensure the file exists."
        )
        return

    ## Process review data in chunks
    if file_path.exists():
        print(f"Processing review data from: {file_path}")
        results = []
        num_rows = 0

        ## Extract, Transform, Load
        with pd.read_json(
            file_path, compression="gzip", lines=True, chunksize=chunk_size
        ) as reader:
            ## batch processing
            for i, chunk in enumerate(reader):
                ## Clean and process each chunk
                chunk = chunk.rename(
                    columns={
                        "user_id": "reviewer_id",
                        "name": "reviewer_name",
                        "time": "review_time(unix)",
                    }
                )
                chunk = clean_review_data(chunk, metadata_df)
                num_rows += 1
                print(f"Processed and saved chunk {i}")
                results.append(chunk)

        ## Combine all processed chunks into a single DataFrame
        clean_reviews = pd.concat(results, ignore_index=True)
        print(f"Processed {num_rows} review entries.")
    else:
        print(f"Review file not found: {file_path}. Please ensure the file exists.")

    return clean_reviews
