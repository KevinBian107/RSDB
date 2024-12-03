# responsible for data cleaning
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

    ## Filter data within latitude/longitude bounds
    lat_min, lat_max = 32.5, 42
    long_min, long_max = -124.4, -114.13
    df = df[
        (df["latitude"] >= lat_min)
        & (df["latitude"] <= lat_max)
        & (df["longitude"] >= long_min)
        & (df["longitude"] <= long_max)
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

    ## Filter data with review_time(unix) before 2005
    before_2005_timestamp = 1107849600 * 1000  # Convert to milliseconds
    df = df[df["review_time(unix)"] >= before_2005_timestamp]

    # merging
    # df = df.merge(meta_df, how='inner', on='gmap_id')
    df = df.merge(meta_df, how="inner", right_on="gmap_id", left_on="gmap_id")
    df = df.drop_duplicates(subset=["reviewer_id", "text", "gmap_id"])

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
    base_path = Path.cwd() / "rsdb"
    print(base_path)
    file_path = base_path / "data" / "data.json.gz"
    meta_file_path = base_path / "data" / "metadata.json.gz"

    ## Download data if not available
    if not file_path.exists():
        print(f"Review file not found. Downloading...")
        download_review_data(url, file_path)

    if not meta_file_path.exists():
        print(f"Metadata file not found. Downloading...")
        download_review_data(meta_url, meta_file_path)

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
        total_chunks = 300
        num_rows = 0

        ## Extract, Transform, Load
        with pd.read_json(
            file_path, compression="gzip", lines=True, chunksize=chunk_size
        ) as reader:
            ## batch processin
            with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
                for i, chunk in enumerate(reader):

                    # if i > total_chunks:
                    #     break
                    
                    # 5% of the data in each chunk
                    sampled_chunk = chunk.sample(frac=0.05, random_state=42)
                    
                    # Rename columns
                    chunk = sampled_chunk.rename(
                        columns={
                            "user_id": "reviewer_id",
                            "name": "reviewer_name",
                            "time": "review_time(unix)",
                        }
                    )

                    # Clean the chunk
                    chunk = clean_review_data(chunk, metadata_df)
                    num_rows += len(chunk)
                    results.append(chunk)

                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({"Processed Rows": num_rows})

        ## Combine all processed chunks into a single DataFrame
        clean_reviews = pd.concat(results, ignore_index=True)
        print(f"Processed {num_rows} review entries.")
    else:
        print(f"Review file not found: {file_path}. Please ensure the file exists.")

    return clean_reviews


def get_single_chunk(url: str, meta_url: str, chunk_size=10000):
    """
    Download data if not available and process only one chunk for testing.

    Args:
        url (str): URL of the review data of a single state.
        meta_url (str): URL of the metadata of a single state.
        chunk_size (int): Size of the chunk to process.

    Returns:
        DataFrame: A single processed chunk of review data.
    """

    ## Set up paths
    base_path = Path.cwd() / "rsdb"
    file_path = base_path / "data" / "data.json.gz"
    meta_file_path = base_path / "data" / "metadata.json.gz"

    ## Download data if not available
    if not file_path.exists():
        print(f"Review file not found. Downloading...")
        download_review_data(url, file_path)

    if not meta_file_path.exists():
        print(f"Metadata file not found. Downloading...")
        download_review_data(meta_url, meta_file_path)

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

    ## Process one chunk of review data
    if file_path.exists():
        print(f"Processing review data from: {file_path}")

        ## Extract, Transform, Load
        with pd.read_json(
            file_path, compression="gzip", lines=True, chunksize=chunk_size
        ) as reader:
            for chunk in reader:
                # Rename columns
                chunk = chunk.rename(
                    columns={
                        "user_id": "reviewer_id",
                        "name": "reviewer_name",
                        "time": "review_time(unix)",
                    }
                )

                # Clean the chunk
                chunk = clean_review_data(chunk, metadata_df)
                print(f"Processed {len(chunk)} entries in the first chunk.")
                return chunk
    else:
        print(f"Review file not found: {file_path}. Please ensure the file exists.")

    return None
