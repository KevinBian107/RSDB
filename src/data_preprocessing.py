# this scripts with downlaod data from the website and parse it for machine learning algorithm
import pandas as pd
import numpy as np
import gzip
from pathlib import Path
import json

import warnings
warnings.filterwarnings("ignore")

def download_review_data(url, save_path):
    try:
        # Send GET request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if the request was successful

        # Write the content to a file (assuming it's gzipped)
        with open(save_path, 'wb') as f:
            # Use stream=True to download in chunks and save it
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Data downloaded successfully and saved to: {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")


def parseData(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


def clean_gmap_meta_data(df):
    df = df.rename(columns={
    'name': 'gmap_name'
    })
    ## reorder column to make them more readable
    df = df[[
    'gmap_id',
    'gmap_name',
    'address',
    'latitude',
    'longitude',
    'description',
    'category',
    'avg_rating',
    'num_of_reviews',
    'price',
    'hours',
    'state',
    'MISC',
    'relative_results',
    'url'
    ]]

    ## remove state col: avoid multicolumnity
    ## remove url col: inrevelant col
    df = df.drop(columns=['state', 'url'])

    ## missingness handlation
    df = df.dropna(subset=['address', 'category', 'gmap_name', 'relative_results'])
    df = df.fillna(np.nan)
    
    ## remove gmap_id that has categories that appear less than 20 times
    categories = df['category'].explode().value_counts()
    spare_categories = categories[categories <= 10].index
    
    df = df[df['category'].apply(lambda categories: not any([cate in spare_categories for cate in categories]))]


def clean_review_data(df): 
    ## remove pics column: since we don't want to deal with image
    df = df.drop(columns=['pics'])

    ## adding feature
    df = df.assign(has_rep = df['resp'].notna())

    # merging
    df = df.merge(
        df,
        how = 'inner',
        right_on = 'gmap_id',
        left_on = 'gmap_id'
    )
    


