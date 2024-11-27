import data_preprocessing
from pathlib import Path
import pandas as pd


def main():
    ## Set up paths
    base_path = Path.cwd().parent
    file_path = base_path / "data" / "data.json.gz"
    meta_file_path = base_path / "data" / "metadata.json.gz"
    url = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/review-California_10.json.gz"
    meta_url = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/meta-California.json.gz"

    ## Download data if not available
    if not file_path.exists():
        print(f"Review file not found. Downloading...")
        data_preprocessing.download_review_data(url, file_path)
    
    if not meta_file_path.exists():
        print(f"Metadata file not found. Downloading...")
        data_preprocessing.download_review_data(meta_url, meta_file_path)

    ## Process meta data 
    if meta_file_path.exists():
        print(f"Loading metadata from: {meta_file_path}")
        metadata = data_preprocessing.parseData(meta_file_path)
        metadata_df = pd.DataFrame(metadata)
        data_preprocessing.clean_gmap_meta_data(metadata_df)
        print(f"Loaded {len(metadata_df)} metadata entries.")
    else:
        print(f"Metadata file not found: {meta_file_path}. Please ensure the file exists.")
        return  


    ## Process review data in chunks
    if file_path.exists():
        print(f"Processing review data from: {file_path}")
        results = []

        with pd.read_json(file_path, compression='gzip', lines=True, chunksize=1000000) as reader:
            for chunk in reader:
                ## Clean and process each chunk
                chunk = chunk.rename(columns={
                    'user_id': 'reviewer_id',
                    'name': 'reviewer_name',
                    'time': 'review_time(unix)',
                })
                data_preprocessing.clean_review_data(chunk)
                results.append(chunk)
        ## Combine all processed chunks into a single DataFrame
        final_df = pd.concat(results, ignore_index=True)
        print(f"Processed {len(final_df)} review entries.")
    else:
        print(f"Review file not found: {file_path}. Please ensure the file exists.")

if __name__ == "__main__":
    main()
