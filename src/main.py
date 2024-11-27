from data_preprocessing import *
from pathlib import Path
import pandas as pd
import json

def main():
    ## Set up paths
    base_path = Path.cwd().parent
    file_path = base_path / "data" / "data.json.gz"
    meta_file_path = base_path / "data" / "metadata.json.gz"
    url = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/review-California_10.json.gz"
    meta_url = "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/meta-California.json.gz"
    data_out_path = base_path /  "data" / "california_clean_data.json.gz"
    metadata_out_path = base_path / "data" / "california_clean_metadata.json.gz"

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
        metadata_df.to_json(metadata_out_path, orient='records', lines=True, compression='gzip')
        print(f"Loaded {len(metadata_df)} metadata entries.")
    else:
        print(f"Metadata file not found: {meta_file_path}. Please ensure the file exists.")
        return  


    ## Process review data in chunks
    if file_path.exists():
        print(f"Processing review data from: {file_path}")
        num_rows = 0
        chunk_size = 100000
        # results = []

        ## Extract, Transform, Load
        with pd.read_json(file_path, compression='gzip', lines=True, chunksize=chunk_size) as reader:
            with gzip.open(data_out_path, 'wt') as gz_out:
                ## batch processing
                for i, chunk in enumerate(reader):
                    ## Clean and process each chunk
                    chunk = chunk.rename(columns={
                        'user_id': 'reviewer_id',
                        'name': 'reviewer_name',
                        'time': 'review_time(unix)',
                    })
                    chunk = clean_review_data(chunk, metadata_df)

                    for row in chunk.to_dict(orient='records'):
                        gz_out.write(json.dumps(row) + '\n')
                    num_rows += chunk_size
                    print(f"Processed and saved chunk {i}")
                    #results.append(chunk)
        
        ## Combine all processed chunks into a single DataFrame
        # final_df = pd.concat(results, ignore_index=True)
        print(f"Processed {num_rows} review entries.")
    else:
        print(f"Review file not found: {file_path}. Please ensure the file exists.")

if __name__ == "__main__":
    main()
