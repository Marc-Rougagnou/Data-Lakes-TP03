import pandas as pd
from pathlib import Path
import boto3

print("Start the function")
def unpack_data(input_dir, output_file_name, bucket_name):

    base_path = Path(input_dir)
    frames = []
    CSV_COLUMNS = ['sequence', 'family_accession', 'sequence_name', 'aligned_sequence', 'family_id']
    print("Inside the function")

    for split in ['train', 'test', 'dev']:
        current_dir = base_path / split
        if current_dir.is_dir():
            files = sorted(current_dir.glob("*"))
            for fil in files:
                if fil.is_file():
                    table = pd.read_csv(fil, names=CSV_COLUMNS)
                    frames.append(table)

    print("After the travel into the folders")
    
    if frames:
        final_data = pd.concat(frames, ignore_index=True)
        temp_path = Path("/tmp") / output_file_name
        final_data.to_csv(temp_path, index=False)

        s3 = boto3.client("s3", endpoint_url="http://localhost:4566")
        s3.upload_file(str(temp_path), bucket_name, output_file_name)
    else:
        print("No data")

    print("End of the function")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fusionne des fichiers CSV de sous-dossiers puis les envoie sur S3 LocalStack.")
    parser.add_argument("--input_dir", required=True, help="Chemin vers le dossier contenant train/test/dev")
    parser.add_argument("--output_file_name", required=True, help="Nom du fichier de sortie")
    parser.add_argument("--bucket_name", required=True, help="Nom du bucket S3 LocalStack")
    
    args = parser.parse_args()

    unpack_data(args.input_dir, args.output_file_name, args.bucket_name)
