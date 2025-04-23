import io
import pandas as pd
import boto3
from transformers import AutoTokenizer


def tokenize_sequences(bucket_staging, bucket_curated, input_file, output_file, model_name="facebook/esm2_t6_8M_UR50D"):
    s3 = boto3.client("s3", endpoint_url="http://localhost:4566")

    print(f"Recuperation")
    obj = s3.get_object(Bucket=bucket_staging, Key=input_file)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))

    print("Load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Tokenisation")
    input_ids = []
    for seq in df["sequence"]:
        encoded = tokenizer(seq, padding="max_length", truncation=True, max_length=1024, return_tensors="np")
        input_ids.append(encoded["input_ids"][0])
    token_df = pd.DataFrame(input_ids)
    token_df.columns = [f"token_{i}" for i in range(token_df.shape[1])]

    print("Fusion")
    df_meta = df.drop(columns=["sequence"])
    final_df = pd.concat([df_meta, token_df], axis=1)

    path_local = f"/tmp/{output_file}"
    final_df.to_csv(path_local, index=False)
    print("File saved")

    print("Upload")
    with open(path_local, "rb") as file_obj:
        s3.upload_fileobj(file_obj, bucket_curated, output_file)
    print("End of the function")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tokenisation des sequences du staging vers curated")
    parser.add_argument("--bucket_staging", type=str, required=True)
    parser.add_argument("--bucket_curated", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D")
    args = parser.parse_args()
    tokenize_sequences(args.bucket_staging,args.bucket_curated,args.input_file,args.output_file,args.model_name)
