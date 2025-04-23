import io
import pandas as pd
import boto3
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
from numba import njit


@njit
def split_data_func(family_accession, class_encoded, test_ratio=0.33, dev_ratio=0.33):
    unique_classes = np.unique(family_accession)
    train_indices = []
    dev_indices = []
    test_indices = []

    for cls in unique_classes:
        print("Handling class")
        class_data_indices = np.where(family_accession == cls)[0]
        count = len(class_data_indices)

        if count == 1:
            test_indices.extend(class_data_indices)

        elif count == 2:
            dev_indices.extend(class_data_indices[:1])
            test_indices.extend(class_data_indices[1:])

        elif count == 3:
            train_indices.append(class_data_indices[0])
            dev_indices.append(class_data_indices[1])
            test_indices.append(class_data_indices[2])

        else:
            randomized_indices = np.random.permutation(class_data_indices)
            num_test = int(count * test_ratio)
            num_dev = int((count - num_test) * dev_ratio)

            test_part = randomized_indices[:num_test]
            dev_part = randomized_indices[num_test:num_test + num_dev]
            train_part = randomized_indices[num_test + num_dev:]

            train_indices.extend(train_part)
            dev_indices.extend(dev_part)
            test_indices.extend(test_part)

    print("After the split")

    return (np.array(train_indices, dtype=np.int64),np.array(dev_indices, dtype=np.int64),np.array(test_indices, dtype=np.int64))


def preprocess_to_staging(bucket_raw, bucket_staging, input_file, output_prefix):

    s3 = boto3.client('s3', endpoint_url='http://localhost:4566')
    response = s3.get_object(Bucket=bucket_raw, Key=input_file)
    data = pd.read_csv(io.BytesIO(response['Body'].read()))

    print("Missing values")
    data = data.dropna()

    print("Label Encoder starting")
    label_encoder = LabelEncoder()
    data['class_encoded'] = label_encoder.fit_transform(data['family_accession'])

    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    label_mapping_csv = pd.DataFrame(list(label_mapping.items()), columns=['family_accession', 'class_encoded'])
    csv_buffer = io.StringIO()
    label_mapping_csv.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket_staging,Key=f"{output_prefix}_label_mapping.csv",Body=csv_buffer.getvalue())

    print("Splitting data")
    family_accession = data['family_accession'].astype('category').cat.codes.values
    class_encoded = data['class_encoded'].values

    family_accession = np.array(family_accession)
    class_encoded = np.array(class_encoded)

    train_indices, dev_indices, test_indices = split_data_func(family_accession, class_encoded)

    train_data = data.iloc[train_indices]
    dev_data = data.iloc[dev_indices]
    test_data = data.iloc[test_indices]

    train_data = train_data.drop(columns=["family_id", "sequence_name", "family_accession"])
    dev_data = dev_data.drop(columns=["family_id", "sequence_name", "family_accession"])
    test_data = test_data.drop(columns=["family_id", "sequence_name", "family_accession"])

    for split_name, split_data in zip(['train', 'dev', 'test'], [train_data, dev_data, test_data]):
        csv_buffer = io.StringIO()
        split_data.to_csv(csv_buffer, index=False)
        s3.put_object(
            Bucket=bucket_staging,
            Key=f"{output_prefix}_{split_name}.csv",
            Body=csv_buffer.getvalue()
        )

    print("Class weights")
    class_counts = train_data['class_encoded'].value_counts()
    class_weights = 1. / class_counts
    class_weights /= class_weights.sum()

    min_weight = class_weights.max()
    weight_scaling_factor = 1 / min_weight
    class_weights *= weight_scaling_factor

    class_weights_dict = OrderedDict(sorted(class_weights.items()))
    class_weights_csv = pd.DataFrame(list(class_weights_dict.items()), columns=['class', 'weight'])
    csv_buffer = io.StringIO()
    class_weights_csv.to_csv(csv_buffer, index=False)
    s3.put_object(
        Bucket=bucket_staging,
        Key=f"{output_prefix}_class_weights.csv",
        Body=csv_buffer.getvalue()
    )
    print("End of the function")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess data from raw to staging bucket")
    parser.add_argument("--bucket_raw", type=str, required=True, help="Name of the raw S3 bucket")
    parser.add_argument("--bucket_staging", type=str, required=True, help="Name of the staging S3 bucket")
    parser.add_argument("--input_file", type=str, required=True, help="Name of the input file in raw bucket")
    parser.add_argument("--output_prefix", type=str, required=True, help="Prefix for output files in staging bucket")
    args = parser.parse_args()

    preprocess_to_staging(args.bucket_raw, args.bucket_staging, args.input_file, args.output_prefix)
