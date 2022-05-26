# import json
# import os
# import cotools
# import shutil
import pandas as pd

DATASET_PATH = 'q12/data/'
METADATA_PATH = 'q12/data/metadata.csv'


def load_metadata(metadata_path: str = METADATA_PATH, num_of_papers: int = 20000) -> pd.DataFrame:
    """
    Load Initial Amount of metadata papers.
    :param metadata_path: Path to metadata.
    :param num_of_papers: Initial # of metadata.
    :return: df contains num_of_papers metadata.
    """
    metadata = pd.read_csv(metadata_path)
    metadata.sort_values('publish_time', ascending=False, inplace=True)
    metadata.dropna(subset=['sha', 'title', 'abstract'], inplace=True)
    metadata = metadata[metadata['publish_time'] < '2022-04-15']
    metadata = metadata[metadata['sha'].map(len) == 40]
    metadata = metadata[metadata['pdf_json_files'].map(len) == 70]
    df = metadata[0:num_of_papers]
    df.set_index('sha', inplace=True)
    return df


