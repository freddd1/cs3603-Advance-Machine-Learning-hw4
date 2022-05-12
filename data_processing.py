# Will hold all the datasets creations and manipultions

import re
import os
import pandas as pd
import datasets
from datasets import load_dataset
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader


def create_folds(df: pd.DataFrame, label_name: str, num_folds: int, seed: int = 11) -> pd.DataFrame:
    """
    The function will recive df, create StratifiedKfold and return df with new columns
    `folds` that will hold the fold number
    """
    print(f'input df shape : {df.shape}')

    skf = StratifiedKFold(n_splits=num_folds, random_state=seed, shuffle=True)
    
    df_f = df.copy()
    df_f['fold'] = -1
    df_f.reset_index(drop=True, inplace=True)
    df_f.dropna(inplace=True)

    print(f'Number of folds: {num_folds}, total samples (after removing NaN): {len(df_f)}')
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=df_f.drop([label_name], axis=1), y=df_f[label_name])):
        print(f'fold: {fold}, num samples: {len(val_idx)}')
        df_f.loc[val_idx, 'fold'] = fold
    
    return df_f


def preprocess(txt: str) -> str:
    """
    All the preprocessing of the text
    """

    # remove urls from tweet
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    txt = re.sub(url_regex, '', txt)

    return txt


def create_datasetdict(df:pd.DataFrame, fold: int = None) -> datasets.dataset_dict.DatasetDict:
    """
    The function creates DatasetDict according to the fold.
    It temporery save the files in `data` folder and delets it.
    """
    # Helper function that will help us remove files.
    def remove_file(filename: str):
        if os.path.isfile(filename):
            os.remove(filename)

    df_f = df.copy()

    train_filename = 'data/train_temp.csv'
    test_filename = 'data/test_temp.csv'


    remove_file(train_filename)
    remove_file(test_filename)

    if fold:
        assert df.fold.max() >= fold, f'fold: {fold} is out of range. max fold is: {df.fold.max()}'
            
        df_f.dropna(inplace=True)

        # split to train and test
        train, test = df_f[df_f.fold != fold], df_f[df_f.fold == fold]
        train = train.dropna()
        test = test.dropna()

        # temporery save them to for easy loading
        train[['tweet', 'label']].to_csv(train_filename, index=False)
        test[['tweet', 'label']].to_csv(test_filename, index=False)

        # loading
        dataset = load_dataset('csv', data_files={'train': train_filename, 'test': test_filename})

        remove_file(train_filename)
        remove_file(test_filename)
    
    else:  # Load the whole dataset as train.
        train = df_f.dropna()
        train[['tweet', 'label']].to_csv(train_filename, index=False)
        dataset = load_dataset('csv', data_files={'train': train_filename})
        remove_file(train_filename)


    return dataset

def create_dataloader_from_tokenized(data  sets: datasets.dataset_dict.DatasetDict) -> (DataLoader, DataLoader):
    datasets = datasets.remove_columns(["tweet"])
    datasets = datasets.rename_column("label", "labels")
    datasets.set_format("torch")
    
    train_dataloader = DataLoader(datasets['train'], shuffle=True, batch_size=8)
    eval_dataloader = None

    if len(datasets.keys()) == 2:
        # If the dataset has test and train
        eval_dataloader = DataLoader(datasets['test'], batch_size=8)

    return train_dataloader, eval_dataloader
    
    
