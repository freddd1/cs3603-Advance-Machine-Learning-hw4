# Will hold all the datasets creations and manipultions

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def create_folds(df: pd.DataFrame, label_name: str, num_folds: int, seed: int = 11):
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