#-- IMPORT --------------------------------------------------------------------
from datasets import load_dataset
import pandas as pd
import os

from utils.util import create_folder
###############################################################################

#-- Function to load and prepare LIAR ds --------------------------------------
def load_and_prepare_LIAR():
    ds_dir = create_folder('data/')
    ds_file = os.path.join(ds_dir, 'liar_df.csv')

    if os.path.isfile(ds_file):
        print("LIAR dataset already exists, skipping download :)\n")
        return

    #-- load ds --
    print('loading LIAR dataset ...')
    ds = load_dataset("liar", trust_remote_code=True)

    ds_train = ds['train']
    ds_val = ds['validation']
    ds_test = ds['test']

    #-- covert ds to df --
    df_train = ds_train.to_pandas()
    df_val = ds_val.to_pandas()
    df_test = ds_test.to_pandas()

    #-- add new column: df_type --
    df_train_copy = df_train.copy()
    df_val_copy = df_val.copy()
    df_test_copy = df_test.copy()

    df_train_copy['df_type'] = 'train'
    df_val_copy['df_type'] = 'val'
    df_test_copy['df_type'] = 'test'

    #-- merge train, val, and test in one df --
    print('Merging Train, Val and Test in one ds file ...')
    df_full = pd.concat([df_train_copy, df_val_copy, df_test_copy], axis=0)
    df_full.reset_index(drop=True, inplace=True)
    print(f'merged df shape: {df_full.shape}')


    #-- Shuffle df --
    df_full = df_full.sample(frac=1).reset_index(drop=True)

    #-- save --
    print('Saving LIAR ds to CSV file ...')
    df_full.to_csv(ds_file, sep=',', encoding='utf-8', index=False)

    print('load_and_prepare_LIAR: DONE :)\n')
#------------------------------------------------------------------------------
