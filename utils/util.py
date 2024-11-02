#-- IMPORT --------------------------------------------------------------------
import os

import pandas as pd
import numpy as np

import torch
###############################################################################

#-- Function to Create Folders -------------------------------------------------
def create_folder(name=""):
    parent_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    new_directory_path = os.path.join(parent_folder_path, name)    
    
    if not os.path.exists(new_directory_path):
        os.makedirs(new_directory_path)

    return new_directory_path
#------------------------------------------------------------------------------

#-- Function to Create an Empty Dataframe for saving results -------------------
def create_empty_df_for_results(results_file):
    cols_names = ['itr',
                  'percents',
                  'test_acc',
                  'test_macro_f1',
                  'test_micro_f1',
                  'test_macro_prec',
                  'test_micro_prec',
                  'test_macro_rec',
                  'test_micro_rec',
                  'train_acc',
                  'train_macro_f1',
                  'train_micro_f1',
                  'train_macro_prec',
                  'train_micro_prec',
                  'train_macro_rec',
                  'train_micro_rec']
    
    df_results = pd.DataFrame(columns=cols_names)
    df_results.to_csv(results_file, index=False)
#--------------------------------------------------------------------------





