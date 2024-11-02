#-- IMPORT --------------------------------------------------------------------
import torch
import pandas as pd
import random
import os

from utils.util import create_folder
###############################################################################

#-- Initiliaze ----------------------------------------------------------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(base_dir, 'data')
df_file = os.path.join(data_dir, 'liar_df_fasttext_embeddings.csv')
#------------------------------------------------------------------------------

#-- Function to split data to train and test -----------------------------------
def split(train_percent, number_of_iterations):
    # -- log --
    print(f'Spliting Data to {int(train_percent*100)}% as labeled and {int((1-train_percent)*100)}% as unlabeled ...')

    #-- create split folder --
    result_dir = create_folder(os.path.join(data_dir, 'split'))

    #-- load data --
    df = pd.read_csv(df_file)

    #-- get trained data in mian LIAR ds --
    main_ds_train_indexes = df[df['df_type'] == 'train'].index.tolist()       

    num_of_nodes = len(df)
    num_of_train_nodes = int(train_percent * num_of_nodes)
    print(f'number of all nodes:{num_of_nodes}\nnumber of train nodes:{num_of_train_nodes}')

    #-- select train_percent% of nodes randomly --
    for itr in range(1, number_of_iterations + 1):
        train_indices = torch.tensor(random.sample(main_ds_train_indexes, num_of_train_nodes))
        torch.save(train_indices, os.path.join(result_dir, f'train_indices_{train_percent}_{itr}.pth'))

    print('Spliting Data: DONE :)\n')
#------------------------------------------------------------------------------


