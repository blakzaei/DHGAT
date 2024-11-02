#-- IMPORT --------------------------------------------------------------------
import pandas as pd
import networkx as nx
import numpy as np
import os

from utils.util import create_folder
###############################################################################

#-- Initiliaze ----------------------------------------------------------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(base_dir, 'data')
df_file = os.path.join(data_dir, 'liar_df_fasttext_embeddings.csv')

valid_features = ['speaker', 'party_affiliation',
                  'subject', 'state_info', 'job_title', 'context']
#-------------------------------------------------------------------------------

#-- function to create feature matrix -------------------------------------------
def create_feature_matrix(feature, df):
    feature_dict = {feature_name: idx for idx, feature_name in enumerate(df[feature].unique())}
    feature_indices = df[feature].map(feature_dict).values
    feature_matrix = (feature_indices[:, np.newaxis] == feature_indices[np.newaxis, :]).astype(int)

    return feature_matrix
#-------------------------------------------------------------------------------

#-- function to create graph ----------------------------------------------------
def creat_graph(g_features):

    for feature in g_features:
        if feature not in valid_features:
            print(f'{feature} is not a valid feature.'
                  f'valid features = {valid_features}')
            return

    print(f'Creating Graph using features {g_features} ...')

    #-- create graphs folder --
    result_dir = create_folder(os.path.join(data_dir, 'graphs'))

    #-- load data --
    df = pd.read_csv(df_file)

    #-- create graph for all features --
    for feature in g_features:

        print(f'Creating Graph using features {feature} ...')

        #-- create feature matrix --
        feature_matrix = create_feature_matrix(feature, df)

        #-- create empty graph --
        G = nx.Graph()
        G.add_nodes_from(df.index)

        #-- add edges --
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                if feature_matrix[i,j]==1:
                    G.add_edge(i, j)

        print(f'G_{feature}:'
              f'\tNumber of nodes:", {nx.number_of_nodes(G)}'
              f'\tNumber of edges:", {nx.number_of_edges(G)}')

        #-- save graph --
        print(f'Saving G_{feature} ...')
        nx.write_graphml(G, os.path.join(result_dir, f'G_{feature}.graphml'))

    print('Creating Graph: Done :)\n')
#------------------------------------------------------------------------------


