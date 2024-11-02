#-- Import -------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F

import networkx as nx

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from IPython import display

import copy
import random
import os

from models import dhgat_net
from utils.util import create_empty_df_for_results
from utils.util import create_folder
#-----------------------------------------------------------------------------------------------


class DHGAT():
    def __init__(self, g_features, decision_feature='speaker', num_iterations=1,
                 num_epochs=100, train_percent=0.3, l1=0.5, l2=0.5):

        #-- log --
        print('Running DHGAT on LIAR ...')

        self.g_features = g_features
        self.decision_feature = decision_feature
        self.num_iterations = num_iterations
        self.num_epochs = num_epochs
        self.train_percent = train_percent
        self.l1 = l1
        self.l2 = l2

        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.df_file = os.path.join(self.data_dir, 'liar_df_fasttext_embeddings.csv')
        self.graphs_dir = os.path.join(self.data_dir, 'graphs')
        self.train_indices_dir = os.path.join(self.data_dir, 'split')
        self.result_dir = create_folder(os.path.join(self.base_dir, 'results'))
        self.results_file = os.path.join(self.result_dir, 'results.csv')
        create_empty_df_for_results(self.results_file)
        self.df_results = pd.read_csv(self.results_file)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'device = {self.device}')

        self.df = None
        self.features = None
        self.edge_index_dict = {}
        self.decision_key = None
        self.model = None
        self.criterion = None
        self.optimizer = None

        self.load_data()
        self.modify_labels()

        self.x = torch.tensor(self.df[self.features].values, dtype=torch.float32)
        self.y = torch.tensor(self.df['label'].values, dtype=torch.long)
        print(f'X:{self.x.shape} - Y:{self.y.shape}')

        self.create_edge_index_dict()
        self.decision_size = len(self.edge_index_dict)

        self.run()





    #-- function to load data --
    def load_data(self):

        #-- log--
        print('loading data ...')

        self.df = pd.read_csv(self.df_file)

        #-- Convert `txt_embedding` from string to list (vector) --
        if not isinstance(self.df.at[0, 'txt_embedding'], list):
            self.df['txt_embedding'] = self.df['txt_embedding'].apply(lambda x: list(map(float, x.strip('[]').split())))

        #-- Split values of `txt_embedding` vector as separate feature values --
        text_emb_size = len(self.df.at[0, 'txt_embedding'])
        text_cols_names = [f'text_emb_{i}' for i in range(text_emb_size)]

        #-- Create a DataFrame from `txt_embedding` and concatenate it to `self.df` --
        text_embeddings_df = self.df['txt_embedding'].apply(pd.Series)
        text_embeddings_df.columns = text_cols_names
        self.df = pd.concat([self.df, text_embeddings_df], axis=1)

        self.features = ['barely_true_counts', 'false_counts',
                         'half_true_counts', 'mostly_true_counts',
                         'pants_on_fire_counts'] + text_cols_names

        #-- log --
        print(f'data loaded: {self.df.shape}')

    #-- function to modify labels on liar ds --
    def modify_labels(self):

        #-- log --
        print('modifying labels ...')

        self.df['label'] = self.df['label'] * -1

        self.df['label'] = self.df['label'].replace(-5, 1)
        self.df['label'] = self.df['label'].replace(0, 2)
        self.df['label'] = self.df['label'].replace(-4, 3)
        self.df['label'] = self.df['label'].replace(-1, 4)
        self.df['label'] = self.df['label'].replace(-2, 5)
        self.df['label'] = self.df['label'].replace(-3, 6)

        self.df['label'] = self.df['label'] - 1

    #-- function to create edge_index dict from graphs --
    def create_edge_index_dict(self):
        #-- log --
        print('Creating Edge Indexes for all graphs ...')

        self.decision_key = 1 #-- default --
        key_number = 0
        for feature in self.g_features:
            g = nx.read_graphml(os.path.join(self.graphs_dir, f'G_{feature}.graphml'))
            g = nx.convert_node_labels_to_integers(g)

            if key_number == 0:
                # -- add self loop --
                edges = []
                for u in g.nodes():
                    edges.append((u, u))

                edge_index = torch.tensor(edges, dtype=torch.long)
                edge_index = edge_index.t().contiguous()
                edge_index = edge_index.to(self.device)
                self.edge_index_dict[key_number] = edge_index.to(self.device)
                key_number += 1

            edge_list = list(g.edges)
            edge_index = torch.tensor(edge_list, dtype=torch.long)
            edge_index = edge_index.t().contiguous()
            self.edge_index_dict[key_number] = edge_index.to(self.device)

            if feature==self.decision_feature:
                self.decision_key = key_number

            key_number += 1


    # -- create and initialize model --
    def create_model(self):

        self.model = dhgat_net.DHGAT_NET(input_size=self.x.size(-1),
                                         hidden_size=512,
                                         output_size=6,
                                         decision_size=self.decision_size,
                                         decision_key=self.decision_key).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-9)

    # -- function to load train indices and create train and test masks --
    def create_masks(self, iteration):
        num_of_nodes = self.x.shape[0]
        train_mask = torch.full((num_of_nodes,), False, dtype=torch.bool)
        train_indices_file = os.path.join(self.train_indices_dir, f'train_indices_{self.train_percent}_{iteration}.pth')
        train_indices = torch.load(train_indices_file)
        train_mask[train_indices] = True
        test_mask = ~train_mask

        return train_mask, test_mask

    #-- custom loss function for training --
    def custom_loss(self, y_pred, y_true, train_mask):

        y_pred = y_pred.float()
        y_true = y_true.float()

        loss = torch.mean(torch.abs(y_pred[train_mask].to(self.device) - y_true[train_mask].to(self.device)))
        loss.requires_grad_(True)

        return loss

    #-- function to train model --
    def train(self, train_mask):

        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(self.x.to(self.device), self.edge_index_dict)

        loss_1 = self.criterion(output[train_mask], self.y[train_mask].to(self.device))

        predictions = output.argmax(dim=1)
        loss_2 = self.custom_loss(predictions, self.y, train_mask)

        loss = self.l1 * loss_1 + self.l2 * loss_2

        accuracy = (predictions[train_mask] == self.y[train_mask].to(self.device)).sum().item() / len(self.y[train_mask])

        loss.backward()
        self.optimizer.step()

        return loss, accuracy, predictions

    #-- function to evaluate model --
    def evaluate(self, test_mask, model):
        model.eval()
        with torch.no_grad():
            output = model(self.x.to(self.device), self.edge_index_dict)
            predictions = output.argmax(dim=1)
            loss = self.criterion(output[test_mask], self.y[test_mask].to(self.device))
            # loss = self.criterion(output[test_mask].to(torch.float32).to(self.device),
            #                       self.y[test_mask].to(torch.float32).to(self.device))

            accuracy = (predictions[test_mask] == self.y[test_mask].to(self.device)).sum().item() / len(self.y[test_mask])

        return loss, accuracy, predictions

    #-- function to calculate all metrics --
    def get_evaluation_scores(self, y_pred, y_true):
        y_pred = y_pred.cpu()
        y_true = y_true.cpu()
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        precision_macro = precision_score(y_true, y_pred, average='macro')
        precision_micro = precision_score(y_true, y_pred, average='micro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        recall_micro = recall_score(y_true, y_pred, average='micro')

        return accuracy, f1_macro, f1_micro, precision_macro, precision_micro, recall_macro, recall_micro


    def run(self):

        test_accs = []
        for itr in range(1, self.num_iterations + 1):
            print(f'\nIteration {itr} ............................................')

            # -- load train indiceis and create train and test masks --
            train_mask, test_mask = self.create_masks(itr)

            #-- create and initialize DHGAT_NET model --
            self.create_model()

            best_acc = 0
            best_model = None
            best_acc_train = 0
            best_preds_train = None
            best_pred_test = None

            # -- train and evaluate model --
            for epoch in range(1, self.num_epochs + 1):
                train_loss, train_acc, preds_train = self.train(train_mask)
                val_loss, val_acc, preds_test = self.evaluate(test_mask, self.model)

                # -- select best model --
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model = copy.deepcopy(self.model)
                    best_preds_train = preds_train
                    best_pred_test = preds_test

                #if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
                log = f'Epoch {epoch}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Train ACC: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val ACC: {val_acc:.4f}'
                print(log)

            # -- get best Acc after training finished --
            print(f'BEST ACC: {best_acc}')
            test_loss, test_acc, preds = self.evaluate(test_mask, best_model)
            test_accs.append(test_acc)
            print(f"Test Accuracy: {test_acc}")

            # -- evaluation metrics on train and test data --
            tr_acc, tr_f1_mac, tr_f1_mic, tr_pr_mac, tr_pr_mic, \
                tr_rec_mac, tr_rec_mic = self.get_evaluation_scores(y_pred=best_preds_train[train_mask],
                                                                    y_true=self.y[train_mask])
            ts_acc, ts_f1_mac, ts_f1_mic, ts_pr_mac, ts_pr_mic, \
                ts_rec_mac, ts_rec_mic = self.get_evaluation_scores(y_pred=preds[test_mask],
                                                                    y_true=self.y[test_mask])

            # -- save evaluation metrics --
            new_row = {'itr': itr,
                       'percents': self.train_percent,
                       'test_acc': ts_acc,
                       'test_macro_f1': ts_f1_mac,
                       'test_micro_f1': ts_f1_mic,
                       'test_macro_prec': ts_pr_mac,
                       'test_micro_prec': ts_pr_mic,
                       'test_macro_rec': ts_rec_mac,
                       'test_micro_rec': ts_rec_mic,
                       'train_acc': tr_acc,
                       'train_macro_f1': tr_f1_mac,
                       'train_micro_f1': tr_f1_mic,
                       'train_macro_prec': tr_pr_mac,
                       'train_micro_prec': tr_pr_mic,
                       'train_macro_rec': tr_rec_mac,
                       'train_micro_rec': tr_rec_mic}
            new_row_df = pd.DataFrame([new_row])
            self.df_results = pd.concat([self.df_results, new_row_df], ignore_index=True)


        # -- log --
        print('finished all iterations :)\n')

        #-- get everage test acc --
        print('AVG TEST ACC = ', np.round(sum(test_accs) / len(test_accs), 4))


        # -- save df results --
        self.df_results.to_csv(self.results_file, index=False)

        print('\nRunning DHGAT on LIAR: DONE :)\n')









