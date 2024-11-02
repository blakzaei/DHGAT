#-- Import -------------------------------------------------------------------------------------
import torch
import torch.nn.functional as F

from models.dhgat_layer import DHGAT_Layer
#-----------------------------------------------------------------------------------------------

class DHGAT_NET(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, decision_size, decision_key):
        super().__init__()

        self.dh_layer_1 = DHGAT_Layer(input_size, hidden_size, decision_size, decision_key)
        self.dh_layer_2 = DHGAT_Layer(hidden_size, hidden_size // 2, decision_size, decision_key)

        self.fc1 = torch.nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc2 = torch.nn.Linear(hidden_size // 4, output_size)

    def forward(self, x, edge_index_dict):
        x = self.dh_layer_1(x, edge_index_dict).relu()
        x = self.dh_layer_2(x, edge_index_dict).relu()

        x = self.fc1(x).relu()
        x = self.fc2(x)
        return F.softmax(x, dim=1)
# -----------------------------------------------------------------------------------------------