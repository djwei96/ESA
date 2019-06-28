import torch
import torch.nn as nn
import torch.nn.functional as F

class ESA(nn.Module):
    def __init__(self, pred2ix_size, pred_embedding_dim, transE_dim, hidden_size, device):
        super(ESA, self).__init__()
        self.pred2ix_size = pred2ix_size
        self.pred_embedding_dim = pred_embedding_dim
        self.transE_dim = transE_dim
        self.input_size = self.transE_dim + self.pred_embedding_dim
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.pred2ix_size, self.pred_embedding_dim)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, bidirectional=True)
        self.device = device
        self.hidden = self._init_hidden()

    def forward(self, input_tensor):
        # bi-lstm
        pred_embedded = self.embedding(input_tensor[0])
        obj_embedded = input_tensor[1]
        embedded = torch.cat((pred_embedded, obj_embedded), 2)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded, self.hidden)

        # attention
        lstm_out = lstm_out.permute(1, 0, 2)
        hidden_state = hidden_state.view(1, -1, 1)
        atten_weight = torch.bmm(lstm_out, hidden_state)
        atten_weight = F.softmax(atten_weight.squeeze(2), dim=1)
        return atten_weight.view(-1, 1)

    def _init_hidden(self):
        return (torch.randn(2, 1, self.hidden_size, device=self.device), 
            torch.randn(2, 1, self.hidden_size, device=self.device))