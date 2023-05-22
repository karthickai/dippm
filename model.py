import pytorch_lightning as pl
import torch
import math
import torch.nn.functional as F
from torchmetrics import MeanAbsolutePercentageError
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from torch_scatter import scatter

def init_tensor(tensor, init_type):
    if tensor is None or init_type is None:
        return
    if init_type =='thomas':
        size = tensor.size(-1)
        stdv = 1. / math.sqrt(size)
        nn.init.uniform_(tensor, -stdv, stdv)
    else:
        raise ValueError(f'Unknown initialization type: {init_type}')



class Model(pl.LightningModule):
    def __init__(
        self,
        num_node_features=32,
        gnn_hidden=512,
        fc_hidden=512,
        reduce_func="sum",
        norm_sf=False,
        model_type="GraphSAGE",
    ):
        super().__init__()
        sf_hidden = 5
        self.type = model_type # type: ignore
        if not self.type == "MLP":
            self.reduce_func = reduce_func
            self.num_node_features = num_node_features
            self.norm_sf = norm_sf
            if self.type == "GraphSAGE":
                self.gnn_layer_func = tgnn.GraphSAGE
            elif self.type == "GAT":
                self.gnn_layer_func = tgnn.GAT
            elif self.type == "GCN":
                self.gnn_layer_func = tgnn.GCN
            elif self.type == "GIN":
                self.gnn_layer_func = tgnn.GIN


            self.graph_conv_1 = self.gnn_layer_func(num_node_features, gnn_hidden, num_layers=2,  normalize=True)
            self.graph_conv_2 = self.gnn_layer_func(gnn_hidden, gnn_hidden, num_layers=2, normalize=True)
            self.graph_conv_3 = self.gnn_layer_func(gnn_hidden, gnn_hidden, num_layers=2, normalize=True)
            self.gnn_drop_1 = nn.Dropout(p=0.05)
            self.gnn_drop_2 = nn.Dropout(p=0.05)
            self.gnn_drop_3 = nn.Dropout(p=0.05)
            self.gnn_relu1 = nn.ReLU()
            self.gnn_relu2 = nn.ReLU()
            self.gnn_relu3 = nn.ReLU()
            self.fc_1 = nn.Linear(gnn_hidden + sf_hidden, fc_hidden)
        else:
            self.fc_1 = nn.Linear(sf_hidden, fc_hidden)
        self.fc_2 = nn.Linear(fc_hidden, fc_hidden)
        self.fc_3 = nn.Linear(fc_hidden, fc_hidden)
        self.fc_drop_1 = nn.Dropout(p=0.05)
        self.fc_drop_2 = nn.Dropout(p=0.05)
        self.fc_drop_3 = nn.Dropout(p=0.05)
        self.fc_relu1 = nn.ReLU()
        self.fc_relu2 = nn.ReLU()
        self.fc_relu3 = nn.ReLU()
        self.predictor = nn.Linear(fc_hidden, 3)
        self._initialize_weights()

        self.train_loss = MeanAbsolutePercentageError()
        self.val_loss = MeanAbsolutePercentageError()
        self.test_loss = MeanAbsolutePercentageError()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_tensor(m.weight, "thomas")
                init_tensor(m.bias, "thomas")

    def forward(self, data):
        x, A, static_feature = data.x, data.edge_index, data.static.view(1, -1)
        if not self.type == "MLP":
            x = self.graph_conv_1(x, A)
            x = self.gnn_relu1(x)
            x = self.gnn_drop_1(x)

            x = self.graph_conv_2(x, A)
            x = self.gnn_relu2(x)
            x = self.gnn_drop_2(x)

            x = self.graph_conv_3(x, A)
            x = self.gnn_relu3(x)
            x = self.gnn_drop_3(x)

            x = scatter(x, data.batch, dim=0, reduce=self.reduce_func)
            x = torch.cat([x, static_feature], dim=1)
            x = self.fc_1(x)
        else:
            x = self.fc_1(static_feature)
        x = self.fc_relu1(x)
        x = self.fc_drop_1(x)
        x = self.fc_2(x)
        x = self.fc_relu2(x)
        x = self.fc_drop_2(x)
        x = self.fc_3(x)
        x = self.fc_relu3(x)
        feat = self.fc_drop_3(x)
        x = self.predictor(feat)

        pred = -F.logsigmoid(x)

        return pred

    def training_step(self, data, batch_idx):
        data.y = torch.Tensor([[data.y[0]*1000, data.y[1]*10, data.y[2]]])
        data = data.to(device=torch.device("cuda"))
        y_hat = self(data)
        y = data.y
        loss = F.huber_loss(y_hat, y)
        self.train_loss(y_hat, y)
        self.log('train_loss', self.train_loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, data, batch_idx):
        data.y = torch.Tensor([[data.y[0]*1000, data.y[1]*10, data.y[2]]])
        data = data.to(device=torch.device("cuda"))
        y_hat = self(data)    
        y = data.y
        self.val_loss(y_hat, y)
        self.log('val_loss', self.val_loss, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, data, batch_idx):
        data.y = torch.Tensor([[data.y[0]*1000, data.y[1]*10, data.y[2]]])
        data = data.to(device=torch.device("cuda"))
        y_hat = self(data)
        y = data.y
        self.test_loss(y_hat, y)
        self.log('test_loss', self.test_loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2.7542287033381663e-05)