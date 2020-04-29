import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from layers import SAGPool
import scipy.sparse as sp

class Net(torch.nn.Module):
    def __init__(self,args):
        super(Net, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.ranks = dict()

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.pool1 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = SAGPool(self.nhid, ratio=self.pooling_ratio)

        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self. num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # neighbors = data.edge_index
        # for n in neighbors:
        #     outlinks = len(neighbors(n))
        #     print(outlinks)

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, perm = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        batch1 = batch
        edge_index1 = edge_index
        perm1 = perm

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, perm = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        batch2 = batch
        edge_index2 = edge_index
        perm2 = perm

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, perm = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        batch3 = batch
        edge_index3 = edge_index
        perm3 = perm

        if x1.shape != x2.shape:
            print('x2 fill 0')
            xt = torch.zeros_like(x1)
            xt[:x2.shape[0], :x2.shape[1]] = x2
            x2 = xt

        if x1.shape != x3.shape:
            print('x3 fill 0')
            xt = torch.zeros_like(x1)
            xt[:x3.shape[0], :x3.shape[1]] = x3
            x3 = xt

        x = x1 + x2 + x3

        if x.shape[0] != data.y.shape[0]:
            print('x fill 0')
            xt = torch.zeros((data.y.shape[0], x.shape[1]))
            xt[:x.shape[0], :x.shape[1]] = x
            x = xt

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x, batch1, batch2, batch3, edge_index1, edge_index2, edge_index3, perm1, perm2, perm3





    