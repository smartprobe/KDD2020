from torch_geometric.nn import GCNConv,APPNP
from torch_geometric.nn.pool.topk_pool import filter_adj, topk
from torch.nn import Parameter
from torch.autograd import Variable
import torch
#from utils import cus_topk as topk


class SAGPool(torch.nn.Module):
    # def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.tanh):
    #     super(SAGPool,self).__init__()
    #     self.in_channels = in_channels
    #     self.ratio = ratio
    #     self.score_layer = Conv(in_channels,1)
    #     self.non_linearity = non_linearity
    # def forward(self, x, edge_index, edge_attr=None, batch=None):
    #     if batch is None:
    #         batch = edge_index.new_zeros(x.size(0))
    #     #x = x.unsqueeze(-1) if x.dim() == 1 else x
    #     score = self.score_layer(x,edge_index).squeeze()
    #
    #     perm = topk(score, self.ratio, batch)
    #     x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
    #     batch = batch[perm]
    #     edge_index, edge_attr = filter_adj(
    #         edge_index, edge_attr, perm, num_nodes=score.size(0))
    #
    #     return x, edge_index, edge_attr, batch, perm

    def __init__(self,in_channels,ratio=0.8,Conv=APPNP,non_linearity=torch.tanh):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        # self.score_layer = Conv(in_channels,1)
        self.lin = torch.nn.Linear(in_channels, 1)
        self.score_layer =Conv(K=10, alpha=0.1)
        self.non_linearity = non_linearity

        self.norm_score = torch.nn.BatchNorm1d(1)
        self.norm_pg_score = torch.nn.BatchNorm1d(1)

        # self.weight_u1 = torch.FloatTensor(1, 1)
        # self.weight_u1 = Variable(self.weight_u1, requires_grad = True)
        # self.weight_u1 = torch.nn.Parameter(self.weight_u1)
        #
        # self.weight_u2 = torch.FloatTensor(1, 1)
        # self.weight_u2 = Variable(self.weight_u2, requires_grad = True)
        # self.weight_u2 = torch.nn.Parameter(self.weight_u2)
        #
        #
        # torch.nn.init.constant_(self.weight_u1, 0.5)
        # torch.nn.init.constant_(self.weight_u2, 0.5)

    '''
    input:
        x:           feature x
        edge_index:  index of selected x
    output:
        x:           output x
        edge_index:  index of selected x
        edge_attr :  (None)
        batch     :  predefined batch
        perm      :  (topk index from score)
    '''
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        # score = self.score_layer(x,edge_index).squeeze()

        # scoreo = self.norm_score(self.score_layer(self.lin(x), edge_index)).squeeze()
        # score_pagerank = self.norm_pg_score(pagerank(p=0.8, x=x, edge_index=edge_index)).squeeze()

        scoreo = self.score_layer(self.lin(x), edge_index).squeeze()
        scoreo = torch.nn.functional.softmax(scoreo)
        score_pagerank = norm_tensor(pagerank(p=0.2, x=x, edge_index=edge_index)).squeeze()

        score = score_pagerank + score_pagerank

        # score = torch.mul(self.weight_u1, scoreo) + torch.mul(self.weight_u2, score_pagerank)
        # score = score.squeeze()
        # print('scoreo max():', scoreo.max())
        # print('scoreo min():', scoreo.min())
        # print('scoreo mean():', scoreo.mean())
        #
        # print('score_pagerank max():', score_pagerank.max())
        # print('score_pagerank min():', score_pagerank.min())
        # print('score_pagerank mean():', score_pagerank.mean())
        #
        # print('score max():', score.max())
        # print('score min():', score.min())
        # print('score mean():', score.mean())

        '''
        2.1. page rank normalization,  weighted fusion
        2.2. page rank normalization,  attention 
        2.3. APPNP and page rank normalize, weighted fusion
        2.4. APPNP and page rank normalize, attention
        '''

        # adj = spare_to_dense(x, edge_index)
        # degree_matrix = adj.sum(dim=1)
        #perm = topk(degree_matrix, self.ratio, batch)
        # '''original score -> new score'''
        # score = score + degree_matrix

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        # print(edge_index.shape)
        # print(perm.shape)
        # print(score.size(0))
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm


def norm_tensor(vector):
    min_v = torch.min(vector)
    range_v = torch.max(vector) - min_v
    if range_v > 0:
        normalised = (vector - min_v) / range_v
    else:
        normalised = torch.zeros(vector.size())
    return normalised

def pagerank(p=0.8,x=None,edge_index=None):
    v = torch.full((x.size(0),1),float(1)/x.size(0),device = edge_index.device)
    adj = edge2sparse(x,edge_index)                                         
    unit = torch.full((x.size(0),x.size(0)),float(1)/x.size(0),device = edge_index.device)
                                                            
    for i in range(10):
        v = (1-p) * torch.sparse.FloatTensor.matmul(adj, v) + p * torch.sparse.FloatTensor.matmul(unit, v)
    return v

def spare_to_dense(x,edge_index):
    adj = torch.zeros(x.size(0),x.size(0),device=edge_index.device)
    row, col = edge_index
    # for i in range(row.size(0)):
    #     adj[row[i],col[i]]=1
    adj[row,col]=1
    return adj
def edge2sparse(x,edge_index):
    v = torch.ones((edge_index.shape[1]),dtype=torch.float,device=edge_index.device)
    adj = torch.sparse.FloatTensor(edge_index, v, (x.shape[0],x.shape[0]))
    return adj
