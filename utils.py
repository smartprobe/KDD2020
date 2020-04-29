import numpy as np
import scipy.sparse as sp
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import manifold
import time
import os
import datetime
import math


def cus_topk(score, ratio, batch):
    # perm = topk(score, self.ratio, batch)

    graph_ids = torch.unique(batch)
    lens = 0
    idx2 = []
    for graph_id in graph_ids:
        scorea = score[batch == graph_id]
        k = int(math.ceil(scorea.shape[0]*ratio))
        if k < 2: k=2
        scoreb,idx = torch.topk(scorea, k)
        # idx2.append(torch.full_like(idx, lens, dtype=torch.int) + idx)
        idx2.append( torch.full_like(idx, lens, dtype=torch.long) + idx )
        lens = lens + scorea.shape[0]
        # print('done')

    idx2 = torch.cat(idx2,dim=0)
    return idx2


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def my_TSNE(x, y, edge_index, title):            # x ：node feature  y： node label
    y = y.cpu().numpy() # np.array(y)
    edge_index = edge_index.cpu().numpy().astype(int)
    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)

    t0 = time.time()
    fig = plot_embedding(x, y,edge_index,
                         title)

    # return  fig

def plot_embedding(data, label,edge_index, title):

    fig = plt.figure(figsize=(6,6))
    plt.scatter(data[:, 0], data[:, 1], 10,label)
    for i in range(edge_index.shape[1]):
        plt.plot([data[edge_index[0, i], 0],data[edge_index[1, i], 0]],\
                 [data[edge_index[0, i], 1],data[edge_index[1, i], 1]], 'ro-')

    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xticks(np.arange(0, 1, 0.1))
    plt.yticks(np.arange(0, 1, 0.1))

    # plt.title(title)
    plt.savefig('./'+ title+'.png')
    plt.clf()
    return fig

def save_log(result_file,i):
    if not os.path.exists(result_file):
        f = open(result_file, 'w')
        f.write(str(i))
        f.write('\n' + str(datetime.datetime.now()))
        datetime.datetime
        f.close()
    else:
        f = open(result_file, 'a')
        f.write('\n' + str(i))
        f.write('\n' + str(datetime.datetime.now()))
        f.close()
