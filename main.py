import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric import utils
from networks import  Net
import torch.nn.functional as F
import argparse
import os
from torch.utils.data import random_split
from utils import my_TSNE,save_log
from sklearn import manifold
import numpy as np
import pickle
from tensorboardX import SummaryWriter
import time

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')

#NCI109
# parser.add_argument('--lr', type=float, default=0.0001,
#                     help='learning rate')
#APPNP
# parser.add_argument('--lr', type=float, default=0.0004,
#                     help='learning rate')
#NCI1
# parser.add_argument('--lr', type=float, default=0.00005,
#                     help='learning rate')

parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio')
parser.add_argument('--dataset', type=str, default='NCI1',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--epochs', type=int, default=100000,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--logpath', type=str, default='result',
                    help='result')

args = parser.parse_args()
args.device = 'cpu'

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'
print(args.device)

dataset = TUDataset(os.path.join('data',args.dataset),name=args.dataset)
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

num_training = int(len(dataset)*0.8)
num_val = int(len(dataset)*0.1)
num_test = len(dataset) - (num_training+num_val)
training_set,validation_set,test_set = random_split(dataset,[num_training,num_val,num_test])

train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
# train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(validation_set,batch_size=args.batch_size,shuffle=False)
#test_loader = DataLoader(test_set,batch_size=1,shuffle=False)
# PROTEINS
test_loader = DataLoader(test_set,batch_size=args.batch_size,shuffle=False)

model = Net(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def trans_x(x):            # x ：node feature  y： node label
    x = x.cpu().detach().numpy()  #x.cpu().numpy() #np.array(x)
    x = np.squeeze(x)
    return x


def cus_fig(data,index,perm1,perm2,perm3,bs1,bs2,bs3, edge_index1, edge_index2, edge_index3, dataset, i, save_pickle=False):
    x0 = data.x[data.batch[data.batch == index]]
    '''
    input: data
    index: index of graph in each batch
    perm: index of elected node in SAG1, SAG2, SAG3
    edge_index: ajacent matrix of selected nodes
    dataset: name of dataset
    i: index of graph in dataset
    '''

    edge_index0 = data.edge_index
    print('i = {}, No. of Nodes = {}'.format(i, x0.shape[0]))

    s0 = data.batch[data.batch == index].shape[0]
    s1 = bs1[bs1 == index].shape[0]
    s2 = bs2[bs2 == index].shape[0]
    s3 = bs3[bs3 == index].shape[0]

    y0 = torch.ones((s0,), dtype=int)
    y1 = torch.ones((s1,), dtype=int)
    y2 = torch.ones((s2,), dtype=int)
    y3 = torch.ones((s3,), dtype=int)

    node_ix0 = torch.where(data.batch == index)[0].cpu().detach().numpy()
    node_ix1 = torch.where(bs1 == index)[0].cpu().detach().numpy()
    node_ix2 = torch.where(bs2 == index)[0].cpu().detach().numpy()
    node_ix3 = torch.where(bs3 == index)[0].cpu().detach().numpy()

    edge_index0 = edge_index0[:, edge_index0[0, :] >= node_ix0.min()]
    edge_index0 = edge_index0[:, edge_index0[0, :] < node_ix0.max()]

    edge_index1 = edge_index1[:, edge_index1[0, :] >= node_ix1.min()]
    edge_index1 = edge_index1[:, edge_index1[0, :] < node_ix1.max()]

    edge_index2 = edge_index2[:, edge_index2[0, :] >= node_ix2.min()]
    edge_index2 = edge_index2[:, edge_index2[0, :] < node_ix2.max()]

    edge_index3 = edge_index3[:, edge_index3[0, :] >= node_ix3.min()]
    edge_index3 = edge_index3[:, edge_index3[0, :] < node_ix3.max()]

    tsne = manifold.TSNE(random_state=42)
    x0 = tsne.fit_transform(trans_x(x0))
    x_min, x_max = np.min(x0, 0), np.max(x0, 0)
    print(x_min, x_max)
    x0 = (x0 - x_min) / (x_max - x_min)

    perm_1 = perm1[node_ix1.min():node_ix1.max()+1].cpu().detach().numpy()
    perm_2 = perm2[node_ix2.min():node_ix2.max()+1].cpu().detach().numpy()
    perm_3 = perm3[node_ix3.min():node_ix3.max()+1].cpu().detach().numpy()

    edge_index0 = edge_index0 - node_ix0.min()
    edge_index1 = edge_index1 - node_ix1.min()
    edge_index2 = edge_index2 - node_ix2.min()
    edge_index3 = edge_index3 - node_ix3.min()

    x1 = x0[perm_1 - node_ix0.min(), :]
    x2 = x1[perm_2 - node_ix1.min(), :]
    x3 = x2[perm_3 - node_ix2.min(), :]

    my_TSNE(x0, y0, edge_index0, './fig/data_{}_{}_SAG0'.format(dataset, i))
    my_TSNE(x1, y1, edge_index1, './fig/data_{}_{}_SAG1'.format(dataset, i))
    my_TSNE(x2, y2, edge_index2, './fig/data_{}_{}_SAG2'.format(dataset, i))
    my_TSNE(x3, y3, edge_index3, './fig/data_{}_{}_SAG3'.format(dataset, i))

    if save_pickle:
        # y0 = y0.cpu().detach().numpy()
        # y1 = y1.cpu().detach().numpy()
        # y2 = y2.cpu().detach().numpy()
        # y3 = y3.cpu().detach().numpy()

        edge_index0 = edge_index0.cpu().detach().numpy()
        edge_index1 = edge_index1.cpu().detach().numpy()
        edge_index2 = edge_index2.cpu().detach().numpy()
        edge_index3 = edge_index3.cpu().detach().numpy()

        # a = [x0, y0, edge_index0,x1, y1, edge_index1,x2, y2, edge_index2,x3, y3, edge_index3 ]
        a = [x0, edge_index0, x1, edge_index1, x2, edge_index2, x3, edge_index3]
        with open('./fig/NC1_G_{}.pickle'.format(i), 'wb') as f:
            pickle.dump(a, f, protocol=pickle.HIGHEST_PROTOCOL)

def test(model,loader, plot = False):
    model = model.to(args.device)
    model.eval()
    correct = 0.
    loss = 0.
    for i, data in enumerate(loader):
        data = data.to(args.device)
        out, bs1,bs2,bs3,edge_index1,edge_index2,edge_index3, perm1,perm2,perm3 = model(data)

        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out,data.y,reduction='sum').item()

        '''
        plot
        '''
        if plot:
            for index in range(data.y.shape[0]):
                try:
                    cus_fig(data, index, perm1, perm2, perm3, bs1, bs2, bs3, edge_index1, edge_index2, edge_index3, args.dataset, index + i * args.batch_size)
                except:
                    print('skip graph {}'.format(index + i * args.batch_size))

    return correct / len(loader.dataset),loss / len(loader.dataset)


min_loss = 1e10
patience = 0

# print('begin testing')
# model_test = Net(args).to(args.device)
# model_test.load_state_dict(torch.load('latest.pth'))
# test_acc, test_loss = test(model_test, test_loader)
# print("Test accuarcy:{}".format(test_acc))
feature = []
label = []

index = 0

writer = SummaryWriter("./logs")

# val_acc,val_loss = test(model,val_loader, plot = True)
test_acc, test_loss = test(model, test_loader,  plot = False)


for epoch in range(args.epochs):
# for epoch in range(1):
    train_time1 = time.time()
    model.train()
    feature = []
    label = []
    for i, data in enumerate(train_loader):

        data = data.to(args.device)
        edge_index0 = data.edge_index
        out, bs1,bs2,bs3,edge_index1,edge_index2,edge_index3, perm1,perm2,perm3 = model(data)

        loss = F.nll_loss(out, data.y)
        print("Training loss:{}".format(loss.item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    val_acc,val_loss = test(model,val_loader)
    # val_acc, val_loss = test(model, test_loader)
    print("Validation loss:{}\taccuracy:{}".format(val_loss,val_acc))
    if val_loss < min_loss:
        torch.save(model.state_dict(),args.dataset+'latest.pth')
        print("Model saved at epoch{}".format(epoch))
        min_loss = val_loss
        patience = 0
    else:
        patience += 1
    if patience > args.patience:
        break

    train_time2 = time.time()
    print('one epoch time:', train_time2-train_time1)
    print('begin testing')
    try:
        model_test = Net(args).to(args.device)
        model_test.load_state_dict(torch.load(args.dataset+'latest.pth'))
        test_acc,test_loss = test(model_test,test_loader)
        print("Test accuarcy:{} \n".format(test_acc))
    except:
        print('device bugs, try to use same bs as training')

print('begin testing')
test_time1 = time.time()

model_test = Net(args).to(args.device)
model_test.load_state_dict(torch.load(args.dataset+'latest.pth'))
test_acc, test_loss = test(model_test, test_loader,  plot = False)

test_time2 = time.time()
print('Test time:', test_time2-test_time1)
print("Test accuarcy:{} \n".format(test_acc))

if not os.path.exists(args.logpath):
    os.makedirs(args.logpath)

logfile = os.path.join(args.logpath, args.dataset + '.txt')
save_log(logfile, test_acc)
