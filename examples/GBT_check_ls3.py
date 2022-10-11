import torch
import os
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch_geometric.transforms as T

from math import sqrt
from tqdm import tqdm
import numpy as np
from numpy.random import randint
from torch.optim import Adam
from GCL.eval import get_split#, LREvaluator
from logistic_regression import LREvaluator
from GCL.models.contrast_model import WithinEmbedContrast
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import WikiCS, Planetoid, Amazon, Coauthor, PPI
from ogb.nodeproppred import PygNodePropPredDataset
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from dataset import get_dataset_split, get_ogb_split
from dataset_cpf import get_dataset_benchmark
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, degree
from sklearn.metrics import f1_score
#from scipy.spatial.distance import cdist as dist

# glnn part
import argparse
import torch.nn.functional as F
from torch.nn.functional import normalize
from torch_geometric.nn import GCN, MLP

parser = argparse.ArgumentParser()
parser.add_argument('--lamb', type=float, default=0.0,
                    help='Balances loss from hard labels and teacher outputs')
parser.add_argument('--gpu',type=int, default=0,help='which gpu')
args = parser.parse_args()
torch.set_num_threads(1)
device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')

times_for_average = 5
mlp_hidden = 128
dataset_name = ''
dataset = ''
deg = None
data = dict()

def read_data(dataset_name):
    path = osp.join(osp.expanduser('~'), 'datasets', dataset_name)
    if dataset_name.startswith('ogbn'):
        data = get_ogb_split(path, dataset_name)
        data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    elif dataset_name == 'Amazon-CS':
        data_name = 'amazon_electronics_computers'
        print('--- Loading data according to CPF')
        data = get_dataset_benchmark(path, data_name, 20, 30)
    elif dataset_name =='Amazon-Photo':
        data_name = 'amazon_electronics_photo'
        print('--- Loading data according to CPF')
        data = get_dataset_benchmark(path, data_name, 20, 30)
    else:
        data = get_dataset_split(path, dataset_name, 20, 30)
    data = data.to(device)    
    
    return data

class GConv2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GConv, self).__init__()
        self.act = torch.nn.PReLU()
        self.bn = torch.nn.BatchNorm1d(2 * hidden_dim, momentum=0.01)
        self.conv1 = GCNConv(input_dim, 2 * hidden_dim, cached=False)
        self.conv2 = GCNConv(2 * hidden_dim, hidden_dim, cached=False)

    def forward(self, x, edge_index, edge_weight=None):
        z = self.conv1(x, edge_index, edge_weight)
        z = self.bn(z)
        z = self.act(z)
        z = self.conv2(z, edge_index, edge_weight)
        return z

class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GConv, self).__init__()
        self.act = torch.nn.PReLU()
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim, momentum=0.01)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim, momentum=0.01)
        self.conv1 = GCNConv(input_dim, hidden_dim, cached=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, cached=False)
        self.conv3 = GCNConv(hidden_dim, hidden_dim, cached=False)

    def forward(self, x, edge_index, edge_weight=None):
        z = self.conv1(x, edge_index, edge_weight)
        z = self.bn1(z)
        z = self.act(z)
        z = self.conv2(z, edge_index, edge_weight)
        z = self.bn2(z)
        z = self.act(z)
        z = self.conv3(z, edge_index, edge_weight)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

def dist(XA, XB, metric):
    if metric=='cosine':
        return 1 - torch.inner(XA, XB)/(torch.norm(XA)*torch.norm(XB))
    elif metric=='euclidean':
        return 
    elif metric=='cityblock':
        return
    return 0

def neighbor_grab():
    pass
    
def instance_queue_generate(mode, num, total):
    sample_mask = np.full(total, False)
    if mode=='random':
        for j in range(num):
            k = randint(total)
            sample_mask[k]=True
        #sample_mask = sample_mask.to(device)
    elif mode=='neighbor_all':
        pass
    return sample_mask
    
def sample_queue_generate(data, threshold=5):
    n = data.num_nodes
    sample_mask = np.full(n, False)
    sample_mask[deg>threshold] = True
    return sample_mask

def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    _, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_attr)
    loss = contrast_model(z1, z2)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(encoder_model, data):
    encoder_model.eval()
    with torch.no_grad():
        z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)        
    #split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    split = dict()
    split['train'] = data.train_mask
    split['test'] = data.test_mask
    split['valid'] = data.val_mask
    result, y_soft = LREvaluator().evaluate(z, data.y, split, model_path)
    return result, y_soft, z

# For students ------------
def train_student(y_soft, data, mlp, mlp_optimizer):
    mlp.train()
    mlp_optimizer.zero_grad()
    out = mlp(data.x)

    loss1 = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss2 = F.kl_div(out.log_softmax(dim=-1), y_soft, reduction='batchmean',
                     log_target=True)
    loss = args.lamb * loss1 + (1 - args.lamb) * loss2
    loss.backward(retain_graph=True)
    mlp_optimizer.step()
    
    return float(loss)

def train_student_via_embedding(embed_ssl, y_soft, data, mlp, mlp_optimizer, weight_loss1, weight_loss2, weight_loss3):
    # MLP only output the embedding 
    mlp.train()
    mlp_optimizer.zero_grad()
    out, embed_mlp = mlp(data.x, return_emb=True)
    if weight_loss1!=0:
        embed_mlp.detach()
    num_data = data.x.shape[0]
    
    loss1 = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss2 = F.kl_div(out.log_softmax(dim=-1), y_soft, reduction='batchmean',
                     log_target=True)
    
    if weight_loss3!=0:
        good_nodes_mask = sample_queue_generate(data, 3)
        sample_mask = instance_queue_generate('random', 500, num_data)
        A_t = embed_ssl[good_nodes_mask]
        A_s = embed_mlp[good_nodes_mask]
        B_t = embed_ssl[sample_mask]
        B_s = embed_mlp[sample_mask]
        metrics = 'cosine'
        if metrics=='cosine':
            A_t = normalize(A_t, p=2.0, dim=1)
            A_s = normalize(A_s, p=2.0, dim=1)
            B_t = normalize(B_t, p=2.0, dim=1)
            B_s = normalize(B_s, p=2.0, dim=1)
            teacher_prob = torch.matmul(A_t, B_t.T)
            student_prob = torch.matmul(A_s, B_s.T)
        elif metrics=='euclidean':
            pass

        ls3 = torch.nn.MSELoss()
        loss3 = ls3(teacher_prob, student_prob)
        #loss3 = F.kl_div(teacher_prob.log_softmax(dim=-1), student_prob.log_softmax(dim=-1), reduction='batchmean', log_target=True)
        loss = weight_loss1*loss1 + weight_loss2*loss2 + weight_loss3*loss3
    else:
        loss = weight_loss1 * loss1 + weight_loss2 * loss2
    
    loss.backward(retain_graph=True)
    mlp_optimizer.step()
    
    return float(loss)

@torch.no_grad()
def test_student(mlp):
    mlp.eval()
    pred = mlp(data.x).argmax(dim=-1)
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum())) 
    return accs
# ------------ For students 

def main():
    acc_teacher = []
    acc_student = []
    acc_emb_stu = []
    std_teacher = 0
    std_student = 0
    log_interval= 50
    lrr, hidden, my_pe, my_pf, num_classes = [1e-4, 256, 0.5, 0.1, 1]
    if dataset_name=='WikiCS':
        lrr, hidden, my_pe, my_pf, num_classes=[5e-4, 256, 0.5, 0.0, 10]
    elif dataset_name=='Amazon-CS': # 0.2 0.2 / 0.1 0.3
        lrr, hidden, my_pe, my_pf, num_classes = [5e-4, 256, 0.0, 0.3, 10]
    elif dataset_name=='Amazon-Photo':
        lrr, hidden, my_pe, my_pf, num_classes = [1e-4, 256, 0.5, 0.1, 8]
    elif dataset_name=='Coauthor-CS':
        lrr, hidden, my_pe, my_pf, num_classes = [1e-5, 256, 0.5, 0.0, 15]
    elif dataset_name=='Coauthor-Phy': # 0.0 0.5 / 0.4 0.2
        lrr, hidden, my_pe, my_pf, num_classes = [1e-5, 256, 0.5, 0.2, 5]
    elif dataset_name=='Cora': 
        lrr, hidden, my_pe, my_pf, num_classes = [1e-4, 256, 0.3, 0.4, 7]
    elif dataset_name=='PubMed':
        lrr, hidden, my_pe, my_pf, num_classes = [1e-4, 256, 0.1, 0.5, 3]
    elif dataset_name=='CiteSeer':
        lrr, hidden, my_pe, my_pf, num_classes = [1e-4, 256, 0.5, 0.5, 6]
    elif dataset_name=='ogbn-arxiv':
        lrr, hidden, my_pe, my_pf, num_classes = [1e-3, 256, 0.3, 0.0, 40]
    elif dataset_name=='ogbn-products':
        lrr=1e-3
        num_classes=47
        
    for n in range(times_for_average): 
        test_teacher = 0
        test_stud = 0
        test_emb_stud = 0
        #mlp = MLP([data.x.shape[1], 256, num_classes], dropout=0.2).to(device)
        #mlp = MLP([data.x.shape[1], 64], dropout=0.5).to(device)
        mlp = MLP(in_channels=data.x.shape[1], hidden_channels = mlp_hidden, out_channels=num_classes, num_layers=3, dropout=0.2).to(device)
        mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=0.02, weight_decay=0)
        mlp_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=mlp_optimizer,
            warmup_epochs=100,
            max_epochs=1000)
        
        aug1 = A.Compose([A.EdgeRemoving(pe=my_pe), A.FeatureMasking(pf=my_pf)])
        aug2 = A.Compose([A.EdgeRemoving(pe=my_pe), A.FeatureMasking(pf=my_pf)])

        gconv = GConv(input_dim=data.x.shape[1], hidden_dim=hidden).to(device)
        encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
        contrast_model = WithinEmbedContrast(loss=L.BarlowTwins()).to(device)

        optimizer = Adam(encoder_model.parameters(), lr=lrr)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=100,
            max_epochs=1000)
        
        best_val_acc_GBT = -1
        best_y_soft = None
        embed_ssl = None
        test_result = None
        for epoch in range(1, 1001):
            loss = train(encoder_model, contrast_model, data, optimizer)
            if epoch % log_interval==0:
                accs, y_soft, embed = test(encoder_model, data)
                encoder_model.train()
                if accs['acc_val'] > best_val_acc_GBT:
                    best_val_acc_GBT = accs['acc_val']
                    # Save the models
                    torch.save(encoder_model.state_dict(), model_path + 'GBT_best.pth')
                    test_result = accs  
            scheduler.step()
        print('Pretrain finished')
        with torch.no_grad():
            encoder_model.load_state_dict(torch.load(model_path + 'GBT_best.pth')) 
        test_result, y_soft, embed_ssl = test(encoder_model, data)
        print('check if with grad')
        print(y_soft.requires_grad, '   ', embed_ssl.requires_grad)
        print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, Acc={test_result["acc"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')
        test_teacher = test_result["acc"]
        
        # Start training a student                
        print('Training Student MLP:')
        best_val_acc = -1
        best_test_acc = -1
        for epoch in range(1, 1001):
            loss = train_student_via_embedding(embed_ssl, y_soft, data, mlp, mlp_optimizer, 0, 1, 0)
            if epoch % 100 == 0:
                train_acc, val_acc, test_acc = test_student(mlp)
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                      f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
            mlp_scheduler.step()
        test_stud = best_test_acc
        
        best_val_acc = -1
        best_test_acc = -1
        #encoder_model.load_state_dict(torch.load(model_path + 'GBT_best.pth')) 
        #test_result, y_soft, embed_ssl = test(encoder_model, data)
        del mlp; del mlp_optimizer; #del mlp_scheduler;
        mlp = MLP(in_channels=data.x.shape[1], hidden_channels = mlp_hidden, out_channels=num_classes, num_layers=3, dropout=0.2).to(device)
        mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=0.02, weight_decay=0)
        mlp_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=mlp_optimizer,
            warmup_epochs=1000,
            max_epochs=3000)
        print(y_soft.requires_grad, '   ', embed_ssl.requires_grad)
        for epoch in range(1, 1001):
            loss = train_student_via_embedding(embed_ssl, y_soft, data, mlp, mlp_optimizer, 0, 0, 1)
            if epoch % 100 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            mlp_scheduler.step()
        for epoch in range(1001, 3001):
            loss = train_student_via_embedding(embed_ssl, y_soft, data, mlp, mlp_optimizer, 1, 0, 0)
            if epoch % 100 == 0:
                train_acc, val_acc, test_acc = test_student(mlp)
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                      f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
            mlp_scheduler.step()
        test_emb_stud = best_test_acc
        
        del gconv; del encoder_model; del optimizer; del scheduler;
        del mlp; del mlp_optimizer; #del mlp_scheduler;
        
        acc_teacher.append(test_teacher)
        acc_student.append(test_stud)
        acc_emb_stu.append(test_emb_stud)

    teacher_average = sum(acc_teacher)/(times_for_average)
    student_average = sum(acc_student)/(times_for_average)
    emb_stu_average = sum(acc_emb_stu)/(times_for_average)
    for i in range(times_for_average):
        std_teacher += (acc_teacher[i]-teacher_average)**2
    std_teacher = sqrt(std_teacher)
    for i in range(times_for_average):
        std_student += (acc_student[i]-student_average)**2
    std_student = sqrt(std_student/(times_for_average))
    
    print(dataset_name)
    print('MLP hidden= %d'%mlp_hidden)
    print('training finished for: ', times_for_average, ' times')
    print('average teacher acc: ', teacher_average, ', and std: ', std_teacher)
    print('average student acc: ', student_average, ', and std: ', std_student)
    print('average student with loss 3 acc: ', emb_stu_average)
    

if __name__ == '__main__':
    dataset_name = 'Amazon-CS'
    data = read_data(dataset_name)
    os.system('mkdir ./output/GBT/%s'%dataset_name)
    os.system('mkdir ./output/GBT/%s/ls30'%dataset_name)
    #os.system('mkdir ./output/GBT/%s/emb/%d'%(dataset_name, mlp_hidden))
    model_path = './output/GBT/%s/ls30/'%(dataset_name)
    deg = degree(data.edge_index[0], data.num_nodes).detach().cpu().numpy()
    main()
