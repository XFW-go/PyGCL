import torch
import os
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch_geometric.transforms as T

from math import sqrt
from tqdm import tqdm
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
from torch_geometric.utils import to_undirected

# glnn part
import argparse
import torch.nn.functional as F
from torch_geometric.nn import GCN, MLP

parser = argparse.ArgumentParser()
parser.add_argument('--lamb', type=float, default=0.0,
                    help='Balances loss from hard labels and teacher outputs')
parser.add_argument('--gpu',type=int, default=0,help='which gpu')
args = parser.parse_args()
torch.set_num_threads(1)
device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')

times_for_average = 5
mlp_hidden = 4
dataset_name = ''
dataset = ''
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
        print(data.edge_index.shape)
        print(data.edge_index.type)
    elif dataset_name =='Amazon-Photo':
        data_name = 'amazon_electronics_photo'
        print('--- Loading data according to CPF')
        data = get_dataset_benchmark(path, data_name, 20, 30)
    else:
        data = get_dataset_split(path, dataset_name, 20, 30)
    data = data.to(device)    
    
    return data

class GConv(torch.nn.Module):
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

class GConv2(torch.nn.Module):
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
    z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)        
    #split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    split = dict()
    split['train'] = data.train_mask
    split['test'] = data.test_mask
    split['valid'] = data.val_mask
    result, y_soft = LREvaluator().evaluate(z, data.y, split, model_path)
    return result, y_soft

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
    std_teacher = 0
    std_student = 0
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
        lrr, hidden, my_pe, my_pf, num_classes = [1e-3, 256, 0.3, 0.0, 40] #0.5 0.5?
    elif dataset_name=='ogbn-products':
        lrr=1e-3
        num_classes=47
        
    for n in range(times_for_average): 
        test_teacher = 0
        test_stud = 0
        #mlp = MLP([data.x.shape[1], 256, 256, num_classes], dropout=0.5).to(device)
        mlp = MLP(in_channels=data.x.shape[1], hidden_channels = mlp_hidden, out_channels=num_classes, num_layers=3, dropout=0.2).to(device)
        mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01, weight_decay=0)
        
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
        
        with tqdm(total=1000, desc='(T)') as pbar:
            for epoch in range(1, 1001):
                loss = train(encoder_model, contrast_model, data, optimizer)
                scheduler.step()
                pbar.set_postfix({'loss': loss})
                pbar.update()
        loss = train(encoder_model, contrast_model, data, optimizer)
        scheduler.step()
        print('Pretrain finished')
           
        test_result, y_soft = test(encoder_model, data)
        print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, Acc={test_result["acc"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')
        test_teacher = test_result["acc"]
        
        # Start training a student                
        print('Training Student MLP:')
        best_val_acc = -1
        best_test_acc = -1
        for epoch in range(1, 2001):
            loss = train_student(y_soft, data, mlp, mlp_optimizer)
            if epoch % 100 == 0:
                train_acc, val_acc, test_acc = test_student(mlp)
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                      f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
        test_stud = best_test_acc
        
        del gconv
        del mlp
        
        acc_teacher.append(test_teacher)
        acc_student.append(test_stud)

    teacher_average = sum(acc_teacher)/(times_for_average)
    student_average = sum(acc_student)/(times_for_average)
    for i in range(times_for_average):
        std_teacher += (acc_teacher[i]-teacher_average)**2
    std_teacher = sqrt(std_teacher)
    for i in range(times_for_average):
        std_student += (acc_student[i]-student_average)**2
    std_student = sqrt(std_student/(times_for_average))
    
    print(dataset_name)
    print('Layer=2   MLP hidden=%d'%mlp_hidden)
    print('training finished for: ', times_for_average, ' times')
    print('average teacher acc: ', teacher_average, ', and std: ', std_teacher)
    print('average student acc: ', student_average, ', and std: ', std_student)
    

if __name__ == '__main__':
    dataset_name = 'Amazon-CS'
    data = read_data(dataset_name)
    os.system('mkdir ./output/GBT/%s'%dataset_name)
    os.system('mkdir ./output/GBT/%s/0926'%(dataset_name))
    model_path = './output/GBT/%s/0926/'%(dataset_name)
    main()
