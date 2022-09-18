import torch
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
from torch_geometric.datasets import WikiCS, Planetoid
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR

# glnn part
import argparse
import torch.nn.functional as F
from torch_geometric.nn import GCN, MLP

parser = argparse.ArgumentParser()
parser.add_argument('--lamb', type=float, default=0.0,
                    help='Balances loss from hard labels and teacher outputs')
parser.add_argument('--gpu',type=int, default=0,help='which gpu')
args = parser.parse_args()
device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')

times_for_average = 5

dataset_name = 'CiteSeer'
path = osp.join(osp.expanduser('~'), 'datasets')
dataset = Planetoid(path, name=dataset_name, transform=T.NormalizeFeatures())
data = dataset[0].to(device)
'''
path = osp.join(osp.expanduser('~'), 'datasets', 'WikiCS')
dataset = WikiCS(path, transform=T.NormalizeFeatures())
data = dataset[0].to(device)
'''

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
    result, y_soft = LREvaluator()(z, data.y, split)
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
    for n in range(times_for_average):
        mlp = MLP([dataset.num_node_features, 64, dataset.num_classes], dropout=0.5).to(device)
        mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01, weight_decay=5e-4)
        
        aug1 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])
        aug2 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])

        gconv = GConv(input_dim=dataset.num_features, hidden_dim=256).to(device)
        encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
        contrast_model = WithinEmbedContrast(loss=L.BarlowTwins()).to(device)

        optimizer = Adam(encoder_model.parameters(), lr=5e-4)
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
        
        print('Pretrain finished')
        test_result, y_soft = test(encoder_model, data)
        print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, Acc={test_result["acc"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')
        acc_teacher.append(test_result["acc"])
        
        # Start training a student            
        print('Training Student MLP:')
        best_val_acc = -1
        best_test_acc = -1
        for epoch in range(1, 501):
            loss = train_student(y_soft, data, mlp, mlp_optimizer)
            if epoch % 20 == 0:
                train_acc, val_acc, test_acc = test_student(mlp)
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                      f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
        acc_student.append(best_test_acc)
        
        del gconv
        del mlp
    
    teacher_average = sum(acc_teacher)/times_for_average
    student_average = sum(acc_student)/times_for_average
    for i in range(times_for_average):
        std_teacher += (acc_teacher[i]-teacher_average)**2
    std_teacher = sqrt(std_teacher)
    for i in range(times_for_average):
        std_student += (acc_student[i]-student_average)**2
    std_student = sqrt(std_student/times_for_average)
    print(dataset_name)
    print('training finished for: ', times_for_average, ' times')
    print('average teacher acc: ', teacher_average, ', and std: ', std_teacher)
    print('average student acc: ', student_average, ', and std: ', std_student)

if __name__ == '__main__':
    main()
