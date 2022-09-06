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
from torch_geometric.datasets import WikiCS, Planetoid, Amazon, Coauthor, PPI
from ogb.nodeproppred import PygNodePropPredDataset
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

# glnn part
import argparse
import torch.nn.functional as F
from torch_geometric.nn import GCN, MLP

parser = argparse.ArgumentParser()
parser.add_argument('--lamb', type=float, default=0.0,
                    help='Balances loss from hard labels and teacher outputs')
parser.add_argument('--pf1', type=int, default=0, help='The value of pf1')
parser.add_argument('--gpu',type=int, default=0,help='which gpu')
args = parser.parse_args()
device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')


times_for_average = 3
dataset_name = ''
dataset = ''
data = dict()

def read_data(dataset_name):
    ds_path = osp.join(osp.expanduser('~'), 'datasets', dataset_name)
    feature_norm = T.NormalizeFeatures()
    create_masks = T.RandomNodeSplit(
                        split="train_rest",
                        num_splits=20,
                        num_val=0.1,
                        num_test=0.8,
                    )
    if dataset_name == 'WikiCS':    
        dataset = WikiCS(ds_path, transform=feature_norm)
        data = dataset[0].to(device)
    elif dataset_name == "Amazon-CS":
        dataset = Amazon(
            root=ds_path,
            name="computers",
            transform=feature_norm,
            pre_transform=create_masks,
        )
        data = dataset[0].to(device)
    elif dataset_name == "Amazon-Photo":
        dataset = Amazon(
            root=ds_path,
            name="photo",
            transform=feature_norm,
            pre_transform=create_masks,
        )
        data = dataset[0].to(device)
    elif dataset_name == "Coauthor-CS":
        dataset = Coauthor(
            root=ds_path,
            name="cs",
            transform=feature_norm,
            pre_transform=create_masks,
        )
        data = dataset[0].to(device)
    elif dataset_name == "Coauthor-Physics":
        dataset = Coauthor(
            root=ds_path,
            name="physics",
            transform=feature_norm,
            pre_transform=create_masks,
        )
        data = dataset[0].to(device)
    elif dataset_name == "ogbn-arxiv":
        data, dataset = read_ogb_dataset(name=dataset_name, path=ds_path)
        data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    elif dataset_name == "ogbn-products":
        data, dataset = read_ogb_dataset(name=dataset_name, path=ds_path)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return data, dataset

def read_ogb_dataset(name: str, path: str) -> Data:
    dataset = PygNodePropPredDataset(root=path, name=name)
    split_idx = dataset.get_idx_split()

    data = dataset[0]

    data.train_mask = torch.zeros((data.num_nodes,), dtype=torch.bool)
    data.train_mask[split_idx["train"]] = True

    data.val_mask = torch.zeros((data.num_nodes,), dtype=torch.bool)
    data.val_mask[split_idx["valid"]] = True

    data.test_mask = torch.zeros((data.num_nodes,), dtype=torch.bool)
    data.test_mask[split_idx["test"]] = True

    data.y = data.y.squeeze(dim=-1)

    return data, dataset

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


def test(encoder_model, data, split_n):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)        
    #split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    split = dict()
    split['train'] = data.train_mask[:, split_n]
    if dataset_name=='WikiCS':
        split['test'] = data.test_mask
    else:
        split['test'] = data.test_mask[:, split_n]
    split['valid'] = data.val_mask[:, split_n]
    result, y_soft = LREvaluator()(z, data.y, split)
    return result, y_soft

# For students ------------
def train_student(y_soft, data, mlp, mlp_optimizer,split_n):
    mlp.train()
    mlp_optimizer.zero_grad()
    out = mlp(data.x)

    # For WikiCS dataset
    loss1 = F.cross_entropy(out[data.train_mask[:,split_n]], data.y[data.train_mask[:,split_n]])
    loss2 = F.kl_div(out.log_softmax(dim=-1), y_soft, reduction='batchmean',
                     log_target=True)
    loss = args.lamb * loss1 + (1 - args.lamb) * loss2
    loss.backward(retain_graph=True)
    mlp_optimizer.step()
    
    return float(loss)

@torch.no_grad()
def test_student(mlp, split_n):
    mlp.eval()
    pred = mlp(data.x).argmax(dim=-1)
    accs = []
    #for _, mask in data('train_mask', 'val_mask', 'test_mask'):
    #    accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    # For WikiCS 
    
    flag = 0
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        flag+=1
        if flag<3:
            tmp = mask[:, split_n]
        else:
            tmp = mask
        accs.append(int((pred[tmp] == data.y[tmp]).sum()) / int(tmp.sum()))
    
    return accs
# ------------ For students 

def main():
    best_pe1 = 0
    best_pf1 = 0
    best_teacher_ave = -1
    for pe1 in range(6):
#        for pf1 in range(5):
        pf1 = args.pf1
        test_acc_sum = 0
        print('Start Training with (pe1=%f, pf1=%f)'%(pe1,pf1))
        print('--------------------------------------------')
        for n in range(times_for_average):                    
            for split_n in range(20):                          
                mlp = MLP([dataset.num_node_features, 64, dataset.num_classes], dropout=0.5).to(device)
                mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01, weight_decay=5e-4)
                
                aug1 = A.Compose([A.EdgeRemoving(pe=pe1*0.1), A.FeatureMasking(pf=pf1*0.1)])
                aug2 = A.Compose([A.EdgeRemoving(pe=pe1*0.1), A.FeatureMasking(pf=pf1*0.1)])

                gconv = GConv(input_dim=dataset.num_features, hidden_dim=128).to(device) # change hidden-dim
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
                test_result, y_soft = test(encoder_model, data, split_n)
                print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, Acc={test_result["acc"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')
                test_acc_sum += test_result["acc"]
                #acc_teacher_split += test_result["acc"]
                #acc_teacher.append(test_result["acc"])
                
                del gconv
                del mlp
        
        average_for_splits = test_acc_sum/(20.0*times_for_average)
        print('Finish training with acc= ', average_for_splits)
        print('-------------------------------------------')
        if average_for_splits > best_teacher_ave:
            best_teacher_ave = average_for_splits
            best_pe1 = pe1
            best_pf1 = pf1
    
    print(best_teacher_ave)
    print('param = ', best_pe1*0.1, '  ', best_pf1*0.1)
    
    with open('./hyperparam.txt', 'a') as f:
        f.write('Dataset: WikiCS \n')
        f.write('Times for average = %d\n' % times_for_average)
        f.write('best_teacher_test_acc: %f\n' % best_teacher_ave)
        f.write('pe1=%f, pf1=%f \n' %(best_pe1*0.1, best_pf1*0.1))
    

if __name__ == '__main__':
    dataset_name = 'Amazon-CS'
    data, dataset = read_data(dataset_name)
    main()
