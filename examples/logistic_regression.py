import torch
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from sklearn.metrics import f1_score
from abc import ABC, abstractmethod

from GCL.eval import BaseEvaluator
from torch_geometric.nn import GCN #, MLP

import warnings
from typing import Any, Callable, Dict, List, Optional, Union
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Identity

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import NoneType


class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z

class MLP(torch.nn.Module):
    def __init__(
        self,
        channel_list: Optional[Union[List[int], int]] = None,
        *,
        in_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: Union[float, List[float]] = 0.,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = "batch_norm",
        norm_kwargs: Optional[Dict[str, Any]] = None,
        plain_last: bool = True,
        bias: Union[bool, List[bool]] = True,
        **kwargs,
    ):
        super().__init__()

        # Backward compatibility:
        act_first = act_first or kwargs.get("relu_first", False)
        batch_norm = kwargs.get("batch_norm", None)
        if batch_norm is not None and isinstance(batch_norm, bool):
            warnings.warn("Argument `batch_norm` is deprecated, "
                          "please use `norm` to specify normalization layer.")
            norm = 'batch_norm' if batch_norm else None
            batch_norm_kwargs = kwargs.get("batch_norm_kwargs", None)
            norm_kwargs = batch_norm_kwargs or {}

        if isinstance(channel_list, int):
            in_channels = channel_list

        if in_channels is not None:
            assert num_layers >= 1
            channel_list = [hidden_channels] * (num_layers - 1)
            channel_list = [in_channels] + channel_list + [out_channels]

        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        self.channel_list = channel_list

        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.act_first = act_first
        self.plain_last = plain_last

        if isinstance(dropout, float):
            dropout = [dropout] * (len(channel_list) - 1)
            if plain_last:
                dropout[-1] = 0.
        if len(dropout) != len(channel_list) - 1:
            raise ValueError(
                f"Number of dropout values provided ({len(dropout)} does not "
                f"match the number of layers specified "
                f"({len(channel_list)-1})")
        self.dropout = dropout

        if isinstance(bias, bool):
            bias = [bias] * (len(channel_list) - 1)
        if len(bias) != len(channel_list) - 1:
            raise ValueError(
                f"Number of bias values provided ({len(bias)}) does not match "
                f"the number of layers specified ({len(channel_list)-1})")

        self.lins = torch.nn.ModuleList()
        iterator = zip(channel_list[:-1], channel_list[1:], bias)
        for in_channels, out_channels, _bias in iterator:
            self.lins.append(Linear(in_channels, out_channels, bias=_bias))

        self.norms = torch.nn.ModuleList()
        iterator = channel_list[1:-1] if plain_last else channel_list[1:]
        for hidden_channels in iterator:
            if norm is not None:
                norm_layer = normalization_resolver(
                    norm,
                    hidden_channels,
                    **(norm_kwargs or {}),
                )
            else:
                norm_layer = Identity()
            self.norms.append(norm_layer)

        self.reset_parameters()

    @property
    def in_channels(self) -> int:
        r"""Size of each input sample."""
        return self.channel_list[0]

    @property
    def out_channels(self) -> int:
        r"""Size of each output sample."""
        return self.channel_list[-1]

    @property
    def num_layers(self) -> int:
        r"""The number of layers."""
        return len(self.channel_list) - 1

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()


    def forward(self, x: Tensor, return_emb: NoneType = None) -> Tensor:
        """"""
        for i, (lin, norm) in enumerate(zip(self.lins, self.norms)):
            x = lin(x)
            if self.act is not None and self.act_first:
                x = self.act(x)
            emb = x
            x = norm(x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout[i], training=self.training)
            
        if self.plain_last:
            x = self.lins[-1](x)
            x = F.dropout(x, p=self.dropout[-1], training=self.training)

        return (x, emb) if isinstance(return_emb, bool) else x


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self.channel_list)[1:-1]})'

class LREvaluator(BaseEvaluator):
    def __init__(self, num_epochs: int = 2000, learning_rate: float = 0.01,
                 weight_decay: float = 0.0, test_interval: int = 100):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval

    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict, model_path: str,):
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.max().item() + 1
        # Which classifier to use?
        classifier = LogisticRegression(input_dim, num_classes).to(device)
        #classifier = MLP(in_channels = 256, hidden_channels = 64, out_channels = num_classes, num_layers=2, norm=None, act=torch.nn.PReLU()).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = nn.LogSoftmax(dim=-1)
        criterion = nn.NLLLoss()

        best_val_micro = 0
        best_test_acc = 0
        best_test_micro = 0
        best_test_macro = 0
        best_epoch = 0

        with tqdm(total=self.num_epochs, desc='(LR)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
            for epoch in range(self.num_epochs):
                classifier.train()
                optimizer.zero_grad()
                output = classifier(x[split['train']])
                
                loss = criterion(output_fn(output), y[split['train']])

                loss.backward()
                optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()
                    #y_train = y[split['train']].detach().cpu().numpy()
                    #y_pred = classifier(x[split['train']]).argmax(-1).detach().cpu().numpy()
                    #train_acc = (y_train == y_pred).sum() / y_train.shape[0]
                    
                    y_test = y[split['test']].detach().cpu().numpy()
                    y_pred = classifier(x[split['test']]).argmax(-1).detach().cpu().numpy()                
                    test_micro = f1_score(y_test, y_pred, average='micro')
                    test_macro = f1_score(y_test, y_pred, average='macro')
                    test_acc = (y_test == y_pred).sum()/y_test.shape[0]

                    y_val = y[split['valid']].detach().cpu().numpy()
                    y_pred = classifier(x[split['valid']]).argmax(-1).detach().cpu().numpy()
                    val_micro = f1_score(y_val, y_pred, average='micro')
                    val_acc = (y_val == y_pred).sum() / y_val.shape[0]
                    
                    #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                    #      f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

                    if val_acc > best_val_micro:
                        best_val_micro = val_acc
                        best_test_micro = test_micro
                        best_test_macro = test_macro
                        best_test_acc = test_acc
                        best_epoch = epoch
                        torch.save(classifier.state_dict(), model_path + 'GBT_classifier_best.pth')

                    pbar.set_postfix({'best test F1Mi': best_test_micro, 'F1Ma': best_test_macro})
                    pbar.update(self.test_interval)
        
        print('LREvaluator FINISH HERE')
        classifier.load_state_dict(torch.load(model_path + 'GBT_classifier_best.pth'))
        with torch.no_grad():
            y_soft = classifier(x)
            y_soft = output_fn(y_soft)
        
        dict_out = {
            'micro_f1': best_test_micro,
            'acc': best_test_acc,
            'macro_f1': best_test_macro,
            'acc_val':best_val_micro,
        }

        return dict_out, y_soft
