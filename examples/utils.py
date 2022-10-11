import numpy as np
import torch
import yaml
import logging
from typing import Dict

from sklearn import linear_model as sk_lm
from sklearn import metrics as sk_mtr
from sklearn import model_selection as sk_ms
from sklearn import multiclass as sk_mc
from sklearn import preprocessing as sk_prep
from torch import nn
from torch_geometric.data import Data
from tqdm import tqdm

def evaluate_node_classification(
    z: torch.Tensor,
    data: Data,
    masks: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    # Normalize input
    z = sk_prep.normalize(X=z, norm="l2")

    train_mask = masks["train"]
    y = data.y.cpu().numpy()
    num_cls = y.max() + 1
    clf = train_pytorch_model(
            emb_dim=z.shape[1],
            num_cls=num_cls,
            X=z,
            y=y,
            masks=masks,
        )
    accs = {}

    # Compute accuracy on train, val and test sets
    for split in ("train", "val", "test"):
        mask = masks[split]
        y_true = y[mask]
        y_pred = clf.predict(X=z[mask])

        acc = sk_mtr.classification_report(
            y_true=y_true,
            y_pred=y_pred,
            output_dict=True,
            zero_division=0,
        )["accuracy"]

        accs[split] = acc

    return accs

def train_pytorch_model(
    emb_dim: int,
    num_cls: int,
    X: np.ndarray,
    y: np.ndarray,
    masks: Dict[str, torch.Tensor],
) -> "LogisticRegression":
    # Define parameter space
    wd = 2.0 ** np.arange(-10, 10, 2)

    best_clf = None
    best_acc = -1

    pbar = tqdm(wd, desc="Train best classifier")
    for weight_decay in pbar:
        lr_model = LogisticRegression(
            in_dim=emb_dim,
            out_dim=num_cls,
            weight_decay=weight_decay,
        )

        lr_model.fit(X[masks["train"]], y[masks["train"]])

        acc = sk_mtr.classification_report(
            y_true=y[masks["val"]],
            y_pred=lr_model.predict(X[masks["val"]]),
            output_dict=True,
            zero_division=0,
        )["accuracy"]

        if acc > best_acc:
            best_acc = acc
            best_clf = lr_model

            pbar.set_description(f"Best acc: {best_acc * 100.0:.2f}")

    pbar.close()

    return best_clf


class LogisticRegression(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, weight_decay: float):
        super().__init__()

        self.fc = nn.Linear(in_dim, out_dim)

        self._optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=0.01,
            weight_decay=weight_decay,
        )
        self._loss_fn = nn.CrossEntropyLoss()
        self._num_epochs = 1000
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        for m in self.modules():
            self.weights_init(m)

        self.to(self._device)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.fc(x)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.train()

        X = torch.from_numpy(X).float().to(self._device)
        y = torch.from_numpy(y).to(self._device)

        for _ in tqdm(range(self._num_epochs), desc="Epochs", leave=False):
            self._optimizer.zero_grad()

            pred = self(X)
            loss = self._loss_fn(input=pred, target=y)

            loss.backward()
            self._optimizer.step()

    def predict(self, X: np.ndarray):
        self.eval()

        with torch.no_grad():
            pred = self(torch.from_numpy(X).float().to(self._device))

        return pred.argmax(dim=1).cpu()

def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k or "dropout" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args


def split_dataset(data, setting):
    num_node = data.y.size(0)
    train_set = torch.where(data.train_mask)[0]
    val_set = torch.where(data.val_mask)[0]
    test_set = torch.where(data.test_mask)[0]
    if setting == 'super':
        curr_all = set(torch.cat([train_set, val_set, test_set], dim=0).data.cpu().numpy())
        all_set = set(np.arange(num_node))
        train_set_ = list(set(all_set).difference(curr_all))
        train_set = train_set_ + train_set.data.cpu().numpy().tolist()
        train_set = np.array(train_set)
        train_set = torch.from_numpy(train_set)

    return train_set, val_set, test_set

def delete_edges(edge_index, del_nodes):
    start = np.isin(edge_index[0], del_nodes)
    start_index = np.where(start == 1)[0]
    edge_index_del = np.delete(edge_index, start_index, axis=1)

    end = np.isin(edge_index_del[1], del_nodes)
    end_index = np.where(end == 1)[0]
    edge_index_final = np.delete(edge_index_del, end_index, axis=1)

    edge_index_final = torch.tensor(edge_index_final)

    return edge_index_final