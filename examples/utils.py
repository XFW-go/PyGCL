import numpy as np
import torch
import yaml
import logging


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