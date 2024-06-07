import pickle
import torch
import numpy as np
import random
import os

import argparse
from model import *
import timeit
os.chdir(os.path.dirname(os.path.abspath(__file__)))
def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy')]

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def init_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True

from sklearn.model_selection import StratifiedKFold
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DrugBank Training')
    parser.add_argument('--model_name', type=str, default='P31644', help='The name of models')
    parser.add_argument('--protein_dim', type=int, default=100, help='embedding dimension of proteins')
    parser.add_argument('--atom_dim', type=int, default=34, help='embedding dimension of atoms')
    parser.add_argument('--hid_dim', type=int, default=64, help='embedding dimension of hidden layers')
    parser.add_argument('--n_layers', type=int, default=3, help='layer count of networks')
    parser.add_argument('--n_heads', type=int, default=8, help='the head count of self-attention')
    parser.add_argument('--pf_dim', type=int, default=256, help='dimension of feedforward neural network')
    parser.add_argument('--dropout', type=float, default=0.2, help='the ratio of Dropout')
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--iteration', type=int, default=200, help='the iteration for training')
    parser.add_argument('--n_folds', type=int, default=5, help='the fold count for cross-entropy')
    parser.add_argument('--seed', type=int, default=2023, help='the random seed')
    parser.add_argument('--kernel_size', type=int, default=9, help='the kernel size of Conv1D in transformer')
    parser.add_argument('--save_name', type=str, default='test', help='the kernel size of Conv1D in transformer')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    init_seed(args.seed)
    with open(f'../data/{args.model_name}.pickle',"rb") as f:
        dataset = pickle.load(f)


    labels = [i[-3] for i in dataset]

    best_epoch, best_AUC_test, best_AUPR_test, best_precision_test, best_recall_test = 0.0, 0.0, 0.0, 0.0, 0.0


    results = np.array([0.0]*4)

    model = Predictor(args.hid_dim, args.n_layers, args.kernel_size, args.n_heads, args.pf_dim, args.dropout, device, args.atom_dim, args.protein_dim)
    model.to(device)
    pretrained_model = torch.load("../model/independent-test_4.pt")
    model.load_state_dict(pretrained_model)
    tester = Tester(model)
    fold = 1
    file_AUCs1 = f'../result/drug_feature.txt'
    file_AUCs2 = f'../result/target_feature.txt'

    AUC_dev, PRC_dev, PRE_dev, REC_dev, t, y, s, trace1= tester.test(dataset)
    print(s)
    indices_of_top_10 = np.argsort(s)[-10:]

    # 输出结果
    print("数组S中最大的10个数的下标:", indices_of_top_10)

    s111 = np.array([s])  # 请用实际的数组替换 your_array_here

# 提取数组s中的前20个数

# 将整个数据排序
sorted_data = sorted(s, reverse=True)

# 查找数组s中每个数在排序后的数据中的位置
rankings = [sorted_data.index(x) + 1 for x in s]

# 打印结果
print("原始数组s中前20个数在整个数据中的排名：", rankings)
print(len(s))
print(s[984],s[1074],s[212],s[76],s[704],s[364],s[324],s[723],s[965],s[993],)