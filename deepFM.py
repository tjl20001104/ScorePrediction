import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from itertools import chain
from tqdm import tqdm
import pdb



class FM(nn.Module):
    def __init__(self, num_user, num_book, latent_dim, n_hidden_1, n_hidden_2, n_output_1, n_output_2):
        """
        latent_dim: 各个离散特征隐向量的维度
        input_shape: 这个最后离散特征embedding之后的拼接和dense拼接的总特征个数
        feature_1_user: 用户带bias的embedding vector 1 x latent_dim
        feature_1_book: 书带bias的embedding vector 1 x latent_dim
        feature_high_user: 用户高阶特征向量 1 x n_output_1
        feature_high_book: 书高阶特征向量 1 x n_output_1
        """
        super(FM, self).__init__()
        self.latent_dim = latent_dim
        # 定义三个矩阵， 一个是全局偏置，一个是一阶权重矩阵， 一个是二阶交叉矩阵，注意这里的参数由于是可学习参数，需要用nn.Parameter进行定义
        self.bias_user = nn.Parameter(torch.ones([1, latent_dim]))
        self.bias_book = nn.Parameter(torch.ones([1, latent_dim]))
        self.emb_user = nn.Embedding(num_user,latent_dim)
        self.emb_book = nn.Embedding(num_book,latent_dim)
        self.network_user = nn.Sequential(
            nn.Linear(latent_dim, n_hidden_1),
            nn.ReLU(),
            nn.Linear(n_hidden_1, n_output_1)
        )
        self.network_book = nn.Sequential(
            nn.Linear(latent_dim, n_hidden_1),
            nn.ReLU(),
            nn.Linear(n_hidden_1, n_output_1)
        )
        self.net_similatiry = nn.Sequential(
            nn.Linear(2*n_output_1 + 2*latent_dim, n_hidden_2),
            nn.ReLU(),
            nn.Linear(n_hidden_2, n_output_2)
        )
 
    def forward(self, inputs):
        feature_1_user = self.emb_user(inputs[0]) + self.bias_user
        feature_1_book = self.emb_book(inputs[0]) + self.bias_book

        feature_high_user = self.network_user(feature_1_user)
        feature_high_book = self.network_book(feature_1_book)

        feature_all = torch.cat([feature_1_user, feature_1_book, feature_high_user, feature_high_book],1)
        score = self.net_similatiry(feature_all)

        return score

    def params(self):
        params = [self.bias_user,
                  self.bias_book,
                  self.emb_user.parameters(),
                  self.emb_book.parameters(),
                  self.network_user.parameters(),
                  self.network_book.parameters(),
                  self.net_similatiry.parameters()]
        return filter(lambda p: p.requires_grad, chain(*params))


class Ratingdataset(Dataset):
    def __init__(self, data_path, user2idx, book2idx):
        df = pd.read_csv(data_path, header=None, index_col=None)
        df = df.drop_duplicates()
        self.user2idx = user2idx
        self.book2idx = book2idx
        self.data = df.values
        self._len = df.shape[0]

    def __getitem__(self, id_index):
        user_idx = self.user2idx[self.data[id_index,0]]
        book_idx = self.book2idx[self.data[id_index,1]]
        rate = self.data[id_index,2]
        return user_idx,book_idx,rate

    def __len__(self):
        return self._len



epoch = 10
batch_size = 1000
lr = 1e-4
latent_dim=50
n_hidden_1=50
n_hidden_2=30
n_output_1=25
n_output_2=1
alpha = 0.5



df_all = pd.read_csv("Ratings.csv",header=0,index_col=None)
df_all = df_all.drop_duplicates()
user_all = df_all.iloc[:,0].unique().tolist()
user2idx = {}
book_all = df_all.iloc[:,1].unique().tolist()
book2idx = {}
for i in range(len(user_all)):
    user2idx[user_all[i]] = i
for i in range(len(book_all)):
    book2idx[book_all[i]] = i


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FM(len(user2idx), len(book2idx), latent_dim=latent_dim, 
        n_hidden_1=n_hidden_1, n_hidden_2=n_hidden_2, 
        n_output_1=n_output_1, n_output_2=n_output_2).to(device)


dataset_train = Ratingdataset("train_ratings.csv", user2idx, book2idx)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

dataset_test = Ratingdataset("test_ratings.csv", user2idx, book2idx)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)


optimizer = optim.Adam(model.parameters(), lr=lr)

print("START TRAINING USING DEVICE {}".format(device))

for it in tqdm(range(0,epoch), disable=None):
    for index,item in enumerate(dataloader_train):
        inputs = [item[i].to(device) for i in range(2)]
        ground_truth = item[2].to(device)
        preds = model(inputs)
        l2_reg = torch.tensor(0.).to(device)
        for param in model.params():
            l2_reg += torch.norm(param)
        loss = torch.norm(preds-ground_truth,p=2) + alpha * l2_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("loss={}".format(loss))
pdb.set_trace()

RMSE = 0
model.eval()
with torch.no_grad():
    for index,item in enumerate(dataloader_test):
        inputs = [item[i].to(device) for i in range(2)]
        preds = model(inputs)
        ground_truth = inputs[2].to(device)
        RMSE += torch.sum(torch.square(preds-ground_truth))
        RMSE = RMSE/dataset_test.__len__()
    print(RMSE)
