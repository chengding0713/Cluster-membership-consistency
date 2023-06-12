import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from sklearn.utils import shuffle

class Dataset():
    def __init__(self,data_path,label_path):
        # self.root = root
        self.data_path = data_path
        self.label_path = label_path
        self.dataset,self.labelset= self.build_dataset()
        self.length = self.dataset.shape[0]
        # self.minmax_normalize()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        step = self.dataset[idx,:]
        step = torch.unsqueeze(step, 0)
        # target = self.label[idx]
        target = self.labelset[idx]
        target = torch.unsqueeze(target, 0)# only one class
        return step, target

    def build_dataset(self):
        '''get dataset of signal'''

        dataset = np.load(self.data_path)
        labelset = np.load(self.label_path)

        # dataset,labelset = shuffle(dataset,labelset)
        dataset = torch.from_numpy(dataset)
        labelset = torch.from_numpy(labelset)

        return dataset,labelset

class Dataset_AE():
    def __init__(self,data_path,label_path,cluster_path):
        # self.root = root
        self.data_path = data_path
        self.label_path = label_path
        self.cluster_path = cluster_path
        self.dataset,self.labelset,self.clusterset= self.build_dataset()
        self.length = self.dataset.shape[0]
        # self.minmax_normalize()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        step = self.dataset[idx,:]
        step = torch.unsqueeze(step, 0)
        # target = self.label[idx]
        target = self.labelset[idx]
        target = torch.unsqueeze(target, 0)

        cluster = self.clusterset[idx]
        cluster = torch.unsqueeze(cluster,0)
        return step, target,cluster

    def build_dataset(self):
        '''get dataset of signal'''

        dataset = np.load(self.data_path)
        labelset = np.load(self.label_path)
        clusterset = np.load(self.cluster_path)

        # dataset,labelset,clusterset = shuffle(dataset,labelset,clusterset)
        dataset = torch.from_numpy(dataset)
        labelset = torch.from_numpy(labelset)
        clusterset = torch.from_numpy(clusterset)

        return dataset,labelset,clusterset

if __name__ == '__main__':
    dataset = Dataset_AE('/opt/localdata/storage/chengding_project_data/alarm_data_npy/x_train2.npy',
                      '/opt/localdata/storage/chengding_project_data/alarm_data_npy/y_train.npy',
                      '/opt/localdata/storage/chengding_project_data/alarm_data_npy/y_cluster.npy' )
    train_loader = DataLoader(dataset,batch_size=100,shuffle=True)
    for i, batch in enumerate(train_loader):
        print(i,batch)
