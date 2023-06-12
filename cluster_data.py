import numpy as np
from sklearn.cluster import KMeans
import time
import datetime
from scipy.signal import resample
AF = np.float16(np.load('/opt/localdata/storage/stark_stuff/ppg_ecg_project/data/AF_v5/train_PPG_resampled2400.npy'))
NSR = np.float16(np.load('/opt/localdata/storage/stark_stuff/ppg_ecg_project/data/NSR_v5/train_PPG_resampled2400.npy'))
PVC = np.float16(np.load('/opt/localdata/storage/stark_stuff/ppg_ecg_project/data/PVC_v5/train_PPG_resampled2400.npy'))
data = np.concatenate([AF,PVC,NSR])
data = np.float16(data)
data = resample(data,300,axis=1)
print(data.shape)
# data = np.load('/opt/localdata/storage/chengding_project_data/alarm_data_npy/x_train_600.npy')
#
for cluster_num in [2]:
    print('starting clustering...%d'%cluster_num)
    print(datetime.datetime.now())
    kmeans = KMeans(n_clusters=cluster_num, random_state=1).fit(data)
    print('Clustering finished...%d'%cluster_num)
    print(datetime.datetime.now())
    labels = kmeans.labels_
    np.save('/opt/localdata/storage/chengding_project_data/alarm_data_npy/y_cluster_2400_%d.npy'%cluster_num,labels)