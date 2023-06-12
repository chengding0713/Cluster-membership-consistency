# Cluster-membership-consistency
# Model Card for cluster membership consistency (CMC)

**CMC** loss is designed to ensure that the latent space of a neural network has a natural cluster structure, which makes the model more  robust. It uses the idea that the label varies smoothly along the  manifold of observable PPG signals within each cluster. This manifold  of realistic PPG signals is mapped to a space that is easy to work  with using an autoencoder. We first perform clustering on the latent   space of a trained autoencoder to get the assigned cluster membership  for each PPG record. This clustering is unsupervised, and thus does not suffer from label noise; it captures only the natural cluster structure of the PPG signals. We then use the cluster membership to  regularize the feature representation learned through the    convolutional layers of a second neural network. More specifically,  pairwise distances of points within the same cluster (intra-cluster    distance) will be minimized and pairwise distances of points in    different clusters (inter-cluster distance) will be maximized. We   then train a ResNet-34 deep neural network with a loss function that combines the CMC loss and the cross-entropy (CE) loss. This new network thus has a robust latent structure that leverages the natural  clustering structure of the signal and helps alleviate the impact of  label noise.

## Intended to use

The model is designed for the AF detection task. The input is supposed to be 30-second of PPG signals, with a sampling rate 240. 

## How to use

To make prediction on new data samples, you need to install the PyTorch package and numpy package. You will also need the resnet1d.py and dataset.py. Here we provide a sample script loading the trained model weights and makeing prediction on an example dataset in npy format:

```python
import numpy as np
from resnet1d import Resnet34
from Dataset import Dataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import datetime
import os
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, recall_score, precision_score, \
    f1_score,roc_curve,auc
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

batch_size = 500
epochs = 50
weight_file = 'path-to-weight-file'
# CHECKPOINT_PATH = 'E:/Generate_MI/SCEmodel_cpsweights_2_1.0_0.0_3/'
#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# load it to the specified device, either gpu or cpu
model = Resnet34(num_classes=1).to(device)

test_dataset = Dataset('test_data.npy','test_label.npy')
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

model.load_state_dict(torch.load(weight_file)['net'])
prediction = []
prediction_bin = []
label = []

for batch_idx, (inputs, targets) in enumerate(test_loader):
    # Copy inputs to device
    inputs = inputs.to(device).float()
    targets = targets.to(device).float()
    # Generate output from the DNN.
    outputs,features = model(inputs)
    label.append(targets.detach().cpu().numpy())
    prediction.append(outputs.view(-1,1).detach().cpu().numpy())
    # Calculate predicted labels
    predicted = (outputs >=0.5).float()
    prediction_bin.append(predicted.view(-1,1).detach().cpu().numpy())

label = np.concatenate(label)
prediction = np.concatenate(prediction)

prediction_bin = np.concatenate(prediction_bin)
fpr, tpr, thresholds = roc_curve(label,prediction)
idx = np.argmax(tpr - fpr)
tt = thresholds[idx]
f1socre = f1_score(label,prediction_bin)
print('Accuracy:',accuracy_score(label,prediction_bin))
print('F1 Score:', f1socre)
print('AUC:', auc(fpr, tpr))
```

## Training data

## Training dataset

The training date contains 28539 patients in hospital settings; the patients' continuous ECG and PPG signals were recorded from the bedside monitors. The bedside monitor produced alarms for events including atrial fibrillation (AF), premature ventricular contraction (PVC), Tachy, Couplet, etc. This study focuses on AF, PVC, and normal sinus rhythm (NSR). The samples with PVC and NSR labels were combined into the Non-AF samples group, thus forming the AF vs Non-AF binary classification task. PPG signals were sliced into 30-second non-overlap segments (each containing 7,200 timesteps).  The dataset is split into the train and validation splits by patient ids. The train split of the Institution A dataset contains 13,432 patients, 2,757,888 AF signal segments, and 3,014,334 Non-AF signal segments; the validation split contains 6,616 patients, 1,280,775 AF segments, and 1,505,119 Non-AF segments. Due to the automatic nature of bedside monitor-generated labels, the dataset likely contains label noise.

## Testing datasets

### Institution B dataset

The Institution B dataset contains 126 patients in hospital settings, and simultaneous continuous ECG and PPG signals were collected at Institution B. The patients have a minimum age of 18 and a maximum age of 95 years old and were admitted from April 2010 to March 2013. The continuous signals were sliced into 30-second non-overlapping segments and downsampled to 2,400 timesteps. The dataset contains 38,910 AF and 220,740 Non-AF segments. A board-certified cardiac electrophysiologist annotated all AF episodes in the Institution B datasets.

### Simband dataset

The Simband dataset contains 98 patients in ambulatory settings from Emory University Hospital (EUH), Emory University Hospital Midtown (EUHM), and Grady Memorial Hospital (GMH). The patients have a minimum age of 18 years old and a maximum age of 89 years old; patients were admitted from October 2015 to March 2016. The ECG signals were collected using Holter monitors, and the PPG signals were collected from a wrist-worn Samsung Simband. The signals used for testing were 30-second segments with 7200 timesteps after pre-processing. This dataset contains 348 AF segments and 506 Non-AF segments.

### Stanford dataset 

The Stanford dataset contains 107 AF patients, 15 paroxysmal AF patients, and 42 healthy patients. The 42 healthy patients also undergo an exercise stress test. All signals in this dataset were recorded in ambulatory settings. The ECG signals were collected from an ECG reference device, and the PPG signals were collected from a wrist-worn device. The signals were sliced into 25-second segments by the original author. In this study, the signals were also upsampled to 7200 timesteps. The dataset contains 52,911 AF segments and 80,620 Non-AF segments. In the evaluations, we use the test split generated by the authors of the Stanford dataset.

## Evaluation results

This model is evaluated with both AUROC, it achieved the following performance on test sets:
| Method   | Institution B dataset|  Simband dataset |Stanford dataset  |
|--|--| -- | -- |
| CE | 0.924 ±  0.02  |	0.836 ±  0.01|0.585 ±  0.01 |
| SCE | 0.929 ± 0.02  |	0.843 ± 0.03|0.558 ± 0.01 |
| Co-teaching | 0.905 ±  0.01  |	0.824 ±  0.02|0.539 ±  0.01 |
| INCV | **0.932 ± 0.01**  |	0.861 ± 0.01|0.605 ± 0.01 |
| DivideMix | 0.931 ±  0.01  |	0.891 ± 0.01|**0.737 ±  0.01** |
| ELR| 0.860 ±  0.02  |	0.811 ±  0.01|0.566 ±  0.01 |
| SOP | 0.930 ± 0.02 |	0.887 ± 0.02|0.661 ± 0.01 |
| CMC (Ours)| **0.932 ± 0.02** |	**0.910 ±**  **0.02**|0.735 ± 0.01 |
















