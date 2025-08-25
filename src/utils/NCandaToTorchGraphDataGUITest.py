'''
# %%
import torch
data = torch.load('data.pt')

# %%
data

# %%
data, data2 = torch.load('data.pt')

# %%
data

# %%
data2

# %%
print(data)               
print(data.x.shape)       
print(data.edge_index.shape)  
print(data.y.shape)      

# %%
node_offsets = data2['x']
edge_offsets = data2['edge_index']
labels = data2['y']
node_offsets, edge_offsets

# %%
i = 0
start_node = node_offsets[i].item()
end_node = node_offsets[i+1].item() if i+1 < len(node_offsets) else adjAll.size(0)
start_node, end_node

# %%
x = data['x'][start_node:end_node]
x.shape , x

# %%
i = 0
edge_indeadjAll = data.edge_index

start_edge = edge_offsets[i].item()
end_edge = edge_offsets[i+1].item() if i+1 < len(edge_offsets) else edge_indeadjAll.size(1)
start_edge, end_edge

edge_index = edge_indeadjAll[:, start_edge:end_edge] - start_node 
edge_index.shape, edge_index

# %%

'''
# %%
import scipy
import scipy.io
import torch
import numpy as np
from torch_geometric.data import Data
from collections import defaultdict
import argparse
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_labels', type=int, default=2) #default assumes binary classification
parser.add_argument('--label_column', type=str, default='cddr15a')
parser.add_argument('--threshold', type=float, default=0.05)
args = parser.parse_args()

def threshold_proportional(W: np.ndarray, p: float = 0.05) -> np.ndarray: #python version of BCT function originally written in MATLAB
    """
    Thresholds a connectivity matrix by retaining the top p proportion of strongest weights.
    
    Parameters:
        W (np.ndarray): Square connectivity matrix (symmetric or asymmetric).
        p (float): Proportion of strongest weights to retain (0 < p < 1).
    
    Returns:
        np.ndarray: Thresholded matrix with only top p weights retained.
    """
    W = W.copy()
    n = W.shape[0]
    np.fill_diagonal(W, 0)  # Remove self-connections

    symmetric = np.allclose(W, W.T, atol=1e-10)
    if symmetric:
        W = np.triu(W)  # Work with upper triangle only
        ud = 2
    else:
        ud = 1

    # Get indices and values of non-zero elements
    inds = np.transpose(np.nonzero(W))
    weights = W[W != 0]
    sorted_inds = inds[np.argsort(-np.abs(weights))]  # Sort by descending absolute weight

    num_edges_to_keep = int(round((n**2 - n) * p / ud))
    keep_inds = sorted_inds[:num_edges_to_keep]

    # Create new thresholded matrix
    W_thr = np.zeros_like(W)
    for i, j in keep_inds:
        W_thr[i, j] = W[i, j]

    if symmetric:
        W_thr = W_thr + W_thr.T  # Restore symmetry

    return W_thr


# %%
DataPath1 = '/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDA_CorrMats500ROIs_1.mat'
DataPath2 = '/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDA_CorrMats500ROIs_2.mat'
DataPath3 = '/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDA_CorrMats500ROIs_3.mat'
DataPath4 = '/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/NCANDA_CorrMats500ROIs_4.mat'
LabelsPath = '/isilon/datalake/lcbn_research/final/beach/JonathanP/DataNew_500And998ROIs/LabelsTotal_500.mat'

# %%
NCanda1 = scipy.io.loadmat(DataPath1)
NCanda2 = scipy.io.loadmat(DataPath2)
NCanda3 = scipy.io.loadmat(DataPath3)
NCanda4 = scipy.io.loadmat(DataPath4)

labels = scipy.io.loadmat(LabelsPath)


NCanda1 = NCanda1['NCANDA_CorrMats500ROIs_1']
NCanda2 = NCanda2['NCANDA_CorrMats500ROIs_2']
NCanda3 = NCanda3['NCANDA_CorrMats500ROIs_3']
NCanda4 = NCanda4['NCANDA_CorrMats500ROIs_4']
labels_array = labels['LabelsTotal_500']

cddr15a = pd.Series(labels_array[:, 0].flatten())
nan_indices = cddr15a[cddr15a.isna()].index.tolist()
cleaned_column = cddr15a.dropna().tolist()


'''
X = np.concatenate([Task1, Task2, Task3, Task4, Task5, Task6, Task7], axis=0)
#X = np.concatenate([Task1, Task2, Task3], axis=2)

del Task1
gc.collect()
del Task2
gc.collect()
del Task3
gc.collect()

del Task4
gc.collect()
del Task5
gc.collect()
del Task6
gc.collect()
del Task7
gc.collect()
'''

# %%
#Task1.shape, Task2.shape, Task3.shape
#[0,0,0, ..., 0,1,....,1,2,...,6] #size = 794*7

# %%
TasksAll = np.concatenate([NCanda1, NCanda2, NCanda3, NCanda4], axis=2)

TasksAll = np.delete(TasksAll, nan_indices, axis=2)

print(f'Tasks shape: {TasksAll.shape}')
print(f'Cleaned Column Length: {len(cleaned_column)}')


# %%
CorrMatsAll = TasksAll 
AdjMats = CorrMatsAll #(CorrMatsAll > 0.5).astype(np.uint8)  
AdjMats.shape

# %%
NodesNum = AdjMats.shape[0]
GraphsNum = AdjMats.shape[2]

# %%
i = 0
Adj_i = AdjMats[:, :, i]
x = torch.tensor(Adj_i, dtype=torch.float32)
x.shape
row, col = np.where(Adj_i > 0)
mask = row != col
row, col = row[mask], col[mask]
edge_index = torch.tensor([row, col], dtype=torch.long)
edge_index

# %%


# %%
x_all = []
edge_index_all = []
labels = []

data2 = defaultdict(dict)
node_offsets = []
edge_offsets = []

node_offset = 0
edge_offset = 0

for i in range(GraphsNum):
    Adj_i = threshold_proportional(AdjMats[:, :, i], args.threshold)

    # threshold Aij - use their function based density thresholoding
    
    x = torch.tensor(Adj_i, dtype=torch.float32)

    row, col = np.where(Adj_i > 0)
    mask = row != col
    row, col = row[mask], col[mask]
    edge_index = torch.tensor([row, col], dtype=torch.long)

    x_all.append(x)
    edge_index_all.append(edge_index)
    #labels.append(torch.tensor(i))
    num_labels = args.num_labels
    labels.append(torch.tensor(cleaned_column[i], dtype=torch.long))


    node_offsets.append(node_offset)
    edge_offsets.append(edge_offset)

    node_offset += x.size(0)
    edge_offset += edge_index.size(1)

# %%


# %%
x_all = torch.vstack(x_all)                        
edge_index_all = torch.cat(edge_index_all, dim=1)  
y_all = torch.vstack(labels)                       

data2['x'] = torch.tensor(node_offsets + [x_all.size(0)])
data2['edge_index'] = torch.tensor(edge_offsets + [edge_index_all.size(1)])
data2['y'] = y_all

TorchGraph_Data = Data(x=x_all, edge_index=edge_index_all, y=y_all)
data = (TorchGraph_Data, data2)

torch.save(data, f'GUItest_NCandaData500{args.label_column}_{int(args.threshold * 100)}pct.pt')

# %%
data