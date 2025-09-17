import scipy
import scipy.io
import torch
import numpy as np
from torch_geometric.data import Data
from collections import defaultdict
import argparse
import pandas as pd
import os

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

def main():
    parser = argparse.ArgumentParser(description="Convert NCANDA .mat files to PyTorch Geometric data.")
    parser.add_argument('--inputs', type=str, nargs='+', required=True, help='List of input .mat file paths.')
    parser.add_argument('--labels', type=str, required=True, help='Path to the labels .mat file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output .pt file.')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of labels for classification (default: 2).')
    parser.add_argument('--label_column', type=str, default='cddr15a', help='The column name in the labels file to use.')
    parser.add_argument('--threshold', type=float, default=0.05, help='Proportional threshold for connectivity matrix (default: 0.05).')
    parser.add_argument('--ROIs', type=int, default=500, help='The number of ROIs examined (default 500).')
    args = parser.parse_args()
    # Load input data
    input_matrices = []
    for path in args.inputs:
        mat_file = scipy.io.loadmat(path)
        # Find the variable name in the .mat file, ignoring metadata
        var_name = [k for k in mat_file.keys() if not k.startswith('__')][0]
        input_matrices.append(mat_file[var_name])

    # Concatenate all input matrices
    TasksAll = np.concatenate(input_matrices, axis=2)

    # Load labels
    labels_mat = scipy.io.loadmat(args.labels)
    labels_var_name = [k for k in labels_mat.keys() if not k.startswith('__')][0]
    labels_array = labels_mat[labels_var_name]

    # Process labels
    cddr15a = pd.Series(labels_array[:, 0].flatten())
    nan_indices = cddr15a[cddr15a.isna()].index.tolist()
    cleaned_column = cddr15a.dropna().tolist()

    # Remove subjects with NaN labels from the data
    TasksAll = np.delete(TasksAll, nan_indices, axis=2)

    print(f'Tasks shape: {TasksAll.shape}')
    print(f'Cleaned Column Length: {len(cleaned_column)}')

    AdjMats = TasksAll
    GraphsNum = AdjMats.shape[2]

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

        x = torch.tensor(Adj_i, dtype=torch.float32)

        row, col = np.where(Adj_i > 0)
        mask = row != col
        row, col = row[mask], col[mask]
        edge_index = torch.tensor([row, col], dtype=torch.long)

        x_all.append(x)
        edge_index_all.append(edge_index)
        labels.append(torch.tensor(cleaned_column[i], dtype=torch.long))

        node_offsets.append(node_offset)
        edge_offsets.append(edge_offset)

        node_offset += x.size(0)
        edge_offset += edge_index.size(1)

    x_all = torch.vstack(x_all)
    edge_index_all = torch.cat(edge_index_all, dim=1)
    y_all = torch.vstack(labels)

    data2['x'] = torch.tensor(node_offsets + [x_all.size(0)])
    data2['edge_index'] = torch.tensor(edge_offsets + [edge_index_all.size(1)])
    data2['y'] = y_all

    TorchGraph_Data = Data(x=x_all, edge_index=edge_index_all, y=y_all)
    data = (TorchGraph_Data, data2)

    output_filename = f'NCandaData{args.ROIs}_{args.label_column}_{int(args.threshold * 100)}pct.pt'
    output_path = os.path.join(args.output_dir, output_filename)
    
    torch.save(data, output_path)
    print(f"Saved data to {output_path}")

if __name__ == "__main__":
    main()