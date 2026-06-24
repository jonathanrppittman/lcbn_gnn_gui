import scipy
import scipy.io
import torch
import numpy as np
from torch_geometric.data import Data
from collections import defaultdict
import argparse
import pandas as pd
import os
import numpy as np

parser = argparse.ArgumentParser(description="Convert NCANDA .mat files to PyTorch Geometric data.")
parser.add_argument('--inputs', type=str, nargs='+', required=True, help='List of input .mat file paths.')
parser.add_argument('--labels', type=str, required=True, help='Path to the labels .mat file.')
parser.add_argument('--output_dir', type=str, default=os.path.join("..", "NeuroGraph", "data", "NCanda", "raw"), help='Directory to save the output .pt file. Defaults to ../NeuroGraph/data/NCanda/raw')
parser.add_argument('--num_labels', type=int, default=2, help='Number of labels for classification (default: 2).')
parser.add_argument('--label_column', type=str, default='cddr15a', help='The column name in the labels file to use.')
parser.add_argument('--threshold', type=float, default=0.05, help='Proportional threshold for connectivity matrix (default: 0.05).')
parser.add_argument('--ROIs', type=int, default=500, help='The number of ROIs examined (default 500).')
parser.add_argument('--device', type=str, default='cuda', help='Enter either cuda or cpu into this field to use either gpu or cpu respectively.')
args = parser.parse_args()




def threshold_proportional(W: np.ndarray, p: float = 0.1) -> np.ndarray:
    """
    Python version of BCT threshold_proportional.
    Preserves a proportion p of the strongest weights.
    """

    W = W.copy().astype(float)
    n = W.shape[0]

    # Remove diagonal
    np.fill_diagonal(W, 0)

    # Symmetry check
    symmetric = np.allclose(W, W.T, atol=1e-10)
    if symmetric:
        Wu = np.triu(W)
        ud = 2
    else:
        Wu = W
        ud = 1

    # Flatten nonzero upper-tri (or full matrix if asymmetric)
    flat = Wu.ravel()
    nz_mask = flat != 0
    nz_vals = flat[nz_mask]

    # Number of edges to preserve
    k = int(round((n*n - n) * p / ud))

    if k == 0:
        return np.zeros_like(W)

    # Sort by absolute magnitude, descending
    # Use argpartition → faster than full argsort
    if k < nz_vals.size:
        topk_idx = np.argpartition(-np.abs(nz_vals), k-1)[:k]
    else:
        topk_idx = np.arange(nz_vals.size)

    # Create thresholded flattened array
    flat_thr = np.zeros_like(flat)
    nz_positions = np.where(nz_mask)[0]
    keep_positions = nz_positions[topk_idx]

    flat_thr[keep_positions] = flat[keep_positions]

    # Reshape
    W_thr = flat_thr.reshape(n, n)

    # Restore symmetry if needed
    if symmetric:
        W_thr = W_thr + W_thr.T

    return W_thr

def main():

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
    threshold = abs(args.threshold)

    if args.device == 'cuda' and torch.cuda.is_available():
        torch.set_default_device('cuda')

    for i in range(GraphsNum):
        Adj_i = threshold_proportional(AdjMats[:, :, i], threshold) if threshold < 1.0 else AdjMats[:, :, i]

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