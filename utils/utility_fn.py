import scipy
import torch
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt

def extract_data_matrix_from_adata(adata, use_rep=None, torch_tensor=True, data_dtype=torch.float32, device='cpu'):
    """
    """

    if use_rep is not None:
        feature_matrix = adata.obsm[use_rep]
    elif isinstance(adata.X, scipy.sparse.spmatrix): 
        feature_matrix = adata.X.todense()
    else:
        feature_matrix = adata.X
        
    if torch_tensor:
        try:
            feature_matrix = torch.tensor(feature_matrix, dtype=data_dtype, device=device)  
        except ValueError as e:
            # Check if the error is due to negative strides
            if "negative strides" in str(e):
                print("Caught ValueError due to negative strides in the given numpy array. Transform it into contiguous array.")
                feature_matrix= np.ascontiguousarray(feature_matrix)
                feature_matrix = torch.tensor(feature_matrix, dtype=data_dtype, device=device) 
            else:
                raise e
        
    return feature_matrix


def evaluate_node_isolation(adata, use_rep='X_fae', predict_pct=0.1):
    
    feature_matrix = extract_data_matrix_from_adata(adata, use_rep=use_rep, torch_tensor=False)
    
    clf = IsolationForest(random_state=0, contamination=predict_pct).fit(feature_matrix)
    node_IF_labels = clf.predict(feature_matrix)  # Get the anomaly labels for each data point
    
    adata.obs['isolation'] = node_IF_labels
    adata.obs['isolation'] = adata.obs['isolation'].astype('category')


##
### Graph edges related functions
##


def feature_to_knn_indices(feature_matrix_src, feature_matrix_trg=None, k_min=None, k_max=None, self_included=True):
    """
    """
    feature_matrix_trg = feature_matrix_src if feature_matrix_trg is None else feature_matrix_trg
    
    k_min = 0 if k_min is None else k_min
    k_max = feature_matrix_trg.shape[0] if k_max is None else k_max
    
    dist_matrix = torch.cdist(feature_matrix_src, feature_matrix_trg, p=2)
    knn_indices = torch.argsort(dist_matrix, dim=1)
    
    if self_included==False:
        if torch.equal(feature_matrix_trg, feature_matrix_src):
            knn_indices = remove_self_edges_from_knn_indices(knn_indices)
    
    knn_indices = knn_indices[:, k_min:k_max]  
        
    return knn_indices


def remove_self_edges_from_knn_indices(knn_indices):
    
    num_rows = knn_indices.size(0)

    # Generate a mask where each element is compared with its row index
    mask = torch.arange(num_rows, device=knn_indices.device).unsqueeze(1) != knn_indices

    # Apply the mask to filter the tensor
    filtered_knn_indices = knn_indices[mask].view(num_rows, -1)
    
    return filtered_knn_indices


def knn_indices_to_edge_index(knn_indices):
    """
    Convert a knn_indices tensor to an edge_index tensor.

    Args:
        knn_indices (torch.Tensor): A tensor of shape (num_points, k) containing the indices of 
                                    the k-nearest neighbors for each point.

    Returns:
        torch.Tensor: An edge_index tensor of shape (2, num_edges) representing the edges in the graph.
    """
    device=knn_indices.device
    
    num_points, k = knn_indices.shape

    # Create source and target node index tensors
    src_nodes = torch.arange(num_points, device=device).view(-1, 1).repeat(1, k).view(-1)
    trg_nodes = knn_indices.reshape(-1)

    # Concatenate the source and target node index tensors to create the edge_index tensor
    edge_index = torch.stack([src_nodes, trg_nodes], dim=0)

    return edge_index


def remove_repetition_edges(edge_index):

    # Remove repetitions
    edges = torch.cat([edge_index[0, :].unsqueeze(0), edge_index[1, :].unsqueeze(0)], dim=0).T
    # Remove repetitions and sort the edges
    unique_edges = edges.unique(dim=0)
    # Split the unique edges back into source and target tensors
    non_repetition_edge_index = unique_edges.T.contiguous()
    
    return non_repetition_edge_index

def remove_self_edges(edge_index):

    mask = edge_index[0] != edge_index[1]
    non_self_edge_index = edge_index[:, mask]
    
    return non_self_edge_index


def extract_mnn_edge_index(edge_index):
    
    N_nodes = torch.max(edge_index).item()
    
    divid = 1
    for ii in str(abs(N_nodes)):
        divid = 10 * divid
        
    edge_index_double = edge_index.double()
        
    edge_label_1 = edge_index_double[0,:] + (edge_index_double[1,:]/divid)
    edge_label_2 = edge_index_double[1,:] + (edge_index_double[0,:]/divid)
        
    mask = torch.isin(edge_label_1, edge_label_2)
    mutual_edge_index = edge_index[:, mask]
    
    return mutual_edge_index


def limit_outgoing_edges(edge_index, max_edges=10):
    # Extract source nodes and target nodes
    sources = edge_index[0]
    targets = edge_index[1]
    
    # Unique nodes in the source
    unique_nodes = sources.unique()

    # To store new edges
    new_sources = []
    new_targets = []

    # Iterate over each unique node
    for node in unique_nodes:
        # Find indices where this node is a source
        indices = (sources == node).nonzero(as_tuple=True)[0]
        
        # If more than max_edges are found, we need to randomly select max_edges to keep
        if indices.shape[0] > max_edges:
            keep_indices = indices[torch.randperm(indices.shape[0])[:max_edges]]
        else:
            keep_indices = indices
        
        # Append the selected edges to the new lists
        new_sources.extend(sources[keep_indices].tolist())
        new_targets.extend(targets[keep_indices].tolist())

    # Create the new edge index tensor
    new_edge_index = torch.tensor([new_sources, new_targets], device=edge_index.device)
    return new_edge_index    


def prune_edges_with_IF_labels(edge_index, node_IF_labels):
    
    node_IF_labels = torch.tensor(node_IF_labels, device=edge_index.device)
    
    normal_nodes = torch.where(node_IF_labels == 1)[0]
    anomalous_nodes = torch.where(node_IF_labels == -1)[0]
    
    mask_1 = torch.isin(edge_index[0,:], anomalous_nodes)
    mask_2 = torch.isin(edge_index[1,:], normal_nodes)
    
    mask = mask_1 & mask_2

    pruned_edge_index = edge_index[:,~mask]
    
    return pruned_edge_index

def node_edges_count(edge_index, num_nodes):
    
    device = edge_index.device

    # Count outgoing edges
    outgoing_counts = torch.zeros(num_nodes, dtype=torch.int, device=device)
    outgoing_counts.index_add_(0, 
                               edge_index[0], 
                               torch.ones(edge_index[0].size(0), 
                                          dtype=torch.int, 
                                          device=device))
    
    # Count incoming edges
    incoming_counts = torch.zeros(num_nodes, dtype=torch.int, device=device)
    incoming_counts.index_add_(0, 
                               edge_index[1], 
                               torch.ones(edge_index[1].size(0), 
                                          dtype=torch.int, 
                                          device=device))

    return outgoing_counts, incoming_counts



def unique_colors(unique_values):
    if len(unique_values) <= 20:
        cmap = plt.get_cmap('tab20')
        color_map = {value: cmap(i % 20) for i, value in enumerate(unique_values)}
        colors = np.array([color_map[val] for val in unique_values])

    elif len(unique_values) <= 120:

        tab20 = plt.get_cmap('tab20').colors
        tab20b = plt.get_cmap('tab20b').colors
        tab20c = plt.get_cmap('tab20c').colors
        tab20_r = plt.get_cmap('tab20_r').colors
        tab20b_r = plt.get_cmap('tab20b_r').colors
        tab20c_r = plt.get_cmap('tab20c_r').colors
        extended_colors = list(tab20) + list(tab20b) + list(tab20c) + list(tab20_r) + list(tab20b_r) + list(tab20c_r)

        colors = extended_colors[:len(unique_values)]

    else:
        cmap = plt.get_cmap('viridis', len(unique_values))
        color_map = {cell_type: cmap(i) for i, cell_type in enumerate(unique_values)}
        colors = np.array([color_map[val] for val in unique_values])
        
    return colors
