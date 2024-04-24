import scipy
import torch
from sklearn.ensemble import IsolationForest

def extract_data_matrix_from_adata(adata, use_rep=None, torch_tensor=True, data_dtype=torch.float32, device='cpu'):

    if use_rep is not None:
        feature_matrix = adata.obsm[use_rep]
    elif isinstance(adata.X, scipy.sparse.spmatrix): 
        feature_matrix = adata.X.todense()
    else:
        feature_matrix = adata.X
        
    if torch_tensor:
        feature_matrix = torch.tensor(feature_matrix, dtype=data_dtype, device=device)  
        
    return feature_matrix


def check_isolation(adata, use_rep='X_fae', predict_pct=0.1):
    
    if use_rep is None:
        clf = IsolationForest(random_state=0, contamination=predict_pct).fit(adata.X)
        node_IF_labels = clf.predict(adata.X)  # Get the anomaly labels for each data point
    else:
        clf = IsolationForest(random_state=0, contamination=predict_pct).fit(adata.obsm[use_rep])
        node_IF_labels = clf.predict(adata.obsm[use_rep])  # Get the anomaly labels for each data point
    
    adata.obs['isolation'] = node_IF_labels
    adata.obs['isolation'] = adata.obs['isolation'].astype('category')

    return adata
