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


def check_isolation(adata, use_rep='X_fae', predict_pct=0.1):
    
    feature_matrix = extract_data_matrix_from_adata(adata, use_rep=use_rep, torch_tensor=False)
    
    clf = IsolationForest(random_state=0, contamination=predict_pct).fit(feature_matrix)
    node_IF_labels = clf.predict(feature_matrix)  # Get the anomaly labels for each data point
    
    adata.obs['isolation'] = node_IF_labels
    adata.obs['isolation'] = adata.obs['isolation'].astype('category')

    return adata
