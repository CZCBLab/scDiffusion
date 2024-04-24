import scanpy as sc
import umap
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import info_log
from anndata_metric.concat import concat_adatas, divide_adata

def umap_embedding(adata, 
                   data = 'gnd.output',  # 'gnd.input', 'gnd.output', 'gnd.steps'
                   method='umap',        # 'umap', 'scanpy'
                   X_umap=False,
                   umap_args = {'n_neighbors': 15,
                                'min_dist': 0.3,
                                'method': 'umap',
                                'metric': 'correlation',
                                'random_state': 2021}):
    if method == 'umap':
    
        if data == 'gnd.output':
            adata.obsm['gnd_output_umap'] = umap_hidden(adata.uns['gnd_steps_data'][-1], umap_args=umap_args)
            if X_umap:
                adata.obsm['X_pca'] = adata.uns['gnd_steps_data'][-1]
                adata.obsm['X_umap'] = adata.obsm['gnd_output_umap']
        elif data == 'gnd.input':
            adata.obsm['gnd_input_umap'] = umap_hidden(adata.uns['gnd_steps_data'][0], umap_args=umap_args)
        elif data == 'gnd.steps':
            adata.uns['gnd_steps_umap'] = []
            for feature_mtx in adata.uns['gnd_steps_data']:
                 adata.uns['gnd_steps_umap'].append(umap_hidden(feature_mtx, umap_args=umap_args))
    
    elif method == 'scanpy':
        
        adata_new = adata.copy()
    
        if data == 'gnd.output':
            adata_new.obsm['X_pca'] = adata_new.uns['gnd_steps_data'][-1]
            sc.pp.neighbors(adata_new, n_neighbors=umap_args['n_neighbors'], n_pcs=50)
            sc.tl.umap(adata_new, min_dist=umap_args['min_dist'], 
                           method=umap_args['method'],
                           random_state=umap_args['random_state'])
            adata.obsm['gnd_output_umap'] = adata_new.obsm['X_umap']
            if X_umap:
                adata.obsm['X_pca'] = adata.uns['gnd_steps_data'][-1]
                adata.obsm['X_umap'] = adata.obsm['gnd_output_umap']
        elif data == 'gnd.input':
            adata_new.obsm['X_pca'] = adata_new.uns['gnd_steps_data'][0]
            sc.pp.neighbors(adata_new, n_neighbors=umap_args['n_neighbors'], n_pcs=50)
            sc.tl.umap(adata_new, min_dist=umap_args['min_dist'], 
                           method=umap_args['method'],
                           random_state=umap_args['random_state'])
            adata.obsm['gnd_input_umap'] = adata_new.obsm['X_umap']
        elif data == 'gnd.steps':
            adata.uns['gnd_steps_umap'] = []
            for feature_mtx in adata.uns['gnd_steps_data']:
                adata_new.obsm['X_pca'] = feature_mtx
                sc.pp.neighbors(adata_new, n_neighbors=umap_args['n_neighbors'], n_pcs=50)
                sc.tl.umap(adata_new, min_dist=umap_args['min_dist'], 
                           method=umap_args['method'],
                           random_state=umap_args['random_state'])
                adata.uns['gnd_steps_umap'].append(adata_new.obsm['X_umap'])
                
                adata.obsm['gnd_output_umap'] = adata.uns['gnd_steps_umap'][-1]
                adata.obsm['gnd_input_umap'] = adata.uns['gnd_steps_umap'][0]
                
                if X_umap:
                    adata.obsm['X_pca'] = adata.uns['gnd_steps_data'][-1]
                    adata.obsm['X_umap'] = adata.obsm['gnd_output_umap']
                
                    
    return adata



def umap_hidden(data, umap_args = {'n_neighbors': 15,
                                'min_dist': 0.3,
                                'metric': 'correlation',
                                'random_state': 2021}):
 
    reducer = umap.UMAP(n_neighbors=umap_args['n_neighbors'],
                      min_dist=umap_args['min_dist'],
                      metric=umap_args['metric'],
                      random_state=umap_args['random_state'])
    embedding= reducer.fit_transform(data) 
    
    return embedding