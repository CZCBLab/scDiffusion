"""
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from grand.gnd import GND
from utils import info_log
from utils.utility_fn import extract_data_matrix_from_adata

def graph_diffusion(adata, use_rep='X_fae', max_epoch=1000, lr=1e-3, device='cpu',
                           num_features_diffusion=128,
                           num_heads_diffusion=6,
                           num_steps_diffusion=8, 
                           time_increment_diffusion=0.5,
                           attention_type = 'sum', 
                           activation=nn.ELU(),
                           data_dtype = torch.float32,
                           dropout=0.0, 
                           encoder=None, 
                           decoder=None,
                           log_diffusion=False,
                           save_model = True,
                           load_model_state = False,
                           loss_adj=0.0,
                           use_adj='adj_edge_index',
                           loss_reduction = "sum",
                           rebuild_graph=False,
                           rebuild_graph_args={
                               'k_min': 0,
                               'k_max': 10,
                               'remov_edge_prob': None,
                           }
                   ):
    
    diffusion_args = {"use_rep": use_rep,
                       "num_features_diffusion": num_features_diffusion,
                       "num_heads_diffusion": num_heads_diffusion,
                       "num_steps_diffusion": num_steps_diffusion, 
                       "time_increment_diffusion": time_increment_diffusion,
                       "attention_type": attention_type, 
                       "dropout": dropout, 
                       "log_diffusion": log_diffusion,
                       "encoder": encoder, 
                       "decoder": decoder,
                       "save_model": save_model,
                       "load_model_state": load_model_state,
                       "loss_adj": loss_adj}
    
    
    
    info_log.print('--------> Starting Graph AE ...')
    
    # data
    feature_matrix = extract_data_matrix_from_adata(adata, use_rep=use_rep, torch_tensor=True, 
                                                    data_dtype=data_dtype, device=device)
    num_of_nodes = feature_matrix.shape[0]
    
    edge_index = torch.tensor(adata.uns['diffusion_edge_index'], dtype=torch.int64, device=device)
    
    use_adj = 'diffusion_edge_index' if use_adj is None else use_adj
    adj_edge_index = torch.tensor(adata.uns[use_adj], dtype=torch.int64)
    adjacency = edge_index_to_adj(adj_edge_index, num_of_nodes).to(device)
        
    target_features = extract_data_matrix_from_adata(adata, use_rep=None, torch_tensor=True, 
                                                     data_dtype=data_dtype, device=device)

    D_in = feature_matrix.shape[1]
    D_out = target_features.shape[1]
    
    if encoder is None:
        encoder = None if D_in==num_features_diffusion else [D_in, num_features_diffusion]
    else:
        encoder = [D_in] + encoder + [num_features_diffusion]
    
    if decoder is None:
        decoder = None if D_out==num_features_diffusion else [num_features_diffusion, D_out]
    else:
        decoder = [num_features_diffusion] + decoder + [D_out]
        

    model_dif = Graph_DIF(num_features_diffusion = num_features_diffusion, 
                           num_heads_diffusion=num_heads_diffusion,
                           num_steps_diffusion= num_steps_diffusion, 
                           time_increment_diffusion=time_increment_diffusion,
                           attention_type = attention_type, 
                           activation=activation,
                           dropout=dropout, 
                           log_diffusion=log_diffusion,
                           encoder=encoder, 
                           decoder=decoder,
                           rebuild_graph=rebuild_graph).to(device)

    if load_model_state:
        try: 
            state_dict_torch = {k: torch.tensor(v).to(device) for k, v in adata.uns['gnd_state_dict'].items()}
            model_dif.load_state_dict(state_dict_torch)
        except:
            print("Graph autoencoder failed to load model state.")
                            
    optimizer = torch.optim.Adam(model_dif.parameters(), lr=lr)

    for epoch in range(max_epoch):
        model_dif.train()
        optimizer.zero_grad()
        
        if rebuild_graph:
            data = (feature_matrix, rebuild_graph_args)
        else:
            data = (feature_matrix, edge_index)

        out_nodes_features, recon_adj, last_embedding = model_dif(data)
        
        target_1 = torch.tensor(adjacency, dtype = recon_adj.dtype)
        target_2 = torch.tensor(target_features, dtype = out_nodes_features.dtype)
        
        if loss_adj==1.0:
            loss = F.binary_cross_entropy_with_logits(recon_adj, target_1, reduction=loss_reduction)
        elif loss_adj==0.0:
            loss = F.mse_loss(out_nodes_features, target_2, reduction=loss_reduction)
        else:
            loss_1 = F.binary_cross_entropy_with_logits(recon_adj, target_1, reduction=loss_reduction)
            loss_2 = F.mse_loss(out_nodes_features, target_2, reduction=loss_reduction)
            
            fold = loss_1.item()/loss_2.item()
            loss = loss_adj*loss_1 + (1.0-loss_adj)*fold*loss_2
        

        # Backprop and Update
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()
        
        if epoch%50 == 0:
            info_log.interval_print(f"----------------> Epoch: {epoch+1}/{max_epoch}, Current loss: {cur_loss:.4f}")
    
    info_log.interval_print(f"----------------> Epoch: {epoch+1}/{max_epoch}, Current loss: {cur_loss:.4f}")

    # save model state
    if save_model:
        state_dict_numpy = {k: v.detach().cpu().numpy() for k, v in model_dif.state_dict().items()}
        adata.uns['gnd_state_dict'] = state_dict_numpy

        
    if log_diffusion:
        adata.uns['gnd_steps_data'] = []
        for it in range(len(model_dif.diffusion_step_outputs)):
            adata.uns['gnd_steps_data'].append(model_dif.diffusion_step_outputs[it].numpy())
        
    adata.obsm['X_dif'] = last_embedding.detach().cpu().numpy()
    
    adata.uns['graph_diffusion_args'] = diffusion_args
        
    return adata 


class Graph_DIF(nn.Module):
    def __init__(self, num_features_diffusion,
                           num_heads_diffusion,
                           num_steps_diffusion, 
                           time_increment_diffusion,
                           attention_type = 'sum', 
                           activation=nn.ELU(),
                           dropout=0.0,  
                           log_diffusion=False,
                           encoder=None, 
                           decoder=None,
                           rebuild_graph=False):
        super().__init__()
        
        self.log_diffusion=log_diffusion
        
        self.attention_weights = None
        self.diffusion_step_outputs = None
        
        self.diffusion = GND(num_features_diffusion = num_features_diffusion, 
                           num_heads_diffusion=num_heads_diffusion,
                           num_steps_diffusion= num_steps_diffusion, 
                           time_increment_diffusion=time_increment_diffusion,
                           attention_type = attention_type, 
                           activation=activation,
                           dropout=dropout, 
                           log_diffusion=log_diffusion,
                           encoder=encoder, 
                           decoder=decoder,
                           rebuild_graph=rebuild_graph)

        self.decode = InnerProductDecoder(0, act=lambda x: x)
        #self.decode = InnerProductDecoder(0, act=torch.sigmoid)


    def forward(self, data):
        # [0] just extracts the node_features part of the data (index 1 contains the edge_index)
        
        data, last_embedding = self.diffusion(data)
        
        out_nodes_features, edge_index = data
        
        recon_adj = self.decode(out_nodes_features)
            
        if self.log_diffusion:
            self.diffusion_step_outputs = self.diffusion.diffusion_step_outputs
        
        return out_nodes_features, recon_adj, last_embedding
        
    
    
class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super().__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj
    
def edge_index_to_adj(edge_index, num_of_nodes):
    """
    construct adjacency matrix from edge index
    """
    adjacency_matrix = torch.zeros((num_of_nodes, num_of_nodes), dtype=edge_index.dtype, device=edge_index.device)
    adjacency_matrix[edge_index[0], edge_index[1]] = 1

    return adjacency_matrix

