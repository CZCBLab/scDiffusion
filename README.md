# About:

scDiffusion(Single-Cell graph neural Diffusion) is a physics-informed graph generative model to do scRNA-seq analysis. scDiffusion investigates cellular dynamics utilizing an attention-based neural network. Unlike methods focusing solely on gene expression in individual cells, scDiffusion targets cell-cell association graph by incorporating two distinct physical effects: local and global equilibrium. It has great potential to apply to multiple scenarios in scRNA-seq data analysis. To better understand the model, we implement it to clustering analysis guided by attention-weighted modularity and trajectory prediction directed by inter-cluster attention network. We demonstrate the balance between local and global equilibrium effects are particularly beneficial for clustering and trajectory determination. Within latent clusters, the local equilibrium effect amplifies the attention-weighted modularity during the diffusion process, resulting to improved clustering accuracy. Simultaneously, the global equilibrium effect strengthens inter-relationships among different clusters, aiding in the accurate prediction of trajectories. More importantly, the diffusion model provides an comprehensible and effective method to do data integration. We show scDiffusion can accurately remove batch effects that caused by technical differences with cell type imbalance. As a deep learning neural network with solid mathematical foundations and rich physical explanations, scDiffusion provided a comprehensive generative model based on cell graph diffusion and showed great potential in scRNA-seq data analysis both theoretically and practically.


# Installation:

Grab this source codes:
```
git clone https://github.com/...
cd scDiffusion
```
Python=3.9.9 is required. See other requirements in the file requirements.txt.

# Tutorials:

For clustering tasks, please check the notebook file "scDiffusion_clustering_Klein.ipynb". 
