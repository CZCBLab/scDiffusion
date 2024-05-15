import torch
import numpy
import math

from scipy.spatial.distance import pdist, squareform



def build_trajectory(scdata, use_graph="graph", use_attention=True, regulate_cell=True, similarity_adjust=True, trg_adjust=True, use_around_community=True, around_community=1.0, tune_pos=None, tune_pos_scale=1.0, origin_cluster=None, use_position=False):
    """
    """
    
    scdata.to_device('cpu')
    scdata.to_numpy()
    
    # cluster_labels shape: N, feature_umap shape: (N,2)
    cluster_labels, feature_umap, graph_object, feature_mtx = extract_scdata(scdata, use_graph=use_graph)
    
    scdata.traj.cluster = cluster_labels
    
    num_community = max(cluster_labels) + 1
    
    # Three lists: nodes, number of nodes, umap in/for every community
    community, community_size, community_umap = build_community(cluster_labels, feature_umap, num_community)
    
    if tune_pos is not None:
        community_umap = tune_community_umap(numpy.array(community_umap), tune_pos=tune_pos, tune_pos_scale=tune_pos_scale)
    
    scdata.traj.expr = numpy.array(community_umap)
    scdata.traj.nodes = community_size 
    
    # shape: num_community * num_community
    community_distance_matrix = get_community_distance(feature_mtx, cluster_labels, num_community)
    
    # shape: num_community * num_community
    community_attention_matrix = get_community_attention(community, num_community, community_size, 
                                                         graph_object, regulate_cell=regulate_cell,
                                            use_attention=use_attention, trg_adjust=trg_adjust)
    
    # Build trajectory using attention matrix
    
    if origin_cluster is not None:
        community_edge_list, similarity_list, trajectory_weight_list = attention_to_traj_with_origin(origin_cluster, num_community, community_attention_matrix, community_distance_matrix, use_position=use_position)
    
    elif use_around_community:
        community_edge_list, similarity_list, trajectory_weight_list = attention_to_traj_use_around_community(num_community,
                                                                           community_attention_matrix, 
                                                                           regulate_cell=True,
                                                                           around_community=around_community)
    else:
        community_edge_list, similarity_list, trajectory_weight_list = attention_to_traj(num_community, 
                                                community_attention_matrix, regulate_cell=regulate_cell)

    # Trajectory weight normalization
    similarity_list, trajectory_weight_list = traj_normalization(similarity_list, trajectory_weight_list, 
                                                            similarity_adjust=similarity_adjust)
                    
    # Load trajectory to scdata
    smlr_edge_list = []
    traj_edge_list = []
    for i in range(len(community_edge_list)):
        smlr_edge_list.append((community_edge_list[i][0], community_edge_list[i][1], similarity_list[i])) 
        traj_edge_list.append((community_edge_list[i][0], community_edge_list[i][1], trajectory_weight_list[i])) 
    
    scdata.traj.smlr_edge_list = smlr_edge_list
    scdata.traj.traj_edge_list = traj_edge_list
    
    return scdata
    

def extract_scdata(scdata, use_graph="graph"):
    
    # list of cell cluster labels, length = number of cells
    assert len(scdata.cluster) > 3, f'Clustering results is requested. Please do clustering first.'
    cluster_labels = scdata.cluster
    
    # UMAP embedding of features
    assert scdata.gae.umap_out.shape[1]==2, f'UMAP embedding of "gae.output" is requested. Please do UMAP first.'
    feature_umap = scdata.gae.umap_out
    
    # select graph to build trajectory
    if use_graph == "graph":
        graph_object = scdata.graph
    elif use_graph == "raw_graph":
        graph_object = scdata.raw_graph
    elif use_graph == "dif_graph":
        graph_object = scdata.dif_graph
    elif use_graph == "graph_le":
        graph_object = scdata.graph_le
    elif use_graph == "graph_ge":
        graph_object = scdata.graph_ge
    else:
        graph_object = None
        raise Exception(f'use_graph should be one of ("graph", "raw_graph", "dif_graph", "graph_le", "graph_ge") but got {use_graph}.')
    
    feature_mtx = scdata.gae.output
    
    return cluster_labels, feature_umap, graph_object, feature_mtx


def build_community(cluster_labels, feature_umap, num_community):
    
    all_nodes = numpy.array(range(len(cluster_labels)))
    # basic cluster informations
    community = []
    community_size = []
    community_umap = []
    for cluster in range(num_community):
        cluster_filter = numpy.where(cluster_labels == cluster, True, False)
        community.append(all_nodes[cluster_filter])
        community_size.append(len(community[cluster]))
        
        cluster_inside_nodes_umap = feature_umap[cluster_filter,:]
        community_umap.append(numpy.mean(cluster_inside_nodes_umap, axis=0))
        
    return community, community_size, community_umap


def tune_community_umap(community_umap, tune_pos=0.3, tune_pos_scale=1.0):
    
    range_1 = community_umap[:, 0].max() - community_umap[:, 0].min()
    range_2 = community_umap[:, 1].max() - community_umap[:, 1].min()
    
    distances = pdist(community_umap, metric='euclidean')

    # Convert the condensed distance matrix to a square matrix
    distance_matrix = squareform(distances)

    threshold = tune_pos*numpy.mean(distances)

    # Extract node pairs with distance below the mean distance
    node_pairs_below = []

    for i in range(distance_matrix.shape[0]):
        for j in range(i+1, distance_matrix.shape[1]):
            if distance_matrix[i][j] < threshold:
                node_pairs_below.append((i, j))

    for node_pairs in node_pairs_below:

        umap_1 = community_umap[node_pairs[0]]
        umap_2 = community_umap[node_pairs[1]]
        umap_dif = umap_2 - umap_1
        distance = math.sqrt(umap_dif[0]*umap_dif[0] + umap_dif[1]*umap_dif[1])
        if distance < threshold:
            difference = threshold - distance
            axis_1 = difference * umap_dif[0]/distance
            axis_2 = difference * umap_dif[1]/distance

            if axis_1 > 0:
                community_umap[community_umap[:, 0] >=umap_2[0], 0] += axis_1
            else:
                community_umap[community_umap[:, 0] <=umap_2[0], 0] += axis_1

            if axis_2 > 0:
                community_umap[community_umap[:, 1] >=umap_2[1], 1] += axis_2
            else:
                community_umap[community_umap[:, 1] <=umap_2[1], 1] += axis_2   
        else:
            pass
    
    new_range_1 = community_umap[:, 0].max() - community_umap[:, 0].min()
    new_range_2 = community_umap[:, 1].max() - community_umap[:, 1].min()
    
    community_umap[:, 0] *= tune_pos_scale*range_1/new_range_1
    community_umap[:, 1] *= tune_pos_scale*range_2/new_range_2
        
    return community_umap


def get_community_distance(feature_mtx, cluster_labels, num_community):
    
    community_distance_matrix = numpy.zeros((num_community,num_community))  
    
    for cluster_src in range(num_community):
        
        filt = (cluster_labels == cluster_src)
        cluster_src_features = feature_mtx[filt]
        
        for cluster_trg in range(num_community):
            filt = (cluster_labels == cluster_trg)
            cluster_trg_features = feature_mtx[filt]
            
            community_distance_matrix[cluster_src][cluster_trg] = single_linkage_distance(cluster_src_features, cluster_trg_features)
    
    return community_distance_matrix

def centroid_distance(cluster1, cluster2):
    centroid1 = numpy.mean(cluster1, axis=0)
    centroid2 = numpy.mean(cluster2, axis=0)
    return numpy.linalg.norm(centroid1 - centroid2)

def single_linkage_distance(cluster1, cluster2):
    distances = numpy.linalg.norm(cluster1[:, numpy.newaxis] - cluster2, axis=2)
    return numpy.min(distances)

def average_linkage_distance(cluster1, cluster2):
    distances = numpy.linalg.norm(cluster1[:, numpy.newaxis] - cluster2, axis=2)
    return numpy.mean(distances)



def get_community_attention(community, num_community, community_size, graph_object, regulate_cell=True,
                            use_attention=True, trg_adjust=True):
    
    # call intercluster attentions
    community_attention_matrix = numpy.zeros((num_community,num_community))    
    
    # Attention form cluster_src to cluster_trg
    # Feature flow form cluster_src to cluster_trg
    for cluster_src in range(num_community):
        cluster_src_nodes = community[cluster_src]
        
        for cluster_trg in range(num_community):
            cluster_trg_nodes = community[cluster_trg]
            
            # edge_trg represents central nodes in KNN graph
            # Feature flow from edge_src to edge_trg
            # Attention from edge_trg to edge_src
            edge_trg = graph_object.edge_index[0]
            edge_src = graph_object.edge_index[1]
            
            # attention takes an opposite sign with the feature flow 
            edge_trg_filter = numpy.isin(edge_trg, cluster_src_nodes)
            edge_src_filter = numpy.isin(edge_src, cluster_trg_nodes)
            edge_filter = edge_trg_filter & edge_src_filter
            
            if use_attention:
                filtered_attention = graph_object.attention[edge_filter]
            else:
                filtered_attention = edge_filter.astype(int)
            
            if trg_adjust:
                cluster_attention = numpy.sum(filtered_attention)/(community_size[cluster_src]*community_size[cluster_trg])
            else:
                cluster_attention = numpy.sum(filtered_attention)/(community_size[cluster_src])
            
            community_attention_matrix[cluster_src][cluster_trg] = cluster_attention
            
    numpy.fill_diagonal(community_attention_matrix, 0)
    
#     if regulate_cell:
#         # For each row, find the indices that would sort it
#         sorted_indices = numpy.argsort(community_attention_matrix, axis=1)

#         # For each row, take all but the top three indices and set the corresponding elements to zero
#         for row in range(community_attention_matrix.shape[0]):
#             community_attention_matrix[row, sorted_indices[row, :-3]] = 0 
#     else:
#         # For each column, find the indices that would sort it
#         sorted_indices = numpy.argsort(community_attention_matrix, axis=0)

#         # For each column, take all but the top three indices and set the corresponding elements to zero
#         for col in range(community_attention_matrix.shape[1]):
#             community_attention_matrix[sorted_indices[:-3, col], col] = 0
    
    return community_attention_matrix


def attention_to_traj(num_community, community_attention_matrix, regulate_cell=True):
    community_edge_list=[]
    trajectory_weight_list=[]
    similarity_list = []
    for cluster_src in range(num_community):
        for cluster_trg in range(cluster_src+1,num_community):
            similarity_list.append(community_attention_matrix[cluster_src][cluster_trg]+community_attention_matrix[cluster_trg][cluster_src])
            trajectory_weight = community_attention_matrix[cluster_src][cluster_trg]-community_attention_matrix[cluster_trg][cluster_src]
            if regulate_cell:
                if trajectory_weight < 0:
                    community_edge_list.append((cluster_src, cluster_trg))
                    trajectory_weight_list.append(-trajectory_weight)
                else:
                    community_edge_list.append((cluster_trg, cluster_src))
                    trajectory_weight_list.append(trajectory_weight)
            else:
                if trajectory_weight > 0:
                    community_edge_list.append((cluster_src, cluster_trg))
                    trajectory_weight_list.append(trajectory_weight)
                else:
                    community_edge_list.append((cluster_trg, cluster_src))
                    trajectory_weight_list.append(-trajectory_weight)
                    
    return community_edge_list, similarity_list, trajectory_weight_list


def attention_to_traj_use_around_community(num_community, community_attention_matrix, 
                                           regulate_cell=True, around_community=1.0):
    community_edge_list=[]
    trajectory_weight_list=[]
    similarity_list = []
    for cluster_src in range(num_community):
        for cluster_trg in range(cluster_src+1,num_community):
            similarity_list.append(community_attention_matrix[cluster_src][cluster_trg]+community_attention_matrix[cluster_trg][cluster_src])
            trajectory_weight = community_attention_matrix[cluster_src][cluster_trg]-community_attention_matrix[cluster_trg][cluster_src]
            
            
            # use around community
            traj_around_weight=0
            for cluster in range(num_community):
                if cluster==cluster_src:
                    pass
                elif cluster==cluster_trg:
                    pass
                else:
                    similarity_src = community_attention_matrix[cluster_src][cluster]+community_attention_matrix[cluster][cluster_src]
                    similarity_trg = community_attention_matrix[cluster_trg][cluster]+community_attention_matrix[cluster][cluster_trg]
                    if similarity_src > (1.2*similarity_trg):
                        traj_around_weight = community_attention_matrix[cluster][cluster_trg]-community_attention_matrix[cluster_trg][cluster]
                        
                    elif (1.2*similarity_src) < similarity_trg:
                        traj_around_weight = community_attention_matrix[cluster_src][cluster]-community_attention_matrix[cluster][cluster_src]
                        
            
            trajectory_weight = trajectory_weight + around_community*traj_around_weight           
            
            
            if regulate_cell:
                if trajectory_weight < 0:
                    community_edge_list.append((cluster_src, cluster_trg))
                    trajectory_weight_list.append(-trajectory_weight)
                else:
                    community_edge_list.append((cluster_trg, cluster_src))
                    trajectory_weight_list.append(trajectory_weight)
            else:
                if trajectory_weight > 0:
                    community_edge_list.append((cluster_src, cluster_trg))
                    trajectory_weight_list.append(trajectory_weight)
                else:
                    community_edge_list.append((cluster_trg, cluster_src))
                    trajectory_weight_list.append(-trajectory_weight)
                    
    return community_edge_list, similarity_list, trajectory_weight_list

def attention_to_traj_with_origin(origin_cluster, num_community, community_attention_matrix, community_distance_matrix, use_position=True):
    
    distance_mean = community_distance_matrix.sum()/(num_community*(num_community-1))
    
    similarity_list = []     
    community_edge_list=[]
    trajectory_weight_list=[]
    
    traj_dict = {}

    community_nodes = numpy.array(range(num_community))
    count_list = []
    count_list.append(origin_cluster) 
    community_nodes = community_nodes[community_nodes != origin_cluster]
    
    
    
    for i in range(num_community-1):
        
        cluster_now = get_next_cluster(community_attention_matrix, community_nodes, count_list)
            
        cluster_past, similarity_past = get_origin_cluster(community_attention_matrix, 
                                                           community_distance_matrix, 
                                    cluster_now, count_list, use_position=use_position)
            
        try:
            cluster_1 = int(traj_dict[str(cluster_past)])
            
            similarity = get_similarity(community_attention_matrix, community_distance_matrix, 
                                        cluster_now, cluster_1, use_position=use_position)
            
            similarity_o = get_similarity(community_attention_matrix, community_distance_matrix, 
                                        cluster_past, cluster_1, use_position=use_position)
            
            
            if (similarity_past < (2*similarity)) or (similarity_o < (2*similarity)):
                similarity_past = similarity
                cluster_past = cluster_1

                
#                 try:
#                     cluster_1 = int(traj_dict[str(cluster_1)])

#                     similarity = get_similarity(community_attention_matrix, community_distance_matrix, 
#                                         cluster_now, cluster_1, use_position=use_position)
            
#                     similarity_o = get_similarity(community_attention_matrix, community_distance_matrix, 
#                                                     cluster_past, cluster_1, use_position=use_position)
#                     print(cluster_1)

#                     if similarity_past < (similarity*similarity_past/similarity_o):
#                         print(cluster_1)
#                         similarity_past = similarity
#                         cluster_past = cluster_1
#                     else:
#                         pass

#                 except:
#                     pass
                
            else:
                pass
            
            
            
            
        except:
            pass
            
                    
        
        community_edge_list.append((cluster_past, cluster_now))
        trajectory_weight_list.append(1) 
        similarity_list.append(similarity_past)
        
        count_list.append(cluster_now) 
        community_nodes = community_nodes[community_nodes != cluster_now]
        traj_dict[str(cluster_now)] = str(cluster_past)
        
                    
    return community_edge_list, similarity_list, trajectory_weight_list


def attention_to_traj_with_origin_use_around_community(origin_cluster, num_community, community_attention_matrix, community_distance_matrix, use_position=True):
    
    distance_mean = community_distance_matrix.sum()/(num_community*(num_community-1))
    
    similarity_list = []     
    community_edge_list=[]
    trajectory_weight_list=[]
    
    traj_dict = {}

    community_nodes = numpy.array(range(num_community))
    count_list = []
    count_list.append(origin_cluster) 
    community_nodes = community_nodes[community_nodes != origin_cluster]
    
    
    
    for i in range(num_community-1):
        
        cluster_now = get_next_cluster(community_attention_matrix, community_nodes, count_list)
        
        community_nodes = community_nodes[community_nodes != cluster_now]
        
        
            
        cluster_past, similarity_past = get_origin_cluster(community_attention_matrix, 
                                                           community_distance_matrix, 
                                    cluster_now, count_list, use_position=use_position)
            
            
        try:
            cluster_1 = int(traj_dict[str(cluster_past)])
            
#             for around_clst in community_nodes:
#                 if 
                
            similarity = get_similarity(community_attention_matrix, community_distance_matrix, 
                                        cluster_now, cluster_1, use_position=use_position)
            
            similarity_o = get_similarity(community_attention_matrix, community_distance_matrix, 
                                        cluster_past, cluster_1, use_position=use_position)
            
            
            if (similarity_past < (2*similarity)) or (similarity_o < (2*similarity)):
                similarity_past = similarity
                cluster_past = cluster_1
                
            else:
                pass    
            
        except:
            pass
            
                    
        
        community_edge_list.append((cluster_past, cluster_now))
        trajectory_weight_list.append(1) 
        similarity_list.append(similarity_past)
        
        count_list.append(cluster_now) 
        
        traj_dict[str(cluster_now)] = str(cluster_past)
        
                    
    return community_edge_list, similarity_list, trajectory_weight_list

def get_next_cluster(community_attention_matrix, community_nodes, count_list):
    
    cluster_now = -1
    similarity_now = 0
    for cluster_src in community_nodes:     
        similarity = 0
        for cluster_trg in count_list:
            similarity += (community_attention_matrix[cluster_src][cluster_trg]+community_attention_matrix[cluster_trg][cluster_src])
        if similarity_now < similarity:
            similarity_now = similarity.copy()
            cluster_now = cluster_src
        
    return cluster_now

def get_origin_cluster(community_attention_matrix, community_distance_matrix, 
                                    cluster_now, count_list, use_position=True):

    cluster_past = -1
    similarity_past = 0
    for cluster_trg in count_list:

        similarity = get_similarity(community_attention_matrix, community_distance_matrix, 
                                    cluster_now, cluster_trg, use_position=use_position)

        if similarity_past < similarity:
            similarity_past = similarity.copy()
            cluster_past = cluster_trg
        else:
            pass
        
    return cluster_past, similarity_past
        


def get_similarity(community_attention_matrix, community_distance_matrix, cluster_src, cluster_trg, use_position=True):
    
    if use_position:
        similarity = (community_attention_matrix[cluster_src][cluster_trg]+community_attention_matrix[cluster_trg][cluster_src])
        distance = community_distance_matrix[cluster_src][cluster_trg]

        similarity= similarity/(distance+2*distance_mean)

    else:
        similarity = (community_attention_matrix[cluster_src][cluster_trg]+community_attention_matrix[cluster_trg][cluster_src])

    return similarity
        

def traj_normalization(similarity_list, trajectory_weight_list, similarity_adjust=True):    
                    
    max_value = max(similarity_list)
    similarity_list = [num / max_value for num in similarity_list]
    #similarity_list = [math.sqrt(num)/10 for num in similarity_list]
    
    if similarity_adjust:
        similarity_mean = sum(similarity_list)/len(similarity_list)
        trajectory_weight_list = [x / (0.01 + abs(y-0.05)) for x, y in zip(trajectory_weight_list, similarity_list)]
    
    max_value = max(trajectory_weight_list)
    trajectory_weight_list = [100*num/max_value for num in trajectory_weight_list]
    trajectory_weight_list = [math.sqrt(num)/10 for num in trajectory_weight_list]
    
    return similarity_list, trajectory_weight_list



