import torch
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from multiprocessing import Pool
import os
import time
import leidenalg as la 
import igraph as ig
from collections import defaultdict, Counter, deque
import random
import multiprocessing as mp
from functools import partial
import shutil
import warnings
from sklearn.exceptions import ConvergenceWarning
import metis

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Hyperparameters
HIDDEN_DIM = 16
EPOCHS = 20
NUM_PROCESSES = mp.cpu_count()
# NUM_PROCESSES = 1
LEIDEN_ITERATIONS = 20
TEMP_DIR = "temp_embeddings"
RANDOM_SEED = 42

# Fix random seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def enforce_array_type(arr, dtype=np.float32, shape=None):
    """Ensure array has correct type and shape"""
    try:
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr, dtype=dtype)
        if arr.dtype != dtype:
            arr = arr.astype(dtype)
        if shape is not None and arr.shape != shape:
            arr = np.zeros(shape, dtype=dtype)
        if np.issubdtype(dtype, np.floating):
            arr[np.isnan(arr) | np.isinf(arr)] = 0.0
        return arr
    except:
        return np.zeros(shape, dtype=dtype) if shape else np.array([], dtype=dtype)
    

# def calculate_block_size(N, E):

    
#     avg_deg = 2 * E / N
    
#     # 目标块数：N < 10^4 时用 5 块，N > 10^6 时用 200 块
#     if N < 50000:
#         target_blocks = 5
#     elif N < 200000:
#         target_blocks = 20
#     else:
#         target_blocks = 100
    
#     # 基础块大小
#     base_block = N // target_blocks
    
#     # 密度因子
#     if avg_deg <= 10:
#         density = 0.5
#     elif avg_deg >= 30:
#         density = 1.5
#     else:
#         density = 0.5 + 0.05 * (avg_deg - 10)
    
#     # 综合
#     block = int(base_block * density)
    
#     # 边界
#     block = max(1000, min(block, 50000))
#     block = min(block, N)  # 不能超过节点数
    
#     return block


 # 1. 基础块大小（基于图规模）
# def calculate_block_size(N, E):
#     if N < 10000:
#         base = 1000
#     elif N < 50000:
#         base = 2000
#     elif N < 200000:
#         base = 5000
#     else:
#         base = 10000
    
#     # 2. 密度因子
#     avg_deg = 2 * E / N
#     density = min(1.5, max(0.5, 0.05 * avg_deg))
    
#     # 3. 综合并约束
#     block = int(base * density)
#     return max(1000, min(block, 50000, N))


# def calculate_block_size(N, E):
#     """自适应计算块大小（使用对数函数确定目标块数）"""
    
#     import math
    
#     # 配置参数
#     BYTES_PER_NODE = 100
#     TARGET_MEM_MB = 50
    
#     # 目标块数：随N对数增长，N=10^4→8块，N=10^5→16块，N=10^6→32块
#     TARGET_BLOCKS = max(4, min(int(math.log2(N) * 2), 200))
    
#     # 密度自适应
#     avg_deg = 2 * E / N
#     density = min(1.5, max(0.5, 0.05 * avg_deg))
    
#     # 内存约束
#     max_by_mem = (TARGET_MEM_MB * 1024 * 1024) // BYTES_PER_NODE
    
#     # 目标块数约束
#     block_by_target = N // TARGET_BLOCKS
    
#     # 综合
#     block = int(min(block_by_target, max_by_mem) * density)
    
#     return max(1000, min(block, 50000, N))
import math
def get_adaptive_params(N, E):
    """自适应计算块大小"""
    #自适应块大小block_size的值
    # 配置参数（详见论文Section 3.1.1）
    
    BYTES_PER_NODE = 100          # 每节点内存（字节）
    # TARGET_MEM_MB = 50             # 目标块内存（MB）
    TARGET_MEM_MB = 50           # 目标块内存（MB）
    BLOCKS_PER_CORE = 3            # 每核心期望块数
    # num_workers= 5                 # worker threads
    num_workers=5
    
    
    SPARSE_DEGREE = 1            # 稀疏图阈值 - d_sparse

    DENSE_DEGREE =  30             # 稠密图阈值 - d_dense
    SPARSE_FACTOR = 0              # 稀疏图因子 - β_min
    DENSE_FACTOR = 2                # 稠密图因子 - β_max
    # 密度自适应
    avg_deg = 2 * E / N
 
    if avg_deg<SPARSE_DEGREE:
        density=SPARSE_FACTOR
    elif avg_deg>DENSE_DEGREE:
        density=DENSE_FACTOR
    else:
        density = SPARSE_FACTOR + (DENSE_FACTOR-SPARSE_FACTOR)/(DENSE_DEGREE-SPARSE_DEGREE) * (avg_deg - SPARSE_DEGREE) 
        # log_ratio = (math.log(avg_deg) - math.log(SPARSE_DEGREE)) / (math.log(DENSE_DEGREE) - math.log(SPARSE_DEGREE))
        # density = SPARSE_FACTOR + (DENSE_FACTOR - SPARSE_FACTOR) * log_ratio
    

    print(f"avg_deg:{avg_deg}")
    print(f"density:{density}")
    
    # 内存约束
    max_by_mem = (TARGET_MEM_MB * 1024 * 1024) // BYTES_PER_NODE
    
    # 并行度约束
    max_by_parallel = N // (num_workers*BLOCKS_PER_CORE)
    
    # 综合计算并约束
    block = int(min(max_by_mem, max_by_parallel) * density)

    block_size=max(1000, min(block, 50000))

    #自适应最小社区tau的值
    d_critical = 6            # 临界点：Leskovec (2009) 发现的稀疏-稠密阈值
    # 基础值：随图规模对数增长
    base_tau = max(2, int(math.log10(N)))

    if avg_deg < d_critical:   # 极稀疏图：社区可能很小
        tau = max(2, base_tau - 1)
    else:
        tau = base_tau


    SPARSE_DEGREE = 1    # 稀疏阈值
    DENSE_DEGREE = 30     # 稠密阈值
    ALPHA_SPARSE = 2    # 稀疏时共同邻居权重大
    ALPHA_DENSE = 0    # 稠密时共同邻居权重小

    if avg_deg <= SPARSE_DEGREE:
        alpha = ALPHA_SPARSE      
    elif avg_deg >= DENSE_DEGREE:
        alpha = ALPHA_DENSE       
    else:
        slope = (ALPHA_DENSE - ALPHA_SPARSE) / (DENSE_DEGREE - SPARSE_DEGREE)  
        alpha = ALPHA_SPARSE + slope * (avg_deg - SPARSE_DEGREE)
        # log_ratio = (math.log(avg_deg) - math.log(SPARSE_DEGREE)) / (math.log(DENSE_DEGREE) - math.log(SPARSE_DEGREE))
        # # norm_squared = log_ratio ** 2  # 平方使下降更快
        # alpha = ALPHA_SPARSE + (ALPHA_DENSE - ALPHA_SPARSE) * log_ratio
    

    return block_size,tau,alpha

def split_data_by_connectivity(edge_df, all_nodes, node_degree_dict, block_size):
    """Split data based on graph connectivity"""
    # block_size = len(all_nodes) // block_num
    
    visited = set()
    adjacency = defaultdict(list)
    for u, v in edge_df[['u', 'v']].values:
        adjacency[u].append(v)
        adjacency[v].append(u)
    
    #print(f"Adjacency nodes: {len(set(adjacency.keys()))}")
    #print(f"Common nodes: {len(adjacency.keys() & all_nodes)}")
    
    sorted_nodes = sorted(all_nodes, key=lambda x: node_degree_dict.get(x, 0), reverse=True)
    #print(f"Sorted nodes: {len(sorted_nodes)}")
    
    blocks = []
    current_block = []
    
    for node in sorted_nodes:
        if node in visited:
            continue
        
        queue = deque([node])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            
            visited.add(current)
            current_block.append(current)
            if len(current_block) >= block_size:
                blocks.append(current_block)
                #print(f"Current block size: {len(current_block)}")
                current_block = []
            
            neighbors = adjacency.get(current, [])
            neighbors_sorted = sorted(neighbors, 
                                    key=lambda x: node_degree_dict.get(x, 0), 
                                    reverse=True)
            for neighbor in neighbors_sorted:
                queue.append(neighbor)
    
    if current_block:
        #print(f"Final block: {len(current_block)}")
        if len(current_block) <= block_size and len(blocks) > 0:
            if len(blocks[-1]) + len(current_block) <= block_size * 1.5:
                blocks[-1].extend(current_block)
            else:
                blocks.append(current_block)
        else:
            blocks.append(current_block)
    
    unvisited_nodes = [node for node in all_nodes if node not in visited]
    if unvisited_nodes:
        for i in range(0, len(unvisited_nodes), block_size):
            small_block = unvisited_nodes[i:i+block_size]
            if small_block:
                blocks.append(small_block)
    
    #print(f"Generated {len(blocks)} blocks")
    
    new_all_nodes = []
    for i, block in enumerate(blocks):
        #print(f"{i}. Block size: {len(block)}")
        new_all_nodes.extend(block)
    
    new_all_nodes = list(set(new_all_nodes))
    
    return blocks,new_all_nodes

# def split_data_by_pymetis(edge_df, all_nodes, num_blocks):
#     """使用 pymetis 进行图划分"""
    
#     # 构建节点映射
#     node_to_idx = {node: i for i, node in enumerate(all_nodes)}
#     idx_to_node = {i: node for node, i in node_to_idx.items()}
    
#     # 构建邻接表
#     adjacency = defaultdict(list)
#     for u, v in edge_df[['u', 'v']].values:
#         adjacency[node_to_idx[u]].append(node_to_idx[v])
#         adjacency[node_to_idx[v]].append(node_to_idx[u])
    
#     # 转换为 Metis 格式
#     adj_list = []
#     for i in range(len(all_nodes)):
#         adj_list.append(sorted(adjacency[i]))
    
#     # 调用 pymetis
#     edgecuts, parts = pymetis.part_graph(num_blocks, adjacency=adj_list)
    
#     # 转换回原始节点 ID
#     blocks_dict = defaultdict(list)
#     for idx, part in enumerate(parts):
#         node = idx_to_node[idx]
#         blocks_dict[part].append(node)
    
#     # 计算切边数量
#     cut_edges = 0
#     for u, v in edge_df[['u', 'v']].values:
#         if parts[node_to_idx[u]] != parts[node_to_idx[v]]:
#             cut_edges += 1
    
#     return list(blocks_dict.values()), cut_edges

from sklearn.cluster import SpectralClustering
def split_data_by_spectral(edge_df, all_nodes, block_size):
    """使用谱聚类作为 METIS 的替代"""
    num_blocks = len(all_nodes) // block_size
    # 构建图
     # 构建节点映射

    G = nx.Graph()
    G.add_nodes_from(all_nodes)
    node_list = sorted(all_nodes)

    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    
    # 构建行、列、数据列表
    rows = []
    cols = []
    
    for u, v in edge_df[['u', 'v']].values:
        i = node_to_idx[u]
        j = node_to_idx[v]
        rows.append(i)
        cols.append(j)
        rows.append(j)  # 无向图需要对称
        cols.append(i)
    
    # 构建稀疏邻接矩阵
    from scipy.sparse import csr_matrix
    data = np.ones(len(rows))
    adj_matrix = csr_matrix((data, (rows, cols)), 
                           shape=(len(all_nodes), len(all_nodes)))
    
    # 谱聚类（使用稀疏矩阵版本）
    from sklearn.cluster import SpectralClustering
    sc = SpectralClustering(n_clusters=num_blocks,
                           affinity='precomputed',
                           random_state=42,
                           n_init=10,
                           assign_labels='kmeans')
    labels = sc.fit_predict(adj_matrix)
    
    # 构建块
    blocks_dict = defaultdict(list)
    for i, label in enumerate(labels):
        blocks_dict[label].append(node_list[i])
    
    # 计算切边数量
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    cut_edges = 0
    for u, v in edge_df[['u', 'v']].values:
        if labels[node_to_idx[u]] != labels[node_to_idx[v]]:
            cut_edges += 1
    
    return list(blocks_dict.values()), cut_edges

def split_data_by_metis(edge_df, all_nodes, block_size):
    """使用Metis进行结构感知划分"""
    # 构建NetworkX图
    G = nx.Graph()
    G.add_nodes_from(all_nodes)
    
    num_blocks = len(all_nodes) // block_size
    # 添加边
    edges = []
    for u, v in edge_df[['u', 'v']].values:
        G.add_edge(u, v)
        edges.append((u, v))
    
    # 将节点映射到连续索引（Metis要求）
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    
    # 构建Metis的邻接表格式
    adj_list = []
    for node in all_nodes:
        neighbors = [node_to_idx[n] for n in G.neighbors(node)]
        adj_list.append(sorted(neighbors))  # Metis要求排序
    
    # 调用Metis划分 - 移除options参数
    try:
        # 尝试不带options参数
        edgecuts, parts = metis.part_graph(adj_list, nparts=num_blocks)
    except TypeError:
        # 如果失败，尝试带options但用正确格式
        edgecuts, parts = metis.part_graph(adj_list, nparts=num_blocks, options=[5,0,0])
    
    # 将划分结果转换回原始节点ID
    blocks_dict = defaultdict(list)
    for i, part in enumerate(parts):
        node = idx_to_node[i]
        blocks_dict[part].append(node)
    
    blocks = list(blocks_dict.values())
    
    # 计算切边数量
    cut_edges = 0
    for u, v in edges:
        if parts[node_to_idx[u]] != parts[node_to_idx[v]]:
            cut_edges += 1
    
    return blocks, cut_edges

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities


def split_data_random(all_nodes, block_size):
    """随机划分（作为基线对比）"""
    import random
    
    nodes_shuffled = list(all_nodes)
    random.shuffle(nodes_shuffled)
    
    blocks = []
    for i in range(0, len(nodes_shuffled), block_size):
        block = nodes_shuffled[i:i+block_size]
        if block:
            blocks.append(block)
    
    return blocks



def split_data_by_node_strict(all_nodes, node_degree_dict, block_size):
    """严格按节点ID顺序分块，每个块固定大小
    
    Args:
        all_nodes: 所有节点列表
        node_degree_dict: 节点度字典 {node: degree}
        block_size: 每个块的固定大小
    
    Returns:
        blocks: 分块后的节点列表
        new_all_nodes: 去重后的所有节点
    """
    import time
    start_time = time.time()
    
    # 1. 去重并按ID排序
    unique_nodes = sorted(set(all_nodes))
    total_nodes = len(unique_nodes)
    
    # 2. 按固定大小分块
    blocks = []
    for i in range(0, total_nodes, block_size):
        end_idx = min(i + block_size, total_nodes)
        block_nodes = unique_nodes[i:end_idx]
        blocks.append(block_nodes)
    
    # 3. 输出统计信息
    elapsed_time = time.time() - start_time
    print(f"数据分块完成（严格顺序）：共{len(blocks)}个块，总节点{total_nodes}，耗时{elapsed_time:.4f}秒")
    
    # 打印每个块的统计信息
    for i, block in enumerate(blocks):
        block_nodes = len(block)
        if block_nodes > 0:
            min_id = min(block)
            max_id = max(block)
            avg_degree = sum(node_degree_dict.get(node, 0) for node in block) / max(block_nodes, 1)
            print(f"  块{i}: {block_nodes}个节点，ID范围[{min_id}, {max_id}]，平均度{avg_degree:.2f}")
    
    return blocks, unique_nodes

def split_data_by_degree(all_nodes, node_degree_dict, block_size, descending=True):
    """根据节点度进行顺序分块
    
    Args:
        all_nodes: 所有节点列表
        node_degree_dict: 节点度字典 {node: degree}
        block_size: 每个块的目标大小
        descending: True=降序（高度优先），False=升序（低度优先）
    
    Returns:
        blocks: 分块后的节点列表
        new_all_nodes: 去重后的所有节点
    """
    import time
    import math
    start_time = time.time()
    
    # 1. 去重
    unique_nodes = list(set(all_nodes))
    
    # 2. 按度排序
    sorted_nodes = sorted(unique_nodes, 
                         key=lambda x: node_degree_dict.get(x, 0), 
                         reverse=descending)
    
    total_nodes = len(sorted_nodes)
    
    # 3. 按block_size分块
    blocks = [sorted_nodes[i:i+block_size] 
              for i in range(0, total_nodes, block_size)]
    
    # 4. 输出统计信息
    elapsed_time = time.time() - start_time
    order = "降序（高度优先）" if descending else "升序（低度优先）"
    print(f"数据分块完成（按度{order}）：共{len(blocks)}个块，块大小目标{block_size}，总节点{total_nodes}，耗时{elapsed_time:.4f}秒")
    
    # 打印每个块的统计信息
    for i, block in enumerate(blocks):
        block_nodes = len(block)
        if block_nodes > 0:
            avg_degree = sum(node_degree_dict.get(node, 0) for node in block) / max(block_nodes, 1)
            min_degree = min(node_degree_dict.get(node, 0) for node in block)
            max_degree = max(node_degree_dict.get(node, 0) for node in block)
            print(f"  块{i}: {block_nodes}个节点，度范围[{min_degree}, {max_degree}]，平均度{avg_degree:.2f}")
    
    return blocks, unique_nodes



def calc_block_edge_weight_no_queue(edge_df, block_nodes, block_id, cn_base_alpha, 
                                   node_embed_dict=None, embed_weight_alpha=0.3):
    """Calculate edge weights for a block with optional embedding-based adjustment"""
    try:
        start_time = time.time()
        block_node_set = set(block_nodes)
        
        block_mask = edge_df['u'].isin(block_node_set) & edge_df['v'].isin(block_node_set)
        block_edge = edge_df[block_mask].copy()
        edge_count = len(block_edge)
        
        if edge_count == 0:
            #print(f"Warning: Block {block_id} has no valid edges")
            return (block_id, pd.DataFrame(columns=['u', 'v', 'weight']))
        
        block_edge[['u_sorted', 'v_sorted']] = np.sort(block_edge[['u', 'v']].values, axis=1)
        edge_counts = block_edge.groupby(['u_sorted', 'v_sorted']).size().reset_index(name='count')
        
        neighbor_dict = defaultdict(list)
        for _, row in edge_counts.iterrows():
            u, v = row['u_sorted'], row['v_sorted']
            neighbor_dict[u].append(v)
            neighbor_dict[v].append(u)
        for u in neighbor_dict:
            neighbor_dict[u].sort()
        
        def count_common(u, v):
            neighbors_u = neighbor_dict.get(u, [])
            neighbors_v = neighbor_dict.get(v, [])
            i = j = common = 0
            len_u, len_v = len(neighbors_u), len(neighbors_v)
            while i < len_u and j < len_v:
                if neighbors_u[i] == neighbors_v[j]:
                    common += 1
                    i += 1
                    j += 1
                elif neighbors_u[i] < neighbors_v[j]:
                    i += 1
                else:
                    j += 1
            return common
        
        edge_counts['common'] = edge_counts.apply(
            lambda row: count_common(row['u_sorted'], row['v_sorted']), axis=1
        )
        
        # 基础权重：边计数 + 共同邻居调整
        edge_counts['weight'] = edge_counts['count'] + cn_base_alpha * edge_counts['common']
        
      
        
        result_edge = block_edge[['u', 'v']].drop_duplicates()
        result_edge = result_edge.merge(
            edge_counts[['u_sorted', 'v_sorted', 'weight']],
            left_on=['u', 'v'],
            right_on=['u_sorted', 'v_sorted'],
            how='left'
        ).fillna(1)[['u', 'v', 'weight']]
        
        # 确保权重为正
        result_edge['weight'] = result_edge['weight'].clip(lower=0.001)
        
        #print(f"Process {os.getpid()}: Block {block_id} completed, {len(result_edge)} edges, "
         #     f"avg_weight={result_edge['weight'].mean():.3f}, time: {time.time()-start_time:.2f}s")
        return (block_id, result_edge)
    
    except Exception as e:
        #print(f"Process {os.getpid()}: Block {block_id} edge weight calculation failed! Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return (block_id, pd.DataFrame(columns=['u', 'v', 'weight']))
    



def compute_embeddings(edges, nodes):
    # 这里使用我们之前设计的嵌入函数，返回一个字典，节点->嵌入向量
    # 假设我们已经实现了 minimal_community_aware_embedding 函数
    _, embeddings = minimal_community_aware_embedding(edges, nodes, block_id=None)
    return embeddings

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))




import numpy as np
import pandas as pd
import time

def simplest_structural_embedding(edges, nodes, block_id):
    """Simple structural embedding based on graph structure"""
    G = nx.Graph()
    for idx, row in edges.iterrows():
        u = row['u']
        v = row['v']
        G.add_edge(u, v)
    
    # 确保所有节点都在图中（可能有孤立节点）
    for node in nodes:
        if node not in G:
            G.add_node(node)
    
    # 计算特征
    embeddings = {}
    for node in nodes:
        # 度
        deg = G.degree(node)
        # 局部聚类系数
        clustering = nx.clustering(G, node)
        # 邻居平均度
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 0:
            avg_neighbor_deg = sum(G.degree(n) for n in neighbors) / len(neighbors)
        else:
            avg_neighbor_deg = 0
        
        # 构建3维向量
        vec = np.array([deg, clustering, avg_neighbor_deg], dtype=np.float32)
        # 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        embeddings[node] = vec
    
    return (block_id, embeddings)
    

def minimal_community_aware_embedding(edges, nodes, block_id):
    """
    优化的社区感知嵌入，确保有变化
    """
    #print(f"计算嵌入，节点数: {len(nodes)}, 边数: {len(edges)}")
    
    # 方法1: 基于度的特征 + 社区边界特征
    degree = {}
    neighbor_counts = {}
    
    # 计算度和邻居
    for _, row in edges.iterrows():
        u, v = row['u'], row['v']
        
        # 更新度
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
        
        # 更新邻居集合
        if u not in neighbor_counts:
            neighbor_counts[u] = set()
        if v not in neighbor_counts:
            neighbor_counts[v] = set()
        
        neighbor_counts[u].add(v)
        neighbor_counts[v].add(u)
    
    n_nodes = len(nodes)
    embeddings = {}
    
    # 计算一些全局统计信息
    if degree:
        max_degree = max(degree.values())
        avg_degree = sum(degree.values()) / n_nodes if n_nodes > 0 else 0
    else:
        max_degree = 1
        avg_degree = 0
    
    # 为每个节点生成嵌入
    for i, node in enumerate(nodes):
        deg = degree.get(node, 0)
        
        # 计算邻居特征
        if node in neighbor_counts:
            neighbors = neighbor_counts[node]
            
            # 邻居的度统计
            neighbor_degrees = [degree.get(n, 0) for n in neighbors]
            avg_neighbor_deg = np.mean(neighbor_degrees) if neighbor_degrees else 0
            
            # 社区边界指标
            # 边界节点: 连接不同社区，其邻居之间的连接较少
            neighbor_connections = 0
            for n1 in neighbors:
                for n2 in neighbors:
                    if n1 != n2 and n2 in neighbor_counts.get(n1, set()):
                        neighbor_connections += 1
            
            possible_connections = len(neighbors) * (len(neighbors) - 1) if len(neighbors) > 1 else 1
            clustering = neighbor_connections / possible_connections if possible_connections > 0 else 0
            
            # 边界强度: 节点作为社区边界的可能性
            boundary_strength = deg * (1 - clustering) if clustering < 1 else 0
        else:
            avg_neighbor_deg = 0
            clustering = 0
            boundary_strength = 0
        
        # 创建8维嵌入向量
        vec = np.zeros(8, dtype=np.float32)
        
        # 特征1-2: 度特征
        vec[0] = deg  # 原始度
        vec[1] = deg / max_degree if max_degree > 0 else 0  # 归一化度
        
        # 特征3: 相对于平均度的位置
        vec[2] = 1 if deg > avg_degree else -1 if deg < avg_degree else 0
        
        # 特征4: 社区边界指标
        vec[3] = boundary_strength / max_degree if max_degree > 0 else 0
        
        # 特征5: 局部聚类系数
        vec[4] = clustering
        
        # 特征6: 邻居平均度比率
        vec[5] = avg_neighbor_deg / max_degree if max_degree > 0 else 0
        
        # 特征7-8: 基于节点ID的确定性随机特征
        node_hash = hash(str(node)) % 10000
        vec[6] = np.sin(node_hash / 1000.0)
        vec[7] = np.cos(node_hash / 777.0)
        
        # 确保有变化：添加时间相关的微小变化
        time_seed = int(time.time() * 1000) % 1000
        vec = vec * (1.0 + 0.001 * (time_seed % 10))
        
        # 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        embeddings[node] = vec
        
        # 每处理1000个节点打印一次进度
        if (i + 1) % 1000 == 0:
            print(f"已处理 {i+1}/{n_nodes} 个节点")
    
    #print(f"嵌入计算完成，生成了 {len(embeddings)} 个嵌入向量")
    return (block_id, embeddings)



def generate_block_community(args):
    """Generate communities for a block using Leiden algorithm"""
    try:
        weighted_edge_block,  block_nodes, block_id = args
        start_time = time.time()
        
        block_G = nx.Graph()
        if not weighted_edge_block.empty:
            valid_edges = weighted_edge_block[['u', 'v', 'weight']].values
            if len(valid_edges) > 0:
                block_G.add_weighted_edges_from(valid_edges)
        for node in block_nodes:
            if node not in block_G:
                block_G.add_node(node)
        
        if not block_G.nodes():
            return (block_id, {})
        
        num_nodes = len(block_G.nodes())
        num_edges = len(block_G.edges())
        # max_possible_edges = num_nodes * (num_nodes - 1) / 2 if num_nodes > 1 else 1
        # density = num_edges / max_possible_edges
        
        def run_leiden():
            # try:
            node_list = sorted(block_G.nodes())
            # node_list = list(block_G.nodes())
            node_to_idx = {node: idx for idx, node in enumerate(node_list)}
            # idx_to_node = {idx: node for idx, node in enumerate(node_list)}
            
            block_ig = ig.Graph(directed=False)
            block_ig.add_vertices(len(node_list))
            
            edges, edge_weights = [], []
            for u, v, data in block_G.edges(data=True):
                edges.append((node_to_idx[u], node_to_idx[v]))
                edge_weights.append(float(data.get('weight', 1.0)))
            if edges:
                block_ig.add_edges(edges)
                block_ig.es['weight'] = edge_weights
            

            
            partition = la.find_partition(
                block_ig,
                la.ModularityVertexPartition,
                weights='weight',
                n_iterations=20,
                seed=42
            )
            
            leiden_comm = np.array(partition.membership)
            return [leiden_comm[node_to_idx[node]] if node in node_to_idx else 0 
                    for node in block_nodes]
            
            # except Exception as e:
            #     print(f"Block {block_id} Leiden failed")
            #     # return run_kmeans()
        

        block_fine_comm = run_leiden()
        
        global_comm_prefix = block_id * 1000000
        block_comm_dict = {node: global_comm_prefix + comm 
                         for node, comm in zip(block_nodes, block_fine_comm)}
        
        #print(f"Process {os.getpid()}: Block {block_id} generated {len(set(block_fine_comm))} communities, time: {time.time()-start_time:.2f}s")
        return (block_id, block_comm_dict)
    
    except Exception as e:
        #print(f"Process {os.getpid()}: Block {block_id} community generation failed! Error: {str(e)}")
        block_comm_dict = {node: block_id * 1000000 + i for i, node in enumerate(block_nodes)}
        return (block_id, block_comm_dict)

def auto_kmeans_elbow(block_z, max_k=15):
    """Determine optimal k using elbow method"""
    inertias = []
    k_range = range(1, min(max_k, len(block_z)//2) + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=3)
        kmeans.fit(block_z)
        inertias.append(kmeans.inertia_)
    
    improvements = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
    if improvements:
        elbow_idx = np.argmax(improvements) + 1
        optimal_k = k_range[elbow_idx]
    else:
        optimal_k = 2
    
    return optimal_k


def build_global_graph_fast(weighted_edge_df, all_nodes):
    """Build global graph from weighted edges"""
    #print("  Building global graph...")
    start_time = time.time()
    
    global_G = nx.Graph()
    global_G.add_nodes_from(all_nodes)
    all_nodes_set = set(all_nodes)
    
    for row in weighted_edge_df.itertuples():
        u, v, weight = row.u, row.v, row.weight
        if u in all_nodes_set and v in all_nodes_set:
            global_G.add_edge(u, v, weight=float(weight))
    
    #print(f"  Global graph: {global_G.number_of_nodes()} nodes, {global_G.number_of_edges()} edges")
    #print(f"  Build time: {time.time()-start_time:.2f}s")
    return global_G

def process_block(args, edge_df, cn_base_alpha):
    """Process block for edge weight calculation"""
    block_id, block_nodes = args
    return calc_block_edge_weight_no_queue(edge_df, block_nodes, block_id, cn_base_alpha)

def load_data(edge_file_path, comm_file_path):
    """Load SNAP Community datasets"""
    with open(comm_file_path) as f:
        communties = [[int(i) for i in x.split()] for x in f]
    
    with open(edge_file_path) as f:
        edges = [[int(i) for i in e.split()] for e in f]
    
    edges = [[u, v] if u < v else [v, u] for u, v in edges if u != v]
    
    raw_nodes = {node for e in edges for node in e}
    mapping = {u: i for i, u in enumerate(sorted(raw_nodes))}
    
    edges = [[mapping[u], mapping[v]] for u, v in edges]
    communties = [[mapping[node] for node in com] for com in communties]
    
    num_node, num_edges, num_comm = len(raw_nodes), len(edges), len(communties)
    #print(f"[{os.path.basename(edge_file_path).upper()}] #Nodes {num_node}, #Edges {num_edges}, #Communities {num_comm}")
    
    new_nodes = list(range(len(raw_nodes)))
    
    return num_node, num_edges, num_comm, new_nodes, edges, communties

def convert_comm_dict_to_comms_list(comm_dict):
    """Convert node->community dict to list of community lists"""
    comm_to_nodes = defaultdict(list)
    for node, comm_id in comm_dict.items():
        comm_to_nodes[comm_id].append(node)
    
    return [nodes for nodes in comm_to_nodes.values()]

def evaluate_with_correct_format(true_comms, comm_dict):
    """Evaluate with correct format"""
    #print("Converting data structure...")
    
    pred_comms = convert_comm_dict_to_comms_list(comm_dict)
    
    #print(f"True communities: {len(true_comms)}, Predicted communities: {len(pred_comms)}")
    
    true_sizes = [len(comm) for comm in true_comms]
    pred_sizes = [len(comm) for comm in pred_comms]
    
    #print(f"True community sizes: min={min(true_sizes)}, max={max(true_sizes)}, avg={np.mean(true_sizes):.1f}")
    #print(f"Predicted community sizes: min={min(pred_sizes)}, max={max(pred_sizes)}, avg={np.mean(pred_sizes):.1f}")
    
    try:
        from metrics import eval_scores_fast_optimized_fixed
        avg_precision, avg_recall, avg_f1, avg_jaccard = eval_scores_fast_optimized_fixed(pred_comms, true_comms, tmp_print=True)
        
        print(f"  Average Precision: {avg_precision:.4f}")
        print(f"  Average Recall: {avg_recall:.4f}")
        print(f"  Average F1 Score: {avg_f1:.4f}")
        print(f"  Average Jaccard: {avg_jaccard:.4f}")
        
        return avg_precision, avg_recall, avg_f1, avg_jaccard
    except ImportError:
        #print("Warning: metrics module not found, using fallback evaluation")
        return 0.0, 0.0, 0.0, 0.0



def build_global_graph_from_original_optimized(edge_df, nodes):
    """修复后的优化版本"""
    #print("  Building global graph from original edges (optimized)...")
    start_time = time.time()
    
    # 1. 快速处理边（使用numpy）
    #print("    Processing edges...")
    
    # 转换为numpy数组
    if isinstance(edge_df, pd.DataFrame):
        edges_array = edge_df[['u', 'v']].values
    else:
        edges_array = edge_df
    
    # 排序边（无向图）
    edges_sorted = np.sort(edges_array, axis=1)
    
    # 使用字典统计边权重
    #print("    Counting edge weights...")
    edge_counts = {}
    batch_size = 1000000
    
    for i in range(0, len(edges_sorted), batch_size):
        batch = edges_sorted[i:i+batch_size]
        for u, v in batch:
            key = (u, v)
            if key in edge_counts:
                edge_counts[key] += 1.0
            else:
                edge_counts[key] = 1.0
    
    #print(f"    Unique edges: {len(edge_counts):,}")
    
    # 2. 构建边列表
    #print("    Building edge list...")
    edge_list = [(u, v, w) for (u, v), w in edge_counts.items()]
    
    # 3. 构建图
    #print("    Creating graph...")
    G = nx.Graph()
    
    # 添加节点
    if nodes is not None:
        G.add_nodes_from(nodes)
        #print(f"    Added {len(nodes):,} nodes from input")
    else:
        # 从边中提取所有节点
        node_set = set()
        for (u, v), _ in edge_counts.items():
            node_set.add(u)
            node_set.add(v)
        G.add_nodes_from(node_set)
        #print(f"    Added {len(node_set):,} nodes from edges")
    
    # 添加边（关键修复：使用add_weighted_edges_from）
    G.add_weighted_edges_from(edge_list)
    
    # 验证图构建
    #print(f"    Graph validation:")
    #print(f"      - Nodes: {G.number_of_nodes():,}")
    #print(f"      - Edges: {G.number_of_edges():,}")
    #print(f"      - Is connected: {nx.is_connected(G) if G.number_of_nodes() > 0 else 'N/A'}")
    
    # 检查一些随机节点的边
    if G.number_of_nodes() > 0:
        sample_nodes = list(G.nodes())[:min(5, G.number_of_nodes())]
        for node in sample_nodes:
            degree = G.degree(node)
            #print(f"      - Node {node}: degree = {degree}")
    
    #print(f"    Time elapsed: {time.time() - start_time:.2f}s")
    
    return G

def merge_small_communities_fast(partition_dict, graph, min_size=3):
    """Fast merging of small communities"""
    from collections import defaultdict
    
    communities = partition_dict.copy()
    
    small_comms = {}
    large_comms = {}
    node_to_comm = {}
    
    for cid, nodes in communities.items():
        node_list = list(nodes)
        if len(node_list) < min_size:
            small_comms[cid] = node_list
        else:
            large_comms[cid] = node_list
        for node in node_list:
            node_to_comm[node] = cid
    
    if not small_comms:
        return communities
    
    #print(f"Small communities to merge: {len(small_comms)}")
    
    comm_connections = defaultdict(lambda: defaultdict(int))
    for u, v in graph.edges():
        comm_u = node_to_comm[u]
        comm_v = node_to_comm[v]
        
        if comm_u != comm_v:
            comm_connections[comm_u][comm_v] += 1
            comm_connections[comm_v][comm_u] += 1
    
    merged_result = {cid: set(nodes) for cid, nodes in large_comms.items()}
    comm_size = {cid: len(nodes) for cid, nodes in merged_result.items()}
    
    for small_cid, small_nodes in small_comms.items():
        connections = comm_connections.get(small_cid, {})
        
        candidate_large = {}
        for neighbor_comm, weight in connections.items():
            if neighbor_comm in merged_result:
                candidate_large[neighbor_comm] = weight
        
        if candidate_large:
            best_comm = max(
                candidate_large.items(),
                key=lambda x: (x[1], -comm_size.get(x[0], 0))
            )[0]
        else:
            if comm_size:
                best_comm = min(comm_size.items(), key=lambda x: x[1])[0]
            else:
                best_comm = max(merged_result.keys(), default=-1) + 1
                merged_result[best_comm] = set()
                comm_size[best_comm] = 0
        
        if best_comm not in merged_result:
            merged_result[best_comm] = set()
            comm_size[best_comm] = 0
        
        merged_result[best_comm].update(small_nodes)
        comm_size[best_comm] += len(small_nodes)
    
    final_result = {}
    for new_id, (old_id, nodes) in enumerate(merged_result.items()):
        if nodes:
            final_result[new_id] = list(nodes)
    
    sizes = [len(nodes) for nodes in final_result.values()]
    remaining_small = sum(1 for size in sizes if size < min_size)
    
    #print(f"Merged communities: {len(final_result)}")
    #print(f"Remaining small communities: {remaining_small}")
    if sizes:
        print(f"Community sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.2f}")
    
    original_nodes = sum(len(nodes) for nodes in communities.values())
    result_nodes = sum(len(nodes) for nodes in final_result.values())
    assert original_nodes == result_nodes, f"Node count mismatch: {original_nodes} != {result_nodes}"
    
    return final_result

def optimize_community_structure(node_to_community_dict, global_G, min_size=3):
    """Optimize community structure"""
    print("Optimizing community structure...")
    
    community_to_nodes = defaultdict(list)
    for node, comm_id in node_to_community_dict.items():
        community_to_nodes[comm_id].append(node)
    
    #print(f"Converted communities: {len(community_to_nodes)}")
    
    comm_sizes = [len(nodes) for nodes in community_to_nodes.values()]
    #print(f"Community size stats: min={min(comm_sizes)}, max={max(comm_sizes)}, avg={np.mean(comm_sizes):.2f}")
    
    merged_community_to_nodes = merge_small_communities_fast(community_to_nodes, global_G, min_size)
    
    #print(f"Merged communities: {len(merged_community_to_nodes)}")
    
    final_node_to_community = {}
    for comm_id, nodes in merged_community_to_nodes.items():
        for node in nodes:
            final_node_to_community[node] = comm_id
    
    return final_node_to_community

def global_optimization_with_overlap(G, comm_dict, all_new_nodes):
    """Global optimization with overlap nodes"""
    from collections import defaultdict
    
    #print(f"Starting global optimization with {len(all_new_nodes)} nodes")
    
    improved_comm_dict = comm_dict.copy()
    comm_sizes = defaultdict(int)
    for node, comm_id in improved_comm_dict.items():
        comm_sizes[comm_id] += 1
    
    for new_node in all_new_nodes:
        if new_node not in G:
            continue
            
        current_comm = improved_comm_dict[new_node]
        neighbors = list(G.neighbors(new_node))
        
        if not neighbors:
            continue
            
        neighbor_comms = defaultdict(int)
        for neighbor in neighbors:
            neighbor_comm = improved_comm_dict.get(neighbor, -1)
            if neighbor_comm != -1:
                neighbor_comms[neighbor_comm] += 1
        
        if neighbor_comms:
            best_comm = max(
                neighbor_comms.items(),
                key=lambda x: (x[1], -comm_sizes.get(x[0], 0))
            )[0]
            
            current_conn = neighbor_comms.get(current_comm, 0)
            best_conn = neighbor_comms[best_comm]
            
            should_move = (
                best_comm != current_comm and 
                best_conn > current_conn
            )
            
            if should_move:
                comm_sizes[current_comm] = max(0, comm_sizes.get(current_comm, 0) - 1)
                comm_sizes[best_comm] = comm_sizes.get(best_comm, 0) + 1
                
                improved_comm_dict[new_node] = best_comm
    
    #print("Global optimization completed")
    return improved_comm_dict

# def split_data_by_connectivity(edge_df, all_nodes, node_degree_dict, max_block_size=5000):
#     """
#     递归BFS二分分割图
    
#     Parameters:
#     -----------
#     edge_df: DataFrame with columns 'u', 'v'
#     all_nodes: set/list of all nodes
#     node_degree_dict: dict of node -> degree
#     max_block_size: 每个块的最大节点数
    
#     Returns:
#     --------
#     blocks: list of lists, 每个子列表是一个块的节点
#     node_sequence: list, 全局节点顺序（BFS顺序）
#     """
#     # 构建邻接表
#     adj = defaultdict(list)
#     for u, v in edge_df[['u', 'v']].values:
#         adj[u].append(v)
#         adj[v].append(u)
    
#     blocks = []
#     node_sequence = []
    
#     def bfs_order(nodes_subset, start_node):
#         """获取子图的BFS顺序"""
#         visited = set()
#         queue = deque([start_node])
#         order = []
        
#         while queue:
#             node = queue.popleft()
#             if node in visited or node not in nodes_subset:
#                 continue
#             visited.add(node)
#             order.append(node)
#             for nb in adj.get(node, []):
#                 if nb in nodes_subset and nb not in visited:
#                     queue.append(nb)
#         return order
    
#     def split(nodes_subset):
#         """递归分割函数"""
#         nonlocal blocks, node_sequence
        
#         # 如果块已经够小，直接加入
#         if len(nodes_subset) <= max_block_size:
#             # 找高度节点作为BFS起点
#             start = max(nodes_subset, key=lambda x: node_degree_dict.get(x, 0))
#             # 记录BFS顺序
#             node_sequence.extend(bfs_order(nodes_subset, start))
#             blocks.append(list(nodes_subset))
#             return
        
#         # 找两个距离远的种子节点
#         nodes_list = list(nodes_subset)
        
#         # 选度数最高的节点作为第一个种子
#         seed1 = max(nodes_list, key=lambda x: node_degree_dict.get(x, 0))
        
#         # BFS找离seed1最远的节点作为第二个种子
#         visited = {seed1}
#         queue = deque([(seed1, 0)])
#         farthest_node = seed1
#         max_dist = 0
        
#         while queue:
#             node, dist = queue.popleft()
#             if dist > max_dist:
#                 max_dist = dist
#                 farthest_node = node
#             for nb in adj.get(node, []):
#                 if nb in nodes_subset and nb not in visited:
#                     visited.add(nb)
#                     queue.append((nb, dist + 1))
        
#         seed2 = farthest_node
        
#         # 从两个种子开始BFS扩张
#         part1 = {seed1}
#         part2 = {seed2}
#         q1 = deque([seed1])
#         q2 = deque([seed2])
#         visited = {seed1, seed2}
        
#         target = len(nodes_subset) // 2
        
#         # 交替扩张
#         while len(part1) < target and len(part2) < target and (q1 or q2):
#             # 扩张part1
#             if q1 and len(part1) < target:
#                 node = q1.popleft()
#                 for nb in adj.get(node, []):
#                     if nb in nodes_subset and nb not in visited:
#                         if len(part1) < target:
#                             part1.add(nb)
#                             visited.add(nb)
#                             q1.append(nb)
#                         else:
#                             break
            
#             # 扩张part2
#             if q2 and len(part2) < target:
#                 node = q2.popleft()
#                 for nb in adj.get(node, []):
#                     if nb in nodes_subset and nb not in visited:
#                         if len(part2) < target:
#                             part2.add(nb)
#                             visited.add(nb)
#                             q2.append(nb)
#                         else:
#                             break
        
#         # 分配剩余节点
#         remaining = [n for n in nodes_subset if n not in visited]
#         if remaining:
#             if len(part1) <= len(part2):
#                 part1.update(remaining)
#             else:
#                 part2.update(remaining)
        
#         # 递归分割两个部分
#         split(part1)
#         split(part2)
    
#     # 开始递归分割
#     split(set(all_nodes))
    
#     return blocks, node_sequence

# def get_adaptive_params(network_type,m,n):
#     """Get adaptive parameters based on network type and size"""

#     d_bar = 2 * m / n if n > 0 else 0
    
#     if n < 50000:
#         K = min(10, max(3, round(n / 2000)))
#     elif d_bar >= 8:
#         K = min(10, max(3, round(n / 16000)))
#     else:
#         K = min(2000, max(100, round(n / 3000)))
    
#     if network_type == 'social':
#         if n < 300000:
#             tau = 4
#         else:
#             tau = 5
#         alpha = 1.0
        
#     elif network_type == 'co-purchase':
#         if n < 300000:
#             tau = 4
#         else:
#             tau = 3
#         alpha = 5.0
        
#     elif network_type == 'collaboration':
#         if n < 300000:
#             tau = 5
#         else:
#             tau = 4
#         alpha = 3.0



        
    
#     return K, tau, alpha

import community 
def execute_HIDC_pipeline_unsupervised(edge_file_path, comm_file_path, network_type):
    """Main unsupervised HIDC pipeline"""
    """Luyun"""
    start_total_time1 = time.time()
    #print("="*50)
    #print(f"Main process started (unsupervised version): {time.strftime('%Y-%m-%d %H:%M:%S')}")
    #print("="*50)


    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR, exist_ok=True)
        #print(f"Created temporary directory: {TEMP_DIR}")
    #print("\n[1/6 Loading data]")
    num_node, num_edges, num_comm, all_nodes, edges, communties = load_data(edge_file_path, comm_file_path)
    edge_df = pd.DataFrame(edges, columns=['u', 'v'])
    
    #print("\n[2/6 Feature preprocessing]")
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    u_idx = np.array([node_to_idx.get(u, -1) for u in edge_df['u']], dtype=np.int32)
    v_idx = np.array([node_to_idx.get(v, -1) for v in edge_df['v']], dtype=np.int32)
    valid_u = u_idx[u_idx != -1]
    valid_v = v_idx[v_idx != -1]
    u_degrees = np.bincount(valid_u, minlength=len(all_nodes))
    v_degrees = np.bincount(valid_v, minlength=len(all_nodes))
    node_degree = (u_degrees + v_degrees).astype(np.float32)
    node_degree_dict = dict(zip(all_nodes, node_degree))
    
    #print(f"  Node degree stats: mean={node_degree.mean():.2f}, non-zero nodes={np.sum(node_degree > 0)}")
    
    #print("\n[3/6 Calculating edge weights]")
    # K, MIN_COMM_SIZE, cn_base_alpha = get_adaptive_params(network_type, len(edges),len(all_nodes))
    # if len(all_nodes)<5000:
    # #     MIN_COMM_SIZE=2
    #     cn_base_alpha=0

    total_time1 = (time.time() - start_total_time1) / 60
    print(f"\nTotal time: {total_time1:.2f} minutes")

    block_size,MIN_COMM_SIZE,cn_base_alpha=get_adaptive_params(len(all_nodes),len(edge_df))
    print(f"  block_size: ={block_size}, MIN_COMM_SIZE:{MIN_COMM_SIZE},cn_base_alpha:{cn_base_alpha}")

    # klist=[0,0.25,0.5,0.75,1,1.25,1.5,1.75,2]
    klist=[block_size]
    # klist=[2,3,4,5,6]
    # cn_base_alpha=0
    for block_size in klist:  
        # print(f"  block_size: ={block_size}, MIN_COMM_SIZE:{MIN_COMM_SIZE},cn_base_alpha:{cn_base_alpha}")
        # blocks,new_all_nodes= split_data_by_node(all_nodes, node_degree_dict, K)
        start_total_time = time.time()
        

        
        print(f"  block_size: ={block_size}, MIN_COMM_SIZE:{MIN_COMM_SIZE},cn_base_alpha:{cn_base_alpha}")
        print(f"块的大小：{block_size}")
        start_cpp = time.time()
        blocks,new_all_nodes = split_data_by_connectivity(edge_df, all_nodes, node_degree_dict, block_size)
        # blocks,cutedge = split_data_by_metis(edge_df, all_nodes, block_size)
        # print(f"cutedge：{cutedge}")
        # blocks = split_data_random(all_nodes, block_size)
        # blocks,uniquenodes = split_data_by_node_strict(all_nodes, node_degree_dict,block_size)

        # blocks,uniquenodes = split_data_by_degree(all_nodes, node_degree_dict,block_size)

        end_cpp = (time.time() - start_cpp) / 60
        print(f"\ncpp time: {end_cpp:.2f} minutes")

        
    
        num_blocks = len(blocks)
        
        if num_blocks == 0:
        
            raise ValueError("Data splitting failed")
        
        with Pool(processes=NUM_PROCESSES) as pool:
            block_args = [(block_id, block_nodes) for block_id, block_nodes in enumerate(blocks)]
            partial_func = partial(process_block, edge_df=edge_df, cn_base_alpha=cn_base_alpha)
            results = pool.imap_unordered(partial_func, block_args)
            
            weighted_edge_dict = {}
            for bid, bedge in results:
                weighted_edge_dict[bid] = bedge
                #print(f"Received edge weights for block {bid}, total {len(weighted_edge_dict)}/{num_blocks}")
        
        weighted_edge_list = []
        for i in sorted(weighted_edge_dict.keys()):
            df = weighted_edge_dict[i]
            if not df.empty and len(df) > 0:
                weighted_edge_list.append(df)
        
        if weighted_edge_list:
            weighted_edge_df = pd.concat(weighted_edge_list, ignore_index=True)
            weighted_edge_df = weighted_edge_df.astype({
                'u': int,
                'v': int,
                'weight': float
            })
        else:
            weighted_edge_df = pd.DataFrame(columns=['u', 'v', 'weight'])
            #print("Warning: No valid edge data generated")
        
        #print(f"Summarized edge weights: {len(weighted_edge_df)} edges")
        
        #print("\n[4/6 Training unsupervised node embeddings]")
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR, exist_ok=True)
        # ##print(f"✓ Created temporary directory: {TEMP_DIR}")
        for f in os.listdir(TEMP_DIR):
            os.remove(os.path.join(TEMP_DIR, f))
        
    
        #print("\n[5/6 Generating local communities]")
        comm_args = []
        for block_id, block_nodes in enumerate(blocks):
            block_edges = weighted_edge_df[weighted_edge_df['u'].isin(block_nodes) | weighted_edge_df['v'].isin(block_nodes)]
            comm_args.append((block_edges, block_nodes, block_id))
        
        global_comm_dict = {}
        with Pool(processes=NUM_PROCESSES) as pool:
            comm_results = pool.imap_unordered(generate_block_community, comm_args)
            
            received_blocks = 0
            for result in comm_results:
                bid, bcomm = result
                global_comm_dict.update(bcomm)
                received_blocks += 1
                #print(f"Received community results for block {bid}, total {received_blocks}/{len(blocks)} blocks")
        
        #print("\n[6/6 Global community optimization]")
        global_G = build_global_graph_from_original_optimized(edge_df, all_nodes)

        # Q = community.modularity(global_comm_dict, global_G)

        # print(f'模块度:{Q}')

        start_new = time.time()
        # final_comm_dict1 = global_optimization_with_overlap(
        #     global_G, 
        #     global_comm_dict, 
        #     new_all_nodes,
        # )
        
        

        final_comm_dict1 = global_optimization_with_overlap(
            global_G, 
            global_comm_dict, 
            all_nodes,
        )

        # 创建节点列表的随机副本
        # all_nodes_random = list(all_nodes)
        # random.shuffle(all_nodes_random)

        # # 使用随机顺序调用优化函数
        # final_comm_dict1 = global_optimization_with_overlap(
        #     global_G, 
        #     global_comm_dict, 
        #     all_nodes_random,  # 使用随机顺序的节点列表
        # )

        end_new = (time.time() - start_new) / 60
        print(f"\nglobal time: {end_new:.2f} minutes")

        final_comm_dict = optimize_community_structure(final_comm_dict1, global_G, MIN_COMM_SIZE)

        # Q = community.modularity(final_comm_dict, global_G)

        # print(f'模块度:{Q}')

        
        #print("\n[Performance evaluation]")
        if communties:
            metrics = evaluate_with_correct_format(communties, final_comm_dict)
        
        # shutil.rmtree(TEMP_DIR, ignore_errors=True)
        
        total_time = (time.time() - start_total_time) / 60
        print(f"\nTotal time: {total_time+total_time1:.2f} minutes")
    return global_comm_dict

# Dataset configurations
DATASET_CONFIGS = {
    'facebook': {
        'edge_path': 'dataset/facebook-1.90.ungraph.txt',
        'community_path': 'dataset/facebook-1.90.cmty.txt',
        'description': 'Facebook social network',
        'network_type': 'social'
    },
    'amazon1': {
        'edge_path': 'dataset/amazon-1.90.ungraph.txt',
        'community_path': 'dataset/amazon-1.90.cmty.txt',
        'description': 'Amazon co-purchasing network',
        'network_type': 'co-purchase'
    },
    'lj1': {
        'edge_path': 'dataset/lj-1.90.ungraph.txt',
        'community_path': 'dataset/lj-1.90.cmty.txt',
        'description': 'LiveJournal social network',
        'network_type': 'social'
    },
    'dblp1': {
        'edge_path': 'dataset/dblp-1.90.ungraph.txt',
        'community_path': 'dataset/dblp-1.90.cmty.txt',
        'description': 'DBLP collaboration network',
        'network_type': 'collaboration'
    },
    'dblp2': {
        'edge_path': 'dataset/dblp.ungraph.txt',
        'community_path': 'dataset/dblp_communities.txt',
        'description': 'DBLP collaboration network',
        'network_type': 'collaboration'
    },
    'amazon2': {
        'edge_path': 'dataset/com-amazon.ungraph.txt',
        'community_path': 'dataset/com-amazon.all.dedup.cmty.txt',
        'description': 'Amazon co-purchasing network',
        'network_type': 'co-purchase'
    },
    'lj2': {
        'edge_path': 'dataset/lj.ungraph.txt',
        'community_path': 'dataset/lj.cmty.txt',
        'description': 'LiveJournal social network',
        'network_type': 'social'
    },
}

if __name__ == "__main__":
    dataset_name = "amazon2"
    configds = DATASET_CONFIGS.get(dataset_name)
    
    EDGE_FILE_PATH = configds["edge_path"]
    COMMUNITY_FILE_PATH = configds["community_path"]
    network_type = configds["network_type"]
    #print(f"CPU cores: {mp.cpu_count()}")
    
    if not os.path.exists(EDGE_FILE_PATH):
        raise FileNotFoundError(f"Edge file not found: {EDGE_FILE_PATH}")
    
    if not os.path.exists(COMMUNITY_FILE_PATH):
        print(f"Warning: Community file not found, running unsupervised community detection")
    
    execute_HIDC_pipeline_unsupervised(EDGE_FILE_PATH, COMMUNITY_FILE_PATH, network_type)