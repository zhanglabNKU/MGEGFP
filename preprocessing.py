
import numpy as np
import scipy.sparse as sp
import torch





def load_gene_list(path,gene_list_id):
    gene_list_path = path + gene_list_id
    gene_list = []
    for i in open(gene_list_path):
        temp = i.strip('\n')
        gene_list.append(str(temp))
    return gene_list


def load_adj(A_all, path, adj_id):
    adj_path = path + adj_id
    for i in open(adj_path):
        m = i.split('\t')
        g1 = int(m[0])-1
        g2 = int(m[1])-1
        edge_weight = float(m[2].strip('\n'))
        A_all[g1][g2] = edge_weight
    return A_all


def extract_sub_gene_feature(gene_list_sub, gene_list_all, feature_matrix_sub, feature_matrix_all):
    cnt = 0
    for i in gene_list_sub:
        index = gene_list_all.index(i)
        feature_matrix_sub[cnt,:] = feature_matrix_all[index,:]
        cnt += 1
    return feature_matrix_sub

def extract_sub_index(gene_list_sub, gene_list_all):
    index_list = []
    for i in gene_list_sub:
        index = gene_list_all.index(i)
        index_list.append(index)
    return index_list



def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()

    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def get_nodepairs(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()

    adj_triu = sp.triu(sparse_mx)
    coords = np.vstack((adj_triu.row, adj_triu.col)).transpose()
    return coords

def get_labels(adj0,adj1,adj2,adj3,adj4,adj5,coords):

    print(coords.shape)
    print(adj1.shape)
    labels = []
    adj = np.stack([adj0,adj1,adj2,adj3,adj4,adj5])

    # print(adj.shape[1])
    for i in range(coords.shape[0]):

        labels.append(adj[:, coords[i,0], coords[i,1]])

    return labels




def preprocess_graph(adj):

    adj = sp.coo_matrix(adj)

    adj_ = adj + sp.eye(adj.shape[0])

    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()

    return sparse_to_tuple(adj_normalized)



def matrix_normalization(feature_matrix):
    # if feature_matrix.min < 0:
    if np.min(feature_matrix) < 0:
        print('## Negative entries in the matrix are not allowed ##')
        feature_matrix[feature_matrix < 0] = 0
        print('## Feature matrix has been converted to nonnegative matrix ##')
    else:
        print('## Feature matrix is nonnegative ##')
    if (feature_matrix.T == feature_matrix).all():
        print('## Feature matrix is symmetric ##')
    else:
        print('## Feature matrix is not symmetric ##')
        feature_matrix = feature_matrix + feature_matrix.T-np.diag(np.diag(feature_matrix))
        print('## Feature matrix has been converted to symmetric ##')

    # normalizing the feature_matrix
    deg = feature_matrix.sum(axis=1).flatten()
    deg = np.divide(1., np.sqrt(deg))
    deg[np.isinf(deg)] = 0
    D = np.diag(deg)
    feature_matrix = D.dot(feature_matrix.dot(D))

    return feature_matrix




def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1), dtype=float)  # 按行求和 每行的和构成一个一维数组
    r_inv = np.power(rowsum, -1).flatten()  # 求倒数
    r_inv[np.isinf(r_inv)] = 0.  # 无穷的赋为0
    r_mat_inv = sp.diags(r_inv)  # 化为对角阵
    features = r_mat_inv.dot(features)  # 用这个矩阵归一化特征矩阵
    if isinstance(features, np.ndarray):   # 如果输入是np数组 就是返回features
        return features
    else:   # 如果是别的类型的输入 返回：1features的dense形式 2元组
        return features.todense(), sparse_to_tuple(features)



def mask_test_edges(adj):

    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)

    adj_tuple = sparse_to_tuple(adj_triu)

    edges = adj_tuple[0]
    train_edges = adj_tuple[0]

    # edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 10.))

    all_edge_idx = list(range(edges.shape[0]))

    np.random.shuffle(all_edge_idx)

    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]

    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]

    train_edges = np.delete(edges, np.hstack([val_edge_idx]), axis=0)
    train_edges = np.delete(edges, np.hstack([test_edge_idx]), axis=0)



    data = np.ones(train_edges.shape[0])

    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    data = np.ones(val_edges.shape[0])
    adj_val = sp.csr_matrix((data, (val_edges[:, 0], val_edges[:, 1])), shape=adj.shape)
    adj_val = adj_val + adj_val.T

    data = np.ones(test_edges.shape[0])
    adj_test = sp.csr_matrix((data, (test_edges[:, 0], test_edges[:, 1])), shape=adj.shape)
    adj_test = adj_test + adj_test.T


    return adj_train, adj_val, adj_test
