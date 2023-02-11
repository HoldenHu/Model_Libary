import numpy as np

def rgcn_layer(A, X, W, rel_matrix):
    """
    Perform a single RGCN layer propagation.

    A: Adjacency matrix of the graph (N x N)
    X: Feature matrix of the nodes (N x F)
    W: Weight matrix of the RGCN layer (K x F x F')
    rel_matrix: Relational matrix indicating the type of relationship between nodes (N x N x K)

    Returns:
    H: Output feature matrix (N x F')
    """
    N, K = rel_matrix.shape[:2]
    F = X.shape[1]
    H = np.zeros((N, F))
    for k in range(K):
        R = np.multiply(rel_matrix[:, :, k][:, :, np.newaxis], A)
        H += np.dot(R, X) @ W[k]
    return H

def rgcn(A, X, W1, W2, rel_matrix1, rel_matrix2):
    """
    Perform a RGCN propagation over multiple layers.

    A: Adjacency matrix of the graph (N x N)
    X: Initial feature matrix of the nodes (N x F)
    W1: Weight matrix of the first RGCN layer (K1 x F x F1)
    W2: Weight matrix of the second RGCN layer (K2 x F1 x F2)
    rel_matrix1: Relational matrix for the first RGCN layer (N x N x K1)
    rel_matrix2: Relational matrix for the second RGCN layer (N x N x K2)

    Returns:
    H: Output feature matrix after two RGCN layers (N x F2)
    """
    H1 = rgcn_layer(A, X, W1, rel_matrix1)
    H2 = rgcn_layer(A, H1, W2, rel_matrix2)
    return H2
