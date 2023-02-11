import numpy as np

def gcn_layer(A, X, W):
    """
    Perform a single GCN layer propagation.

    A: Adjacency matrix of the graph (N x N)
    X: Feature matrix of the nodes (N x F)
    W: Weight matrix of the GCN layer (F x F')

    Returns:
    H: Output feature matrix (N x F')
    """
    H = np.dot(A, X)
    H = np.dot(H, W)
    return H

def gcn(A, X, W1, W2):
    """
    Perform a GCN propagation over multiple layers.

    A: Adjacency matrix of the graph (N x N)
    X: Initial feature matrix of the nodes (N x F)
    W1: Weight matrix of the first GCN layer (F x F1)
    W2: Weight matrix of the second GCN layer (F1 x F2)

    Returns:
    H: Output feature matrix after two GCN layers (N x F2)
    """
    H1 = gcn_layer(A, X, W1)
    H2 = gcn_layer(A, H1, W2)
    return H2
