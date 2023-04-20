from functools import partial
from typing import Tuple, Optional

import numpy as np
import torch
from scipy.cluster.vq import kmeans as sp_kmeans
from scipy.spatial import cKDTree


def initialize(X: torch.Tensor, k: int, seed: Optional[int]) -> torch.Tensor:
    """initialize cluster centers
    Args:
        X (torch.Tensor): matrix
        k (int): number of clusters
        seed (Optional[int]): seed for kmeans
    Returns:
        torch.Tensor: initial state
    """
    num_samples = len(X)
    if seed == None:
        indices = np.random.choice(num_samples, k, replace=False)
    else:
        np.random.seed(seed)
        indices = np.random.choice(num_samples, k, replace=False)
    initial_state = X[indices]
    return initial_state


def k_means_fast(
    X: torch.Tensor,
    k: int,
    distance: str = "euclidean",
    tol: float = 1e-5,
    iter_limit=20,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor]:
    """k-means
    Args:
        X (torch.Tensor): matrix
        k (int): number of clusters
        distance (str, optional): distance [options: "euclidean", "cosine"]. Defaults to "euclidean".
        tol (float, optional): threshold. Defaults to 1e-4.
        iter_limit (int, optional): hard limit for max number of iterations. Defaults to 20.
        seed (Optional[int], optional): random seed. Defaults to None.
    Raises:
        NotImplementedError: invalid distance metric
    Returns:
        Tuple[torch.Tensor]: cluster centers & cluster ids
    """
    if distance == 'euclidean':
        pairwise_distance_function = partial(pairwise_distance)
    elif distance == 'cosine':
        pairwise_distance_function = partial(pairwise_cosine)
    else:
        raise NotImplementedError

    # initialize
    centers = initialize(X, k, seed=seed)

    best_shift = torch.inf
    best_state = None
    best_cluster = None
    iteration = 0
    while True:
        dis = pairwise_distance_function(X, centers)

        cluster_ids = torch.argmin(dis, dim=1)

        centers_pre = centers.clone()

        for index in range(k):
            selected = torch.nonzero(cluster_ids == index).squeeze()

            selected = torch.index_select(X, 0, selected)

            # https://github.com/subhadarship/kmeans_pytorch/issues/16
            if selected.shape[0] == 0:
                selected = X[torch.randint(len(X), (1,))]

            centers[index] = selected.mean(dim=0)

        center_shift = torch.sum(torch.sqrt(torch.sum((centers - centers_pre) ** 2, dim=1)))

        # increment iteration
        iteration = iteration + 1

        if center_shift ** 2 < best_shift:
            best_shift = center_shift ** 2
            best_state = centers
            best_cluster = cluster_ids
        if center_shift**2 < tol:
            break
        if iter_limit != 0 and iteration >= iter_limit:
            break

    if best_state is not None:
        return best_state, best_cluster
    else:
        return centers, cluster_ids


def pairwise_distance(data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis


def k_means(points, k, **kwargs):
    """
    Find k centroids that attempt to minimize the k- means problem:
    https://en.wikipedia.org/wiki/Metric_k-center
    Parameters
    ----------
    points:  (n, d) float
      Points in space
    k : int
      Number of centroids to compute
    **kwargs : dict
      Passed directly to scipy.cluster.vq.kmeans
    Returns
    ----------
    centroids : (k, d) float
      Points in some space
    labels: (n) int
      Indexes for which points belong to which centroid
    """
    device = points.device
    points = np.asanyarray(points.cpu().numpy(), dtype=np.float32)
    points_std = points.std(axis=0)
    points_std[points_std < 1e-12] = 1
    whitened = points / points_std
    centroids_whitened, _ = sp_kmeans(whitened, k, **kwargs)
    centroids = centroids_whitened * points_std

    # find which centroid each point is closest to
    tree = cKDTree(centroids)
    labels = tree.query(points, k=1)[1]

    return torch.from_numpy(centroids).float().to(device), \
           torch.from_numpy(labels).to(torch.int32).to(device)


def k_means_fast_euclidean(points, k):
    points_std = points.std(dim=0)
    points_std[points_std < 1e-12] = 1
    whitened = points / points_std

    centroids_whitened, labels = k_means_fast(
      whitened, k, distance='euclidean'
    )
    centroids = centroids_whitened * points_std

    return centroids, labels


def k_means_fast_cosine(points, k):
    centroids, labels = k_means_fast(
      points, k, distance='cosine'
    )
    return centroids, labels
