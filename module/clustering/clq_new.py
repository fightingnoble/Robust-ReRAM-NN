from typing import Tuple, Union

import torch
import torch.nn as nn
from sklearn.cluster._kmeans import *

# from sklearn.utils import check_random_state

NORM_PPF_0_75 = 0.6745


def check_random_state(seed, device=torch.device("cuda")):
    """Turn seed into a torch.Generator instance

    Parameters
    ----------
    seed : None | int | instance of torch.Generator
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a torch.Generator instance, return it.
        Otherwise raise ValueError.
    device:
    """
    if seed is None:
        g = torch.Generator(device=device)
        g.initial_seed()
        return g
    if isinstance(seed, int):
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return g
    if isinstance(seed, torch.Generator):
        return seed
    raise ValueError('%r cannot be used to seed a torch.Generator'
                     ' instance' % seed)


def _init_centroids(X: torch.Tensor, k, init=None,
                    random_state: Union[int, torch.Generator, None] = None,
                    x_squared_norms=None, init_size=None,
                    n=2, device=torch.device("cuda")) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the initial centroids

    Parameters
    ----------

    X : array, shape (n_samples, n_features)

    k : int
        number of centroids

    init : {'k-means++', 'random' or ndarray or callable} optional
        Method for initialization

    random_state : int, torch.Generator instance or None (default)
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    x_squared_norms : array, shape (n_samples,), optional
        Squared euclidean norm of each data point. Pass it if you have it at
        hands already to avoid it being recomputed here. Default: None

    init_size : int, optional
        Number of samples to randomly sample for speeding up the
        initialization (sometimes at the expense of accuracy): the
        only algorithm is initialized by running a batch KMeans on a
        random subset of the data. This needs to be larger than k.

    Returns
    -------
    centers : array, shape(k, n_features)
    """
    # Pick 'k' examples are random to serve as initial centroids
    # random_g = torch.Generator(device=device)
    # random_g.manual_seed(random_state)
    if hasattr(init, '__array__'):
        # ensure that the centers have the same dtype as X
        # this is a requirement of fused types of cython
        if isinstance(init, torch.Tensor):
            centers = init.to(X.device, X.dtype)
        else:
            centers = torch.tensor(init).to(X.device, X.dtype)
        seeds = None
    elif isinstance(init, str) and init == 'kaiming':
        serial = torch.arange(1, 2 * k, 2, dtype=X.dtype, device=X.device) + \
                 torch.normal(0, 0.5 / 3, (k,), dtype=X.dtype, device=X.device)  # , generator=random_g
        basis = NORM_PPF_0_75 * ((2. / n) ** 0.5) * (serial - k) / k * 2
        centers = basis.view(-1, 1)
        seeds = None
    elif isinstance(init, str) and init == 'uniform':
        serial = torch.arange(1, 2 * k, 2, dtype=X.dtype, device=X.device) + \
                 torch.normal(0, 0.5 / 3, (k,), dtype=X.dtype, device=X.device)  # , generator=random_g
        basis = (serial - k) / k * 2
        centers = basis.view(-1, 1)
        seeds = None
    else:
        n_samples = X.shape[0]
        seeds = torch.randint(n_samples, (k,))  # , generator=random_g
        centers = X[seeds]
        # the indexes get copied over so reset them

    return centers, seeds

    # random_state = check_random_state(random_state)
    # n_samples = X.shape[0]
    #
    # if x_squared_norms is None:
    #     x_squared_norms = row_norms(X, squared=True)
    #
    # if init_size is not None and init_size < n_samples:
    #     if init_size < k:
    #         warnings.warn(
    #             "init_size=%d should be larger than k=%d. "
    #             "Setting it to 3*k" % (init_size, k),
    #             RuntimeWarning, stacklevel=2)
    #         init_size = 3 * k
    #     init_indices = random_state.randint(0, n_samples, init_size)
    #     X = X[init_indices]
    #     x_squared_norms = x_squared_norms[init_indices]
    #     n_samples = X.shape[0]
    # elif n_samples < k:
    #     raise ValueError(
    #         "n_samples=%d should be larger than k=%d" % (n_samples, k))
    #
    # if isinstance(init, str) and init == 'k-means++':
    #     centers = _k_init(X, k, random_state=random_state,
    #                       x_squared_norms=x_squared_norms)
    # elif isinstance(init, str) and init == 'random':
    #     seeds = random_state.permutation(n_samples)[:k]
    #     centers = X[seeds]
    # elif hasattr(init, '__array__'):
    #     # ensure that the centers have the same dtype as X
    #     # this is a requirement of fused types of cython
    #     centers = np.array(init, dtype=X.dtype)
    # else:
    #     raise ValueError("the init parameter for the k-means should "
    #                      "be 'k-means++' or 'random' or an ndarray, "
    #                      "'%s' (type '%s') was passed." % (init, type(init)))
    #
    # if sp.issparse(centers):
    #     centers = centers.toarray()
    #
    # _validate_center_shape(X, k, centers)
    # return centers


def _labels_inertia(X: torch.Tensor, sample_weight, x_squared_norms, centers: torch.Tensor,
                    precompute_distances=True, distances=None,
                    alpha=0.1, gamma=1.0, device=torch.device("cuda")) -> Tuple[torch.Tensor, float]:
    """E step of the K-means EM algorithm.

    Compute the labels and the inertia of the given samples and centers.
    This will compute the distances in-place.

    Parameters
    ----------
    X : float64 array-like or CSR sparse matrix, shape (n_samples, n_features)
        The input samples to assign to the labels.

    sample_weight : array-like, shape (n_samples,)
        The weights for each observation in X.

    x_squared_norms : array, shape (n_samples,)
        Precomputed squared euclidean norm of each data point, to speed up
        computations.

    centers : float array, shape (k, n_features)
        The cluster centers.

    precompute_distances : boolean, default: True
        Precompute distances (faster but takes more memory).

    distances : float array, shape (n_samples,)
        Pre-allocated array to be filled in with each sample's distance
        to the closest center.

    Returns
    -------
    labels : int array of shape(n)
        The resulting assignment

    inertia : float
        Sum of squared distances of samples to their closest cluster center.
    """
    n_samples = X.shape[0]
    # sample_weight = _check_normalize_sample_weight(sample_weight, X)
    # set the default value of centers to -1 to be able to detect any anomaly
    # easily
    labels = torch.full((n_samples,), -1, dtype=torch.int8, device=X.device)
    # if distances is None:
    #     distances = torch.zeros((0,), dtype=X.dtype, device=X.device)

    # distances will be changed in-place
    if X.shape[1] == 1:
        point_norm = X ** 2
        centers_norm = centers.T ** 2
    else:
        point_norm = torch.norm(X, dim=1, keepdim=True) ** 2
        centers_norm = torch.norm(centers, dim=1, keepdim=True).T ** 2
    # print(point_norm.shape, centers_norm.shape, (point_norm + centers_norm).shape, torch.matmul(X, centers.transpose(1,0)).shape)
    similarities = point_norm + centers_norm - 2.0 * torch.matmul(X, centers.transpose(1, 0))
    var2_loss = gamma * alpha ** 2 * centers_norm
    # print("++++++")
    # print(var2_loss.sum())
    similarities = similarities + var2_loss / X.shape[1]
    # print(similarities)
    # print("++++++")
    # labels.data = torch.argmin(similarities, dim=1)
    pt2centroid, labels.data = torch.min(similarities, dim=1)
    # print(labels.shape)
    # print(pt2centroid.shape,'lllll')
    # distances.data = torch.sqrt(pt2centroid)
    inertia = pt2centroid.sum().item()
    # print(inertia)
    # print("-----")
    return labels, inertia


def _centers_dense(X: torch.Tensor, sample_weight, labels: torch.Tensor, n_clusters: int,
                   distances, alpha=0.1, gamma=1.0, device=torch.device("cuda")):  # real signature unknown
    """
    M step of the K-means EM algorithm

        Computation of cluster centers / means.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        sample_weight : array-like, shape (n_samples,)
            The weights for each observation in X.

        labels : array of integers, shape (n_samples)
            Current label assignment

        n_clusters : int
            Number of desired clusters

        distances : array-like, shape (n_samples)
            Distance to closest cluster for each sample.

        Returns
        -------
        centers : array, shape (n_clusters, n_features)
            The resulting centers
    """
    # For every centroid, recompute it as an average of the points
    # assigned to it
    centroids = torch.empty(n_clusters, X.shape[1], device=X.device)
    numCentroids = n_clusters
    for cen in range(numCentroids):
        Subset = X[labels == cen]  # all points for centroid
        if len(Subset):  # if there are points assigned to the centroid
            clusterAvg = torch.sum(Subset, dim=0) / len(Subset)
            centroids[cen] = clusterAvg
    # print((1 + gamma * alpha ** 2))
    centroids = centroids / (1 + gamma * alpha ** 2)
    return centroids


def _kmeans_single(X: torch.Tensor, sample_weight, n_clusters, max_iter=300,
                   init='k-means++', verbose=False, x_squared_norms=None,
                   random_state=None, tol=1e-4,
                   precompute_distances=True, alpha=0.1, gamma=1.0, device=torch.device("cuda")):
    """A single run of k-means, assumes preparation completed prior.

    Parameters
    ----------
    X : array-like of floats, shape (n_samples, n_features)
        The observations to cluster.

    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    sample_weight : array-like, shape (n_samples,)
        The weights for each observation in X.

    max_iter : int, optional, default 300
        Maximum number of iterations of the k-means algorithm to run.

    init : {'k-means++', 'random', or ndarray, or a callable}, optional
        Method for initialization, default to 'k-means++':

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose k observations (rows) at random from data for
        the initial centroids.

        If an ndarray is passed, it should be of shape (k, p) and gives
        the initial centers.

        If a callable is passed, it should take arguments X, k and
        and a random state and return an initialization.

    tol : float, optional
        The relative increment in the results before declaring convergence.

    verbose : boolean, optional
        Verbosity mode

    x_squared_norms : array-like
        Precomputed x_squared_norms.

    precompute_distances : boolean, default: True
        Precompute distances (faster but takes more memory).

    random_state : int, RandomState instance or None (default)
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    centroid : float ndarray with shape (k, n_features)
        Centroids found at the last iteration of k-means.

    label : integer ndarray with shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """
    # init
    best_labels, best_inertia, best_centers = None, None, None
    centers, centers_ix = _init_centroids(X, n_clusters, init, random_state, device=device)
    if verbose:
        print("Initialization complete")

    # Allocate memory to store the distances for each sample to its
    # closer center for reallocation in case of ties
    # distances = torch.zeros((X.shape[0],), dtype=X.dtype, device=X.device)

    center_shift_total = 999.0
    iter_time = max_iter
    for i in range(max_iter):
        # Save old mapping of points to centroids
        centers_old = centers.clone()
        # labels assignment is also called the E-step of EM
        labels, inertia = \
            _labels_inertia(X, sample_weight, x_squared_norms, centers,
                            precompute_distances=precompute_distances,
                            distances=None, alpha=alpha, gamma=gamma)

        # computation of the means is also called the M-step of EM
        centers = _centers_dense(X, sample_weight, labels,
                                 n_clusters, None, alpha, gamma)
        # print(centers)

        if verbose:
            print("Iteration %2d, inertia %.3f" % (i, inertia))

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.clone()
            best_centers = centers.clone()
            best_inertia = inertia

        # print(centers_old.shape, centers.shape)
        center_shift_total = torch.norm((centers_old - centers)).item()
        if center_shift_total <= tol:
            if verbose:
                print("Converged at iteration %d: "
                      "center shift %e within tolerance %e"
                      % (i, center_shift_total, tol))

            iter_time = i
            break

    if center_shift_total > 0:
        # rerun E-step in case of non-convergence so that predicted labels
        # match cluster centers
        best_labels, best_inertia = \
            _labels_inertia(X, sample_weight, x_squared_norms, best_centers,
                            precompute_distances=precompute_distances,
                            distances=None, alpha=alpha, gamma=gamma)

    return best_labels, best_inertia, best_centers, iter_time + 1

    # random_state = check_random_state(random_state)
    #
    # sample_weight = _check_normalize_sample_weight(sample_weight, X)
    #
    # best_labels, best_inertia, best_centers = None, None, None
    # # init
    # centers = _init_centroids(X, n_clusters, init, random_state=random_state,
    #                           x_squared_norms=x_squared_norms)
    # if verbose:
    #     print("Initialization complete")
    #
    # # Allocate memory to store the distances for each sample to its
    # # closer center for reallocation in case of ties
    # distances = np.zeros(shape=(X.shape[0],), dtype=X.dtype)
    #
    # # iterations
    # for i in range(max_iter):
    #     centers_old = centers.copy()
    #     # labels assignment is also called the E-step of EM
    #     labels, inertia = \
    #         _labels_inertia(X, sample_weight, x_squared_norms, centers,
    #                         precompute_distances=precompute_distances,
    #                         distances=distances)
    #
    #     # computation of the means is also called the M-step of EM
    #     if sp.issparse(X):
    #         centers = _k_means._centers_sparse(X, sample_weight, labels,
    #                                            n_clusters, distances)
    #     else:
    #         centers = _k_means._centers_dense(X, sample_weight, labels,
    #                                           n_clusters, distances)
    #         # print((1 + gamma * alpha ** 2))
    #     centers = centers / (1 + gamma * alpha ** 2)
    #
    #     if verbose:
    #         print("Iteration %2d, inertia %.3f" % (i, inertia))
    #
    #     if best_inertia is None or inertia < best_inertia:
    #         best_labels = labels.copy()
    #         best_centers = centers.copy()
    #         best_inertia = inertia
    #
    #     center_shift_total = squared_norm(centers_old - centers)
    #     if center_shift_total <= tol:
    #         if verbose:
    #             print("Converged at iteration %d: "
    #                   "center shift %e within tolerance %e"
    #                   % (i, center_shift_total, tol))
    #         break
    #
    # if center_shift_total > 0:
    #     # rerun E-step in case of non-convergence so that predicted labels
    #     # match cluster centers
    #     best_labels, best_inertia = \
    #         _labels_inertia(X, sample_weight, x_squared_norms, best_centers,
    #                         precompute_distances=precompute_distances,
    #                         distances=distances)
    #
    # return best_labels, best_inertia, best_centers, i + 1


class KmeansTorch(nn.Module):
    def __init__(self, weight_size, n_feature, n_clusters=8, init='k-means++', n_init=10,
                 max_iter=300, tol=1e-4, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True,
                 n_jobs=None, algorithm='full', alpha=0.1, gamma=1.0, device=torch.device("cuda")):
        super(KmeansTorch, self).__init__()
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.precompute_distances = precompute_distances
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs
        self.algorithm = algorithm

        self.register_buffer("labels_", torch.LongTensor(weight_size, ).to(device))
        self.register_buffer("cluster_centers_", torch.empty(n_clusters, n_feature).to(device))
        self.alpha = alpha
        self.gamma = gamma
        self.device = device

    def fit(self, X: torch.Tensor, y=None, sample_weight=None, n_init=None, init=None, tol=None):
        """Compute k-means clustering.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None).

        Returns
        -------
        self
            Fitted estimator.
        """
        # random_state = check_random_state(self.random_state, device=self.device)

        if n_init is None:
            n_init = self.n_init
        # if n_init <= 0:
        #     raise ValueError("Invalid number of initializations."
        #                      " n_init=%d must be bigger than zero." % n_init)
        #
        # if self.max_iter <= 0:
        #     raise ValueError(
        #         'Number of iterations should be a positive number,'
        #         ' got %d instead' % self.max_iter
        #     )

        variances = torch.var(X, dim=0)
        tol = torch.mean(variances) * (self.tol if tol is None else tol)

        # If the distances are precomputed every job will create a matrix of
        # shape (n_clusters, n_samples). To stop KMeans from eating up memory
        # we only activate this if the created matrix is guaranteed to be
        # under 100MB. 12 million entries consume a little under 100MB if they
        # are of type double.
        # precompute_distances = self.precompute_distances
        # if precompute_distances == 'auto':
        #     n_samples = X.shape[0]
        #     precompute_distances = (self.n_clusters * n_samples) < 12e6
        # elif isinstance(precompute_distances, bool):
        #     pass
        # else:
        #     raise ValueError(
        #         "precompute_distances should be 'auto' or True/False"
        #         ", but a value of %r was passed" %
        #         precompute_distances
        #     )

        # Validate init array
        if init is None:
            init = self.init

        # subtract of mean of x for more accurate distance computations
        X_mean = X.mean(dim=0)
        # The copy was already done above
        X -= X_mean

        if hasattr(init, '__array__'):
            # ensure that the centers have the same dtype as X
            # this is a requirement of fused types of cython
            if isinstance(init, torch.Tensor):
                init = init.to(X.device, X.dtype)
            else:
                init = torch.tensor(init).to(X.device, X.dtype)
            init -= X_mean

        # precompute squared norms of data points
        # x_squared_norms = (X * X).sum(dim=1)

        best_labels, best_inertia, best_centers, best_n_iter = None, None, None, None
        kmeans_single = _kmeans_single

        seeds = torch.randint(torch.iinfo(torch.long).max, size=(n_init,), device=self.device)
        # , generator=random_state

        # For a single thread, less memory is needed if we just store one
        # set of the best results (as opposed to one set per run per
        # thread).
        for seed in seeds:
            # run a k-means once
            labels, inertia, centers, n_iter_ = kmeans_single(
                X, sample_weight, self.n_clusters,
                max_iter=self.max_iter, init=init, verbose=self.verbose,
                precompute_distances=None, tol=tol,
                x_squared_norms=None, random_state=seed.item(), alpha=self.alpha,
                gamma=self.gamma, device=self.device)
            # print(n_iter_)
            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.clone()
                best_centers = centers.clone()
                best_inertia = inertia
                best_n_iter = n_iter_

        # fix bug: mean loss
        X += X_mean
        if hasattr(init, '__array__'):
            # ensure that the centers have the same dtype as X
            # this is a requirement of fused types of cython
            if isinstance(init, torch.Tensor):
                init = init.to(X.device, X.dtype)
            else:
                init = torch.tensor(init).to(X.device, X.dtype)
            init += X_mean
        best_centers += X_mean

        distinct_clusters = len(set(best_labels))
        if distinct_clusters < self.n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters),
                ConvergenceWarning, stacklevel=2
            )
        self.cluster_centers_.data = best_centers
        self.labels_.data = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def predict(self, X: torch.Tensor, sample_weight=None, alpha=0.1, gamma=1.0):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        # check_is_fitted(self)
        #
        x_squared_norms = None  # (X * X).sum(dim=1)

        return _labels_inertia(X, sample_weight, x_squared_norms, self.cluster_centers_
                               , alpha=alpha, gamma=gamma)[0]
