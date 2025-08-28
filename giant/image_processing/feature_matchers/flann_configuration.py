from enum import IntEnum

from dataclasses import dataclass


class FLANNIndexAlgorithmType(IntEnum):
    """
    An enum specifying the FLANN algorithm to use.

    .. Warning:: the description of each algorithm type here was generated with AI because 
                 there isn't a ton of information available in the OpenCV documentation
    """

    FLANN_INDEX_LINEAR = 0
    """
    Brute force linear search
    
    * Best for: Small datasets or when exact matches are needed
    * Parameters: None
    * Use case: Small datasets, exact nearest neighbor search
    """

    FLANN_INDEX_KDTREE = 1
    """
    K-dimensional tree
    
    * Best for: Low-dimensional data (typically ≤ 8 dimensions)
    * Parameters:
        * trees: Number of randomized trees (default: 4)
    * Use case: SIFT, SURF descriptors
    """

    FLANN_INDEX_KMEANS = 2
    """
    Hierarchical k-means clustering
    
    * Best for: High-dimensional data
    * Parameters:
        * branching: Branching factor (default: 32)
        * iterations: Maximum iterations for k-means (default: 11)
        * centers_init: Algorithm for picking the initial cluster centers for kmeans tree (default = FLANN_CENTERS_RANDOM)
        * cb_index: cluster boundary index.  Used when searching the kmeans tree (default = 0.2)
        * trees: number of kmeans trees to search in (default = 1)
    * Use case: High-dimensional feature descriptors
    """

    FLANN_INDEX_COMPOSITE = 3
    """
    Combination of randomized k-d trees and hierarchical k-means
    
    * Best for: General purpose, combines multiple algorithms
    * Parameters:
        * trees: Number of randomized trees (default = 4)
        * branching: Branching factor for k-means (default = 32)
        * iterations: Maximum iterations to perform in on kmeans clustering (kmeans tree) (default = 11)
        * centers_init: Algorithm used for picking the initial cluster centers for kmeans tree (default = FLANN_CENTERS_RANDOM)
        * cb_index: cluster boundary index. Used when searching the kmeans tree (default = 0.2f)
    * Use case: When you're unsure about data characteristics
    """

    FLANN_INDEX_KDTREE_SINGLE = 4
    """
    Single randomized k-d tree (vs multiple trees in regular KDTREE)
    
    * Best for: Low-dimensional data when you want faster build time
    * Parameters:
        * leaf_max_size: Maximum number of points in leaf nodes (default: 10)
        * reorder: reorder the tree? (default = True)
        * dim: dimension? (default = -1)
    * Use case: When build time is more important than search accuracy, or for smaller datasets
    """

    FLANN_INDEX_HIERARCHICAL = 5
    """
    Hierarchical clustering
    
    * Best for: Large datasets with hierarchical structure
    * Parameters:
        * branching: Branching factor used in the hierarchical clustering (default = 32)
        * centers_init: Algorithm used for centers initialization (default = FLANN_CENTERS_RANDOM)
        * trees: Number of trees (default=4)
        * leaf_size: Leaf size (default=100)
    * Use case: Large-scale image retrieval
    """

    FLANN_INDEX_LSH = 6
    """
    Locality Sensitive Hashing
    
    * Best for: Binary descriptors (like ORB, BRIEF, BRISK)
    * Parameters:
        * table_number: Number of hash tables (default = 12)
        * key_size: the length of the key in the hash tables (default=20)
        * multi_probe_level: number of levels to use in multi-probe (0 for standard LSH) (default=2)
    * Use case: Binary feature descriptors
    """

    FLANN_INDEX_SAVED = 254
    """
    Index saved to a file?
    
    Generally shouldn't be used in GIANT
    """

    FLANN_INDEX_AUTOTUNED = 255
    """
    Automatically selects the best algorithm based on data

    * Best for: Automatic algorithm selection
    * Parameters:
        * target_precision: Desired precision (0-1) (default = 0.8)
        * build_weight: Weight for build time vs search time (default = 0.01)
        * memory_weight: Weight for memory usage (default = 0.0)
        * sample_fraction: Fraction of data to use for testing (default = 0.1)
    * Use case: When you want optimal performance without manual tuning
    """


class FLANNCentersInit(IntEnum):
    """
    Enum specifying the algorithm to use to initialize kmeans centers
    """
    
    FLANN_CENTERS_RANDOM = 0
    FLANN_CENTERS_GONZALES = 1
    FLANN_CENTERS_KMEANSPP = 2
    FLANN_CENTERS_GROUPWISE = 3


class FLANNDistance(IntEnum):
    FLANN_DIST_EUCLIDEAN = 1
    FLANN_DIST_L2 = 1
    FLANN_DIST_MANHATTAN = 2
    FLANN_DIST_L1 = 2
    FLANN_DIST_MINKOWSKI = 3
    FLANN_DIST_MAX = 4
    FLANN_DIST_HIST_INTERSECT = 5
    FLANN_DIST_HELLINGER = 6
    FLANN_DIST_CHI_SQUARE = 7
    FLANN_DIST_CS = 7
    FLANN_DIST_KULLBACK_LEIBLER = 8
    FLANN_DIST_KL = 8
    FLANN_DIST_HAMMING = 9
    FLANN_DIST_DNAMMING = 10
    

@dataclass
class FLANNIndexLinearParams:
    """
    Parameters for initializing brute force linear search

    * Best for: Small datasets or when exact matches are needed
    * Parameters: None
    * Use case: Small datasets, exact nearest neighbor search
    
    .. Warning:: Much of the description here was generated by AI since
                 there isn't a ton of information available in the OpenCV documentation
    """

    pass


@dataclass
class FLANNIndexKdTreeParams:
    """
    Parameters for configuring k-dimensional tree

    * Best for: Low-dimensional data (typically ≤ 8 dimensions)
    * Use case: SIFT, SURF descriptors
    
    .. Warning:: Much of the description here was generated by AI since
                 there isn't a ton of information available in the OpenCV documentation
    """

    trees: int = 4
    """
    Number of randomized trees 
    """


@dataclass
class FLANNIndexKMeansParams:
    """
    Parameters for config of hierarchical k-means clustering

    * Best for: High-dimensional data
    * Use case: High-dimensional feature descriptors
    
    .. Warning:: Much of the description here was generated by AI since
                 there isn't a ton of information available in the OpenCV documentation
    """

    branching: int = 32 
    """
    Branching factor 
    """

    iterations: int = 11
    """
    Maximum iterations for k-means
    """

    centers_init: FLANNCentersInit = FLANNCentersInit.FLANN_CENTERS_RANDOM
    """
    Algorithm for picking the initial cluster centers for kmeans tree 
    """

    cb_index: float = 0.2
    """
    cluster boundary index.  Used when searching the kmeans tree
    """
    
    trees: int = 1
    """
    number of kmeans trees to search in
    """


@dataclass
class FLANNIndexCompositeParams:
    """
    Parameters for config of combination of randomized k-d trees and hierarchical k-means

    * Best for: General purpose, combines multiple algorithms
    * Use case: When you're unsure about data characteristics
    
    .. Warning:: Much of the description here was generated by AI since
                 there isn't a ton of information available in the OpenCV documentation
    """
    
    trees: int = 4
    """
    Number of randomized trees 
    """
    
    branching: int = 32 
    """
    Branching factor for k-means 
    """
    
    iterations: int = 11
    """
    Maximum iterations to perform in on kmeans clustering (kmeans tree) 
    """
    
    centers_init: FLANNCentersInit = FLANNCentersInit.FLANN_CENTERS_RANDOM
    """
    Algorithm used for picking the initial cluster centers for kmeans tree 
    """
    
    cb_index: float = 0.2
    """
    cluster boundary index. Used when searching the kmeans tree 
    """


@dataclass
class FLANNIndexKdTreeSingleParams:
    """
    Parameters for configuring single randomized k-d tree (vs multiple trees in regular KDTREE)

    * Best for: Low-dimensional data when you want faster build time
    * Use case: When build time is more important than search accuracy, or for smaller datasets
    
    .. Warning:: Much of the description here was generated by AI since
                 there isn't a ton of information available in the OpenCV documentation
    """
    
    leaf_max_size: int = 10
    """
    Maximum number of points in leaf nodes 
    """
    
    reorder: bool = True 
    """
    reorder the tree? 
    """
    
    dim: int = -1
    """
    dimension? 
    """


@dataclass
class FLANNIndexHierarchicalParams:
    """
    Parameters for config of hierarchical clustering

    * Best for: Large datasets with hierarchical structure
    * Use case: Large-scale image retrieval
    
    .. Warning:: Much of the description here was generated by AI since
                 there isn't a ton of information available in the OpenCV documentation
    """
    
    branching: int = 32
    """
    Branching factor used in the hierarchical clustering 
    """
    
    centers_init: FLANNCentersInit = FLANNCentersInit.FLANN_CENTERS_RANDOM
    """
    Algorithm used for centers initialization 
    """
    
    trees: int = 4
    """
    Number of trees 
    """
    
    leaf_size: int = 100
    """
    Leaf size 
    """


@dataclass
class FLANNIndexLSHParams:
    """
    Paramters for config of locality sensitive hashing

    * Best for: Binary descriptors (like ORB, BRIEF, BRISK)
    * Use case: Binary feature descriptors
    
    .. Warning:: Much of the description here was generated by AI since
                 there isn't a ton of information available in the OpenCV documentation
    """
    table_number: int = 12
    """
    Number of hash tables 
    """
    key_size: int = 20
    """
    the length of the key in the hash tables 
    """
    
    multi_probe_level: int = 2
    """
    number of levels to use in multi-probe (0 for standard LSH) 
    """


@dataclass
class FLANNIndexAutotunedParams:
    """
    Parameters for confi of automatically selecting the best algorithm based on data

    * Best for: Automatic algorithm selection
    * Use case: When you want optimal performance without manual tuning
    
    .. Warning:: Much of the description here was generated by AI since
                 there isn't a ton of information available in the OpenCV documentation
    """
    target_precision: float = 0.8 
    """
    Desired precision (0-1) 
    """
    
    build_weight: float = 0.01
    """
    Weight for build time vs search time 
    """
    
    memory_weight: float = 0.0
    """
    Weight for memory usage 
    """
    
    sample_fraction: float = 0.1
    """
    Fraction of data to use for testing 
    """


FLANN_INDEX_PARAM_TYPES = FLANNIndexAutotunedParams | FLANNIndexCompositeParams | \
                          FLANNIndexHierarchicalParams | FLANNIndexKdTreeParams | \
                          FLANNIndexKdTreeSingleParams | FLANNIndexKMeansParams | \
                          FLANNIndexLinearParams | FLANNIndexLSHParams
"""
An alias for all the potential FLANN Index parameter dataclasses
"""


@dataclass 
class FLANNSearchParams:
    """
    Parameters for configuring how FLANN does a search
    """
    
    checks: int = 32
    """
    How many leafs to visit when searching for neighbours (-1 for unlimited)
    """
    
    eps: float = 0.0
    """
    Search for eps-approximate neighbours
    """
    
    sorted: bool = True
    """
    If false, search stops at the tree reaching the number of max checks (original behavior).
    When Ture, we do a descent in each tree and, like before the alternative paths stored in 
    the heap are not be processed further when max checks is reached.
    """

