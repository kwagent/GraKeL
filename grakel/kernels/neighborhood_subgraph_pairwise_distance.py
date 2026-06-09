"""Neighborhood subgraph pairwise distance kernel :cite:`costa2010fast`."""
# Author: Ioannis Siglidis <y.siglidis@gmail.com>
# License: BSD 3 clause
import warnings
from collections import defaultdict

import joblib
import numpy as np

from scipy.sparse import csr_matrix

from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from grakel.kernels import Kernel
from grakel.graph import Graph

from grakel.kernels._c_functions import APHash

# Python 2/3 cross-compatibility import
from six import iteritems
from six.moves import filterfalse
from builtins import range
from six.moves.collections_abc import Iterable


class NeighborhoodSubgraphPairwiseDistance(Kernel):
    """The Neighborhood subgraph pairwise distance kernel.

    See :cite:`costa2010fast`.

    Parameters
    ----------
    r : int, default=3
        The maximum considered radius between vertices.

    d : int, default=4
        Neighborhood depth.

    Attributes
    ----------
    _ngx : int
        The number of graphs upon fit.

    _ngy : int
        The number of graphs upon transform.

    _fit_keys : dict
        A dictionary with keys from `0` to `_d+1`, constructed upon fit
        holding an enumeration of all the found (in the fit dataset)
        tuples of two hashes and a radius in this certain level.

    _X_level_norm_factor : dict
        A dictionary with keys from `0` to `_d+1`, that holds the self
        calculated kernel `[krg(X_i, X_i) for i=1:ngraphs_X]` for all levels.

    """

    _graph_format = "dictionary"

    def __init__(self, n_jobs=None, normalize=False, verbose=False, r=3, d=4):
        """Initialize an NSPD kernel."""
        # setup valid parameters and initialise from parent
        super(NeighborhoodSubgraphPairwiseDistance, self).__init__(
            n_jobs=n_jobs,
            normalize=normalize,
            verbose=verbose)

        self.r = r
        self.d = d
        self._initialized.update({"r": False, "d": False})

    def initialize(self):
        """Initialize all transformer arguments, needing initialization."""
        if not self._initialized["n_jobs"]:
            if type(self.n_jobs) is not int and self.n_jobs is not None:
                raise ValueError('n_jobs parameter must be an int '
                                 'indicating the number of jobs as in joblib or None')
            elif self.n_jobs is None:
                self._parallel = None
            else:
                # Use a process-based backend so per-graph hashing bypasses
                # the GIL and achieves real CPU parallelism.
                self._parallel = joblib.Parallel(n_jobs=self.n_jobs,
                                                 backend="loky",
                                                 pre_dispatch='all')
                self._n_jobs = self._parallel._effective_n_jobs()
            self._initialized["n_jobs"] = True

        if not self._initialized["r"]:
            if type(self.r) is not int or self.r < 0:
                raise ValueError('r must be a positive integer')
            self._initialized["r"] = True

        if not self._initialized["d"]:
            if type(self.d) is not int or self.d < 0:
                raise ValueError('d must be a positive integer')
            self._initialized["d"] = True

    def parse_input(self, X):
        """Parse and create features for the NSPD kernel.

        Parameters
        ----------
        X : iterable
            For the input to pass the test, we must have:
            Each element must be an iterable with at most three features and at
            least one. The first that is obligatory is a valid graph structure
            (adjacency matrix or edge_dictionary) while the second is
            node_labels and the third edge_labels (that correspond to the given
            graph format). A valid input also consists of graph type objects.

        Returns
        -------
        M : dict
            A dictionary with keys all the distances from 0 to self.d
            and values the the np.arrays with rows corresponding to the
            non-null input graphs and columns to the enumerations of tuples
            consisting of pairs of hash values and radius, from all the given
            graphs of the input (plus the fitted one's on transform).

        """
        if not isinstance(X, Iterable):
            raise TypeError('input must be an iterable\n')

        # Phase 1 (serial): validate inputs and collect Graph-ready entries.
        graph_entries = []
        for (idx, x) in enumerate(iter(X)):
            is_iter = False
            if isinstance(x, Iterable):
                is_iter, x = True, list(x)
            if is_iter and len(x) in [0, 3]:
                if len(x) == 0:
                    warnings.warn('Ignoring empty element' +
                                  ' on index: ' + str(idx))
                    continue
                else:
                    graph_entries.append(('tuple', x))
            elif type(x) is Graph:
                graph_entries.append(('graph', x))
            else:
                raise TypeError('each element of X must have either ' +
                                'a graph with labels for node and edge ' +
                                'or 3 elements consisting of a graph ' +
                                'type object, labels for vertices and ' +
                                'labels for edges.')

        ng = len(graph_entries)
        if ng == 0:
            raise ValueError('parsed input is empty')

        # Phase 2: extract features — parallel when self._parallel is set.
        # _extract_graph_hashes is a module-level function so it can be
        # pickled for the loky (process-based) parallel backend.
        if self._parallel is not None:
            all_graph_features = self._parallel(
                joblib.delayed(_extract_graph_hashes)(
                    entry, self.r, self.d, self._graph_format)
                for entry in graph_entries
            )
        else:
            all_graph_features = [
                _extract_graph_hashes(entry, self.r, self.d, self._graph_format)
                for entry in graph_entries
            ]

        # Phase 3 (serial): enumerate global feature keys and accumulate counts.
        data = defaultdict(lambda: defaultdict(int))
        all_keys = defaultdict(dict)

        for ng_idx, graph_features in enumerate(all_graph_features):
            if self._method_calling == 1:
                for (rd_key, pair_counts) in graph_features.items():
                    keys = all_keys[rd_key]
                    for hash_pair, count in pair_counts.items():
                        idx = keys.get(hash_pair, None)
                        if idx is None:
                            idx = len(keys)
                            keys[hash_pair] = idx
                        data[rd_key][ng_idx, idx] += count
            elif self._method_calling == 3:
                for (rd_key, pair_counts) in graph_features.items():
                    keys = all_keys[rd_key]
                    fit_keys = self._fit_keys[rd_key]
                    for hash_pair, count in pair_counts.items():
                        idx = fit_keys.get(hash_pair, None)
                        if idx is None:
                            idx = keys.get(hash_pair, None)
                            if idx is None:
                                idx = len(keys) + len(fit_keys)
                                keys[hash_pair] = idx
                        data[rd_key][ng_idx, idx] += count

        # Phase 4 (serial): build sparse feature matrices.
        if self._method_calling == 1:
            M = dict()
            for (key, d) in filterfalse(lambda a: len(a[1]) == 0,
                                        iteritems(data)):
                indexes, values = zip(*iteritems(d))
                rows, cols = zip(*indexes)
                M[key] = csr_matrix((values, (rows, cols)),
                                    shape=(ng, len(all_keys[key])),
                                    dtype=np.int64)
            self._fit_keys = all_keys
            self._ngx = ng

        elif self._method_calling == 3:
            M = dict()
            for (key, d) in filterfalse(lambda a: len(a[1]) == 0,
                                        iteritems(data)):
                indexes, values = zip(*iteritems(d))
                rows, cols = zip(*indexes)
                M[key] = csr_matrix((values, (rows, cols)),
                                    shape=(ng, len(all_keys[key]) + len(self._fit_keys[key])),
                                    dtype=np.int64)
            self._ngy = ng

        return M

    def transform(self, X, y=None):
        """Calculate the kernel matrix, between given and fitted dataset.

        Parameters
        ----------
        X : iterable
            Each element must be an iterable with at most three features and at
            least one. The first that is obligatory is a valid graph structure
            (adjacency matrix or edge_dictionary) while the second is
            node_labels and the third edge_labels (that fitting the given graph
            format).

        y : Object, default=None
            Ignored argument, added for the pipeline.

        Returns
        -------
        K : numpy array, shape = [n_targets, n_input_graphs]
            corresponding to the kernel matrix, a calculation between
            all pairs of graphs between target an features

        """
        self._method_calling = 3
        # Check is fit had been called
        check_is_fitted(self, ['X'])

        # Input validation and parsing
        if X is None:
            raise ValueError('transform input cannot be None')
        else:
            Y = self.parse_input(X)

        try:
            check_is_fitted(self, ['_X_level_norm_factor'])
        except NotFittedError:
            self._X_level_norm_factor = \
                {key: np.array(M.power(2).sum(-1))
                 for (key, M) in iteritems(self.X)}

        N = self._X_level_norm_factor
        S = np.zeros(shape=(self._ngy, self._ngx))
        for (key, Mp) in filterfalse(lambda x: x[0] not in self.X,
                                     iteritems(Y)):
            M = self.X[key]
            K = M.dot(Mp.T[:M.shape[1]]).toarray().T
            S += np.nan_to_num(K / np.sqrt(np.outer(np.array(Mp.power(2).sum(-1)), N[key])))

        self._Y = Y
        self._is_transformed = True
        if self.normalize:
            S /= np.sqrt(np.outer(*self.diagonal()))
        return S

    def fit_transform(self, X, y=None):
        """Fit and transform, on the same dataset.

        Parameters
        ----------
        X : iterable
            Each element must be an iterable with at most three features and at
            least one. The first that is obligatory is a valid graph structure
            (adjacency matrix or edge_dictionary) while the second is
            node_labels and the third edge_labels (that fitting the given graph
            format). If None the kernel matrix is calculated upon fit data.
            The test samples.

        Returns
        -------
        K : numpy array, shape = [n_input_graphs, n_input_graphs]
            corresponding to the kernel matrix, a calculation between
            all pairs of graphs between target an features

        """
        self._method_calling = 2
        self.fit(X)

        S, N = np.zeros(shape=(self._ngx, self._ngx)), dict()
        for (key, M) in iteritems(self.X):
            K = M.dot(M.T).toarray()
            K_diag = K.diagonal()
            N[key] = K_diag
            Q = K / np.sqrt(np.outer(K_diag, K_diag))
            np.fill_diagonal(Q, np.nan_to_num(np.diag(Q), nan=1.))
            Q = np.nan_to_num(Q)
            S = S + Q

        self._X_level_norm_factor = N

        if self.normalize:
            return S / len(self.X)
        else:
            return S

    def diagonal(self):
        """Calculate the kernel matrix diagonal of the fitted data.

        Static. Added for completeness.

        Parameters
        ----------
        None.

        Returns
        -------
        X_diag : int
            Always equal with r*d.

        Y_diag : int
            Always equal with r*d.

        """
        # constant based on normalization of krd
        check_is_fitted(self, ['X'])
        try:
            check_is_fitted(self, ['_X_diag'])
        except NotFittedError:
            # Calculate diagonal of X
            self._X_diag = len(self.X)

        try:
            check_is_fitted(self, ['_Y'])
            return self._X_diag, len(self._Y)
        except NotFittedError:
            return self._X_diag

    def _hash_neighborhoods(self, vertices, edges, Lv, Le, N, D_pair):
        """Hash all neighborhoods and all root nodes (thin wrapper)."""
        return _compute_neighborhood_hashes(
            vertices, edges, Lv, Le, N, D_pair, self.r)


def _compute_neighborhood_hashes(vertices, edges, Lv, Le, N, D_pair, r_max):
    """Hash all (radius, vertex) neighborhood subgraphs of one graph.

    Module-level so it can be pickled for process-based parallel dispatch.

    Parameters
    ----------
    vertices : set
        The graph vertices.
    edges : set
        All edges of the graph (used as the starting set at radius r_max).
    Lv : dict
        Vertex labels.
    Le : dict
        Edge labels.
    N : dict
        Neighborhoods: N[radius][v] is the list of vertices within `radius`
        hops of v.
    D_pair : dict
        Pairwise distances: D_pair[(u, v)] = shortest-path distance.
    r_max : int
        Maximum radius.

    Returns
    -------
    H : dict
        Maps (radius, vertex) → hash integer.
    """
    H, sel = dict(), sorted(list(edges))
    for v in vertices:
        re, lv, le = sel, Lv, Le
        for radius in range(r_max, -1, -1):
            sub_vertices = sorted(N[radius][v])
            re = {(i, j) for (i, j) in re
                  if i in sub_vertices and j in sub_vertices}
            lv = {vv: lv[vv] for vv in sub_vertices}
            le = {e: le[e] for e in re}
            H[radius, v] = hash_graph(D_pair, sub_vertices, re, lv, le)
    return H


def _extract_graph_hashes(entry, r_max, d_max, graph_format):
    """Parse one graph and return its (r,d) → {hash_pair: count} features.

    Module-level (picklable) for joblib loky / multiprocessing backends.
    Each call is fully independent and dominates the runtime of parse_input,
    making it the right unit of work for parallelisation.

    Parameters
    ----------
    entry : tuple
        ('tuple', [adj, vlabels, elabels]) or ('graph', Graph).
    r_max, d_max : int
        Kernel radius and distance bounds.
    graph_format : str
        Internal graph representation format.

    Returns
    -------
    features : dict
        Maps (r_val, d_val) → {(hash_A, hash_B): count}.
    """
    kind, raw = entry
    if kind == 'tuple':
        g = Graph(raw[0], raw[1], raw[2])
        g.change_format("adjacency")
    else:
        g = Graph(raw.get_adjacency_matrix(),
                  raw.get_labels(purpose="adjacency", label_type="vertex"),
                  raw.get_labels(purpose="adjacency", label_type="edge"))
    g.change_format(graph_format)
    vertices = set(g.get_vertices(purpose=graph_format))
    ed = g.get_edge_dictionary()
    edges = {(j, k) for j in ed.keys() for k in ed[j].keys()}
    Lv = g.get_labels(purpose=graph_format)
    Le = g.get_labels(purpose=graph_format, label_type="edge")
    N, D, D_pair = g.produce_neighborhoods(r_max, purpose="dictionary",
                                           with_distances=True, d=d_max)
    H = _compute_neighborhood_hashes(vertices, edges, Lv, Le, N, D_pair, r_max)
    features = {}
    for d_val in range(d_max + 1):
        if d_val not in D:
            continue
        for (A, B) in D[d_val]:
            for r_val in range(r_max + 1):
                rd_key = (r_val, d_val)
                hp = (H[r_val, A], H[r_val, B])
                level = features.setdefault(rd_key, {})
                level[hp] = level.get(hp, 0) + 1
    return features


def hash_graph(D, vertices, edges, glv, gle):
    """Make labels for hashing according to the proposed method.

    Produces the graph hash needed for fast comparison.

    Parameters
    ----------
    D_pairs : dict
        A dictionary that maps edges (tuple pairs of vertex symbols) to
        element distances (int - as produced from a BFS traversal).

    vertices : set
        A set of vertices.

    edges : set
        A set of edges.

    glv : dict
        Labels for vertices of the graph.

    gle : dict
        Labels for edges of the graph.

    Returns
    -------
    hash : int.
        The hash value for the given graph.

    """
    # Make labels for vertices
    Lv = dict()
    vertex_parts = []
    for i in vertices:
        label = "|".join(sorted([str(D[(i, j)]) + ',' + str(glv[j])
                                 for j in vertices if (i, j) in D]))
        vertex_parts.append(label + ".")
        Lv[i] = label

    # Build edge encoding using a list accumulator, then join once.
    # This avoids O(n^2) string copies that would occur with repeated +=.
    edge_parts = [Lv[i] + ',' + Lv[j] + ',' + str(gle[(i, j)]) + "_"
                  for (i, j) in edges]

    encoding = (("".join(vertex_parts)[:-1] if vertex_parts else "") + ":"
                + "".join(edge_parts))

    # Arash Partov hashing, as in the original
    # implementation of NSPK.
    return APHash(encoding)
