from collections import deque
from collections.abc import Iterable, MutableMapping, MutableSequence
import weakref

import numpy as np
cimport numpy as np

from commonnn import helper
from commonnn.report import Summary
from commonnn._primitive_types import P_AINDEX, P_AVALUE, P_ABOOL
from commonnn._types import InputData


cdef class Bundle:
    """Bundles input data and clustering output"""

    def __cinit__(
            self,
            input_data=None,
            graph=None,
            labels=None,
            reference_indices=None,
            children=None,
            alias=None,
            parent=None,
            meta=None,
            summary=None,
            hierarchy_level=0):

        self.input_data = input_data
        self._graph = graph
        self.labels = labels
        self._reference_indices = reference_indices
        self.children = children

        if alias is None:
            alias = "root"
        self.alias = str(alias)

        if parent is not None:
            self._parent = weakref.proxy(parent)
            self.hierarchy_level = parent.hierarchy_level + 1
        else:
            self.hierarchy_level = hierarchy_level

        self.meta = meta
        self.summary = summary
        self._lambda = -np.inf
        self._size = 0
        self._checked = False

    @property
    def input_data(self):
        if self._input_data is not None:
            return self._input_data.data
        return None

    @input_data.setter
    def input_data(self, value):
        if (value is not None) & (not isinstance(value, InputData)):
            raise TypeError(
                f"Can't use object of type {type(value).__name__} as input data. "
                f"Expected type {InputData.__name__}."
                )
        self._input_data = value

    @property
    def labels(self):
        if self._labels is not None:
            return self._labels.labels
        return None

    @labels.setter
    def labels(self, value):
        if (value is not None) & (not isinstance(value, Labels)):
            value = Labels(value)
        self._labels = value

    @property
    def root_indices(self):
        if self._reference_indices is not None:
            return self._reference_indices.root
        return None

    @property
    def graph(self):
        if self._graph is None:
            return set()
        return self._graph

    @property
    def parent_indices(self):
        if self._reference_indices is not None:
            return self._reference_indices.parent
        return None

    @property
    def hierarchy_level(self):
        """
        The level of this clustering in the hierarchical
        tree of clusterings (0 for the root instance).
        """
        return self._hierarchy_level

    @hierarchy_level.setter
    def hierarchy_level(self, value):
        self._hierarchy_level = int(value)

    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        return helper.get_dict_attribute(self, "_children")

    @children.setter
    def children(self, value):
        helper.set_dict_attribute(self, "_children", value)

    @property
    def summary(self):
        return self._summary

    @summary.setter
    def summary(self, value):
        if (value is not None) and (not isinstance(value, MutableSequence)):
            raise TypeError("Expected a mutable sequence")
        self._summary = value

    @property
    def meta(self):
        return helper.get_dict_attribute(self, "_meta")

    @meta.setter
    def meta(self, value):
        helper.set_dict_attribute(self, "_meta", value)

    def info(self):
        access = []
        if self._input_data is not None:
            for kind, check in (
                    ("components", "access_components"),
                    ("distances", "access_distances"),
                    ("neighbours", "access_neighbours")):

                if self._input_data.meta.get(check, False):
                    access.append(kind)

        if not access:
            access = ["unknown"]

        n_points = self._input_data.n_points if self._input_data is not None else None

        attr_str = "\n".join([
            f"alias: {self.alias!r}",
            f"hierarchy_level: {self._hierarchy_level}",
            f"access: {', '.join(access)}",
            f"points: {n_points}",
            f"children: {len(self.children)}",
        ])

        return attr_str

    def __repr__(self):
        return f"{type(self).__name__}(alias={self.alias!r}, hierarchy_level={self._hierarchy_level})"

    def get_child(self, label):
        """Retrieve a child of this bundle

        Args:
            label:
                Can be
                    * an integer in which case the child with the respective label is returned
                    * a list of integers in which case the hierarchy of children is traversed and the last child is returned
                    * a string of integers separated by a dot (e.g. "1.1.2") which will be interpreted as a  list of integers (e.g. [1, 1, 2])

        Returns:
            A :obj:`~commonnn._bundle.Bundle`

        Note:
            It is not checked if a children mapping exists.
        """

        if isinstance(label, str):
            label = label.split(".")

        if isinstance(label, Iterable):
            label = [int(x) for x in label]

            if len(label) == 1:
                label = label[0]

        if isinstance(label, int):
            try:
                return self.children[label]
            except KeyError:
                raise KeyError(
                    f"Clustering {self.alias!r} has no child with label {label}"
                    )

        next_label, *rest = label
        try:
            return self._children[next_label].get_child(rest)
        except KeyError:
            raise KeyError(
                f"Clustering {self.alias!r} has no child with label {next_label}"
                )

    def add_child(self, label, bundle=None):
        """Add a child for this bundle

        Args:
            label: Add child with this label. Compare :func:`get_child`
                for which arguments are allowed.

        Keyword args:
            bundle: The child to add. If `None`, creates a new bundle.

        Note:
            If the label already exists, the respective child is silently
            overridden. It is not checked if a children mapping exists.

            Overrides parent of the newly added bundle.
        """
        if bundle is None:
            bundle = type(self)()

        assert isinstance(bundle, Bundle)
        bundle._parent = weakref.proxy(self)

        if isinstance(label, str):
            label = label.split(".")

        if isinstance(label, Iterable):
            label = [int(x) for x in label]

            if len(label) == 1:
                label = label[0]

        if isinstance(label, int):
            children = self.children
            children[label] = bundle
            self._children = children
            return

        *rest, label = label
        child = self.get_child(rest)
        bundle._parent = weakref.proxy(child)
        children = child.children
        children[label] = bundle
        child._children = children

    def __getitem__(self, key):
        return self.get_child(key)

    def __setitem__(self, key, value):
        self.add_child(key, value)

    cpdef void isolate(
            self,
            bint purge: bool = True,
            bint isolate_input_data: bool = True):
        """Create a child for each existing cluster label

        Note:
            see :func:`~commonnn._bundle.isolate`
        """
        isolate(self, purge, isolate_input_data)

    cpdef void reel(self, AINDEX depth: int = UINT_MAX):
        reel(self, depth)


cpdef void isolate(
        Bundle bundle,
        bint purge: bool = True,
        bint isolate_input_data: bool = True):
    """Create child clusterings from cluster labels

    Args:
        bundle: A bundle to operate on.
        purge: If `True`, creates a new mapping for the children of this
            clustering.
        isolate_input_data: If `True`, attaches a subset of the input data
            of this clustering to the child. Note, that the used
            input data type needs to support this via
            `~commonnn._types.InputData.get_subset`.

    Note:
        Does not create a child for noise points (label = 0)
    """

    cdef AINDEX label, index
    cdef list indices
    cdef AINDEX index_part_end, child_index_part_end

    if purge or (bundle._children is None):
        bundle._children = {}

    for label, indices in bundle._labels.mapping.items():
        if label == 0:
            continue

        # Assumes indices to be sorted
        parent_indices = np.array(indices, dtype=P_AINDEX)
        if bundle._reference_indices is None:
            root_indices = parent_indices
        else:
            root_indices = bundle._reference_indices.root[parent_indices]

        bundle._children[label] = Bundle(parent=bundle)
        bundle._children[label]._reference_indices = ReferenceIndices(
            root_indices,
            parent_indices
            )
        parent_alias = bundle.alias if bundle.alias is not None else ""
        bundle._children[label].alias = f"{parent_alias}.{label}"

        if not isolate_input_data:
            continue

        bundle._children[label]._input_data = bundle._input_data.get_subset(indices)

        edges = bundle._input_data.meta.get("edges", None)
        if edges is None:
            continue

        meta = bundle._children[label]._input_data.meta
        meta["edges"] = child_edges = []
        bundle._children[label]._input_data._meta = meta

        if not edges:
            continue

        edges_iter = iter(edges)
        index_part_end = next(edges_iter)
        child_index_part_end = 0

        for index in range(parent_indices.shape[0]):
            if parent_indices[index] < index_part_end:
                child_index_part_end += 1
                continue

            while parent_indices[index] >= index_part_end:
                child_edges.append(child_index_part_end)
                index_part_end += next(edges_iter)
                child_index_part_end = 0

            child_index_part_end += 1

        child_edges.append(child_index_part_end)

        while len(child_edges) < len(edges):
            child_edges.append(0)

    return


cpdef void reel(Bundle bundle, AINDEX depth: int = UINT_MAX):
    """Wrap up label assignments of lower hierarchy levels

    Args:
        depth: How many lower levels to consider. If `None`,
        consider all.
    """

    assert depth > 0

    _reel(bundle, depth)

    return


cdef inline void _reel(Bundle parent, AINDEX depth):
    cdef AINDEX n, m,
    cdef AINDEXindex, label, old_label, new_label, parent_index, n_clusters
    cdef Labels parent_labels, child_labels
    cdef AINDEX* plabels_ptr
    cdef AINDEX* clabels_ptr
    cdef Bundle child
    cdef dict meta, params

    if not parent._children:
        return

    depth -= 1

    parent_labels = parent._labels
    plabels_ptr = &parent_labels._labels[0]
    n = parent_labels._labels.shape[0]

    meta = parent_labels.meta
    meta["origin"] = "reel"
    parent_labels._meta = meta

    for label, child in parent._children.items():
        if child._labels is None:
            continue

        if (depth > 0):
            _reel(child, depth)

        n_clusters = maxint(plabels_ptr, n)

        child_labels = child._labels
        clabels_ptr = &child_labels._labels[0]
        m = child_labels._labels.shape[0]

        for index in range(m):
            old_label = clabels_ptr[index]
            if old_label == 0:
                new_label = 0
            else:
                new_label = old_label + n_clusters

            parent_index = child.parent_indices[index]
            plabels_ptr[parent_index] = new_label

        try:
            _ = parent_labels.meta["params"].pop(label)
        except KeyError:
            pass

        params = child_labels.meta.get("params", {})
        for old_label, p in params.items():
            if old_label == 0: continue
            parent_labels.meta["params"][old_label + n_clusters] = p


cpdef int fold_same_lambda(Bundle bundle) except 1:

    cdef list leafs = []
    cdef object queue = deque()
    cdef Bundle child, grandchild, candidate
    cdef AINDEX count

    for child in bundle.children.values():
        if child._lambda == bundle._lambda:
            queue.append(child)
        else:
            leafs.append(child)

    while queue:
        candidate = queue.popleft()
        for child in candidate.children.values():
            if child._lambda == bundle._lambda:
                queue.append(child)
            else:
                leafs.append(child)

    count = 1
    bundle._children = {}
    for child in leafs:
        bundle._children[count] = child
        count += 1

    return 0
    

cdef inline int _trim_small_children(Bundle bundle, AINDEX member_cutoff) except 1:
    cdef AINDEX label, count
    cdef dict children = bundle.children
    cdef dict new_children = {}

    for label, child in children.items():
        if len(child._graph) >= member_cutoff:
            new_children[label] = child

    bundle._children = new_children 

    return 0


cdef inline int _trim_lone_child(Bundle bundle) except 1:
    cdef AINDEX label, count
    cdef dict children = bundle.children

    if len(children) == 1:
        for child in children.values():
            bundle.children = child.children

    return 0


cpdef void check_children(
        Bundle bundle,
        AINDEX member_cutoff,
        bint needs_folding: bool = False):
    """Modify a bundles children mapping
    
    Note:
        These actions will be available through a number of
        hierarchy processing functions in the future

    Note:
        Always removes lone children

    Args:
        bundle: Bundle whose children to check
        member_cutoff: Children with less than this many members
            will be removed

    Keyword args:
        needs_folding: If `True`, will replace children with grand children and so
            forth if their lambda value is the same as that of the parent bundle
    """

    cdef list leafs
    cdef Bundle child, grandchild, candidate
    cdef AINDEX count, label

    # Replace children with descendants if lambda value does not change
    if needs_folding:
        leafs = []
        queue = deque()
        for child in bundle.children.values():
            if child._lambda == bundle._lambda:
                queue.append(child)
            else:
                leafs.append(child)

        while queue:
            candidate = queue.popleft()
            for child in candidate.children.values():
                if child._lambda == bundle._lambda:
                    queue.append(child)
                else:
                    leafs.append(child)

        count = 1
        bundle._children = {}
        for child in leafs:
            bundle._children[count] = child
            count += 1

    # Remove children with not enough members
    bundle._children = {
        k: v
        for k, v in enumerate(bundle.children.values(), 1)
        if len(v._graph) >= member_cutoff
        }

    # Replace lone children with grandchildren
    if len(bundle._children) == 1:
        child = bundle._children.popitem()[1]
        for label, grandchild in enumerate(child.children.values(), 1):
            bundle._children[label] = grandchild
            bundle._lambda = child._lambda

    for child in bundle.children.values():
        child._parent = weakref.proxy(bundle)

def reset_hierarchy_levels(Bundle bundle, AINDEX hierarchy_level=0) -> None:
    """Recursively reset the hierarchy levels of a bundle and its children

    Args:
        bundle: Root bundle

    Keyword args:
        hierarchy_level: The level to start from
    """

    cdef Bundle child

    bundle._hierarchy_level = hierarchy_level
    for child in bundle.children.values():
        reset_hierarchy_levels(child, hierarchy_level=hierarchy_level + 1)

def leafs_to_labels(Bundle root, n_points=None) -> None:

    cdef AINDEX label, p
    cdef Bundle b

    if n_points is None:
        try:
            n_points = root._input_data.n_points
        except AttributeError:
            try:
                n_points = root._labels.n_points
            except AttributeError:
                raise LookupError(
                    "Bundle has no input data or labels. "
                    "Please provide `n_points` explicitly."
                    )

    bundles = bfs_leafs(root)
    root.labels = Labels(
        np.zeros(n_points, order="C", dtype=P_AINDEX)
        )

    for label, b in enumerate(bundles, 1):
        for p in b.graph:
            root._labels.labels[p] = label

def bfs(Bundle root):
    cdef Bundle bundle, child

    yield root
    q = deque()
    q.append(root)

    while q:
        bundle = q.popleft()
        for child in bundle.children.values():
            yield child
            q.append(child)


def bfs_leafs(Bundle root):
    cdef Bundle bundle, child

    q = deque()
    q.append(root)

    while q:
        bundle = q.popleft()
        children = bundle.children
        if not children:
            yield bundle
        else:
            for child in children.values():
                q.append(child)

def trim_small(Bundle bundle, member_cutoff=2):
    """Scan cluster hierarchy for removable nodes

    If a cluster's child does not have enough members, it will be removed.
    """

    def _trim_small(bundle, member_cutoff):
        raise NotImplementedError('comming soon')

    _trim_small(bundle, member_cutoff)

def trim_lonechild(Bundle bundle):
    """Scan cluster hierarchy for removable nodes

    If a cluster does only have one child, the child will be replaced by
    the grandchildren.
    """
    def _trim_lonechild(Bundle bundle):
        # print(bundle.alias)

        if not bundle._children:
            # print("    has no children")
            return

        while len(bundle._children) == 1:
            # print("    has only one child")
            _, child = bundle._children.popitem()
            bundle._children = child._children
            bundle._labels._meta = child._labels._meta

            blabels = bundle._labels.labels
            clabels = child._labels.labels
            parent_indices = child._reference_indices.parent
            for i in range(child._labels.n_points):
               blabels[parent_indices[i]] = clabels[i]

        for child in bundle.children.values():
            _trim_lonechild(child)

    _trim_lonechild(bundle)

    return

def trim_shrinking(Bundle bundle):
    """Scan cluster hierarchy for removable nodes

    If a cluster does only shrink (i.e. has only one actual child with
    lower or equal member count) and does not split at any later point,
    its children will be removed.
    """
    def _trim_shrinking(Bundle bundle):

        if not bundle._children:
            splits = will_split = False
            # print(f"{bundle.alias} has no children")
        else:
            label_set = bundle._labels.set
            label_set.discard(0)

            if len(label_set) <= 1:
                splits = False
                # print(f"{bundle.alias} has not enough children")
            else:
                splits = True
                # print(f"{bundle.alias} splits")

            will_split = []
            for child in bundle.children.values():
                will_split.append(
                    _trim_shrinking(child)
                )

            will_split = any(will_split)
            # print(f"Will a child of {bundle.alias} split? {will_split}")

        keep = will_split or splits
        # print(f"Keep {bundle.alias}? {keep}")
        if not keep:
            bundle._labels = None
            bundle._children = None

        return will_split or splits

    _trim_shrinking(bundle)

    return

def trim_trivial(bundle=None):
    """Scan cluster hierarchy for removable nodes

    If the cluster label assignments on a bundle are all zero
    (noise), the clustering is considered trivial.  In this case,
    the labels and children are reset to `None`.
    """

    def _trim_trivial(Bundle bundle):
        if bundle._labels is None:
            return

        if bundle._labels.set == {0}:
            bundle._labels = None
            bundle._children = None
            return

        for child in bundle.children.values():
            _trim_trivial(child)

    _trim_trivial(bundle)

    return
