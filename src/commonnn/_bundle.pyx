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
        if value is None:
            value = Summary()
        if not isinstance(value, MutableSequence):
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
                    ("coordinates", "access_coordinates"),
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
        return f"{type(self).__name__}(alias={self.alias!r}, hierarchy_lavel={self._hierarchy_level})"

    def get_child(self, label):
        """Retrieve a child of this bundle
        Args:
            label:
                Can be
                    * an integer in which case the child with the respective
                      label is returned
                    * a list of integers in which case the hierarchy of children
                      is traversed and the last child is returned
                    * a string of integers separated by a dot (e.g. "1.1.2") which
                      will be interpreted as a  list of integers (e.g. [1, 1, 2])
        Returns:
            A :obj:`~cnnclustering._bundle.Bundle`
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
                return self._children[label]
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
        bundle._children[label].alias += f"{parent_alias}.{label}"

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