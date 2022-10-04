from collections import deque
from collections.abc import Iterable, MutableMapping, MutableSequence
import weakref

import numpy as np
cimport numpy as np

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
        """
        Return a mapping of child cluster labels to
        :obj:`~commonnn._bundle.Bundle` instances representing
        the children of this clustering.
        """
        return self._children

    @children.setter
    def children(self, value):
        """
        Return a mapping of child cluster labels to
        :obj:`~commonnn._bundle.Bundle` instances representing
        the children of this clustering.
        """
        if value is None:
            value = {}
        if not isinstance(value, MutableMapping):
            raise TypeError("Expected a mutable mapping")
        self._children = value

    @property
    def summary(self):
        """
        Return an instance of :obj:`commonnn.cluster.Summary`
        collecting clustering results for this clustering.
        """
        return self._summary

    @summary.setter
    def summary(self, value):
        """
        Return an instance of :obj:`cnnclustering.cluster.Summary`
        collecting clustering results for this clustering.
        """
        if value is None:
            value = Summary()
        if not isinstance(value, MutableSequence):
            raise TypeError("Expected a mutable sequence")
        self._summary = value

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
            bundle: The child to add. If `None`, creates a new bundle with
                set parent.
        Note:
            If the label already exists, the respective child is silently
            overridden. It is not checked if a children mapping exists.
        """
        if bundle is None:
            bundle = type(self)(parent=self)

        assert isinstance(bundle, Bundle)

        if isinstance(label, str):
            label = label.split(".")

        if isinstance(label, Iterable):
            label = [int(x) for x in label]

            if len(label) == 1:
                label = label[0]

        if isinstance(label, int):
            self._children[label] = bundle

        *rest, label = label
        child = self.get_child(rest)
        child._children[label] = bundle

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

        pass
        # isolate(self, purge, isolate_input_data)