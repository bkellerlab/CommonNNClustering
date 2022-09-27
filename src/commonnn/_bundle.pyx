from collections.abc import Iterable, MutableMapping, MutableSequence
import weakref

import numpy as np
cimport numpy as np

from commonnn._primitive_types import P_AINDEX, P_AVALUE, P_ABOOL


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
        # self.summary = summary
        self._lambda = -np.inf

    @property
    def input_data(self):
        if self._input_data is not None:
            return self._input_data.data
        return None

    @input_data.setter
    def input_data(self, value):
        # if (value is not None) & (not isinstance(value, InputData)):
        #     raise TypeError(
        #         f"Can't use object of type {type(value).__name__} as input data. "
        #         f"Expected type {InputData.__name__}."
        #         )
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