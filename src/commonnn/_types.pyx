from collections import Counter, defaultdict
from itertools import count
from typing import Any, Optional, Type

import numpy as np

from commonnn._primitive_types import P_AINDEX, P_AVALUE, P_ABOOL


cdef class ClusterParameters:
    """Input parameters for clustering procedure"""

    _fparam_names = []
    _iparam_names = []

    def __cinit__(self, fparams: list, iparams: list, *, **kwargs):
        self.fparams = _allocate_and_fill_avalue_array(len(fparams), fparams)
        self.iparams = _allocate_and_fill_aindex_array(len(iparams), iparams)

    def __init__(self, *args, **kwargs):
        if type(self) is ClusterParameters:
            raise RuntimeError(
                f"Cannot instantiate abstract class {type(self)}"
                )

    def __dealloc__(self):
        if self.fparams != NULL:
            free(self.fparams)

        if self.iparams != NULL:
            free(self.iparams)

    @classmethod
    def from_mapping(cls, parameters: dict, *, **kwargs):
        fparams = [parameters[pn] for pn in cls._fparam_names]
        iparams = [parameters[pn] for pn in cls._iparam_names]
        return cls(fparams, iparams, **kwargs)

    def to_dict(self):
        """Return a Python dictionary of cluster parameter key-value pairs"""

        cdef AINDEX i

        dict_ = {}
        for i, pn in enumerate(self._fparam_names):
            dict_[pn] = self.fparams[i]

        for i, pn in enumerate(self._iparam_names):
            dict_[pn] = self.iparams[i]

        return dict_

    def __repr__(self):
        return f"{self.to_dict()!r}"

    def __str__(self):
        return f"{self.to_dict()!s}"


cdef class CommonNNParameters(ClusterParameters):
    _fparam_names = ["radius_cutoff"]
    _iparam_names = ["similarity_cutoff", "_support_cutoff", "start_label"]


cdef class Labels:
    """Represents cluster label assignments"""

    def __cinit__(self, labels not None, *, consider=None, meta=None):

        self._labels = labels

        if consider is None:
            self._consider = np.ones_like(self._labels, dtype=P_ABOOL)
        else:
            self._consider = consider

            if self._labels.shape[0] != self._consider.shape[0]:
                raise ValueError(
                    "'labels' and 'consider' must have the same length"
                    )

        if meta is None:
            meta = {}
        self.meta = meta

    def __init__(self, labels, *, consider=None, meta=None):
        """
        Args:
            labels: A container of integer cluster labels
                supporting the buffer protocol

        Keyword args:
            consider: A boolean (uint8) container of same length as `labels`
                indicating whether a cluster label should be considered for assignment
                during clustering.  If `None`, will be created as all true.
            meta: Meta information.  If `None`, will be created as empty
                dictionary.

        Attributes:
            n_points: The length of the labels container
            labels: The labels container converted to a NumPy array
            meta: The meta information dictionary
            consider: The consider container converted to a NumPy array
            mapping: A mapping of cluster labels to indices in `labels`
            set: The set of cluster labels
            consider_set: A set of cluster labels to consider for cluster
                label assignments
        """
        pass

    @property
    def mapping(self):
        return self.to_mapping()

    @property
    def set(self):
        return self.to_set()

    @property
    def labels(self):
        return np.asarray(self._labels)

    @property
    def consider(self):
        return np.asarray(self._consider)

    @property
    def n_points(self):
        return self._labels.shape[0]

    @property
    def consider_set(self):
        return self._consider_set

    @consider_set.setter
    def consider_set(self, value):
        self._consider_set = value

    def __repr__(self):
        return f"{type(self).__name__}({list(self.labels)!s})"

    def __str__(self):
        return f"{self.labels!s}"

    @classmethod
    def from_sequence(cls, labels, *, consider=None, meta=None) -> Type[Labels]:
        """Construct from any sequence (not supporting the buffer protocol)"""

        labels = np.array(labels, order="C", dtype=P_AINDEX)

        if consider is not None:
            consider = np.array(consider, order="C", dtype=P_ABOOL)

        return cls(labels, consider=consider, meta=meta)

    @classmethod
    def from_length(cls, n: int, meta=None) -> Type[Labels]:
        """Construct all zero labels with length"""

        labels = np.zeros(n, order="C", dtype=P_AINDEX)
        return cls(labels, meta=meta)


    def to_mapping(self):
        """Convert labels container to `mapping` of labels to lists of point indices"""

        cdef AINDEX index, label

        mapping = defaultdict(list)

        for index in range(self._labels.shape[0]):
            label = self._labels[index]
            mapping[label].append(index)

        return mapping

    def to_set(self):
        """Convert labels container to `set` of unique labels"""

        cdef AINDEX index, label
        cdef set label_set = set()

        for index in range(self._labels.shape[0]):
            label = self._labels[index]
            label_set.add(label)

        return label_set

    def sort_by_size(
            self,
            member_cutoff: Optional[int] = None,
            max_clusters: Optional[int] = None,
            bundle=None):
        """Sort labels by clustersize in-place

        Re-assigns cluster numbers so that the biggest cluster (that is
        not noise) is cluster 1.  Also filters out clusters, that have
        not at least `member_cutoff` members (default 2).
        Optionally, does only
        keep the `max_clusters` largest clusters.

        Args:
           member_cutoff: Valid clusters need to have at least this
              many members.
           max_clusters: Only keep this many clusters.
        """

        cdef AINDEX _max_clusters, _member_cutoff, cluster_count
        cdef AINDEX index, old_label, new_label, member_count
        cdef dict reassign_map, params

        if member_cutoff is None:
            member_cutoff = 2
        _member_cutoff = member_cutoff

        frequencies = Counter(self._labels)
        if 0 in frequencies:
            _ = frequencies.pop(0)

        if frequencies:
            if max_clusters is None:
               _max_clusters = len(frequencies)
            else:
               _max_clusters = max_clusters

            order = frequencies.most_common()
            reassign_map = {}
            reassign_map[0] = 0

            new_labels = count(1)
            for cluster_count, (old_label, member_count) in enumerate(order, 1):
                if cluster_count > _max_clusters:
                    reassign_map[old_label] = 0
                    continue

                if member_count >= _member_cutoff:
                    new_label = next(new_labels)
                    reassign_map[old_label] = new_label
                    continue

                reassign_map[old_label] = 0

            for index in range(self._labels.shape[0]):
                old_label = self._labels[index]
                self._labels[index] = reassign_map[old_label]

            params = self.meta.get("params", {})
            self.meta["params"] = {
                reassign_map[k]: v
                for k, v in params.items()
                if (k in reassign_map) and (reassign_map[k] != 0)
                }

            if bundle is not None:
                if bundle._children is not None:
                    processed_children = {}
                    for old_label, child in bundle._children.items():
                        new_label = reassign_map[old_label]
                        if new_label != 0:
                            processed_children[new_label] = child
                    bundle._children = processed_children

        return


cdef class ReferenceIndices:
    """Root and parent indices relating child with parent clusterings"""

    def __cinit__(self, root_indices not None, parent_indices not None):
        self._root = root_indices
        self._parent = parent_indices

    @property
    def root(self):
        return np.asarray(self._root)

    @property
    def parent(self):
        return np.asarray(self._parent)