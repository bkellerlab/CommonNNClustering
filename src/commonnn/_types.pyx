from abc import ABC, abstractmethod
from collections import Counter, defaultdict, deque
from collections.abc import Container, Iterator, Sequence
import heapq
from itertools import count
from typing import Any, Optional, Type

import numpy as np

try:
    import sklearn.neighbors
    SKLEARN_FOUND = True
except ModuleNotFoundError:
    SKLEARN_FOUND = False  # pragma: no cover

from commonnn import helper
from commonnn._primitive_types import P_AINDEX, P_AVALUE, P_ABOOL

from libc.math cimport sqrt as csqrt, pow as cpow, fabs as cfabs
from cython.operator cimport dereference, preincrement

cdef class ClusterParameters:
    """Input parameters for clustering procedure"""

    _fparam_names = []
    _iparam_names = []

    _defaults = {}

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
        fparams = []
        iparams = []

        for pn in cls._fparam_names:
            p = parameters.get(pn, cls._defaults.get(pn))
            if p is None: raise KeyError(f"{pn}")
            fparams.append(p)

        for pn in cls._iparam_names:
            p = parameters.get(pn, cls._defaults.get(pn))
            if p is None: raise KeyError(f"{pn}")
            iparams.append(p)

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

    def get_iparam(self, AINDEX i):
        return self.iparams[i]

    def get_fparam(self, AINDEX i):
        return self.fparams[i]


cdef class CommonNNParameters(ClusterParameters):
    _fparam_names = ["radius_cutoff"]
    _iparam_names = ["similarity_cutoff", "_support_cutoff", "start_label"]

    _defaults = {
        "_support_cutoff": 0,
        "start_label": 1
    }

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
    def meta(self):
        return helper.get_dict_attribute(self, "_meta")

    @meta.setter
    def meta(self, value):
        helper.set_dict_attribute(self, "_meta", value)

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


class InputData(ABC):
    """Defines the input data interface"""

    @property
    @abstractmethod
    def data(self):
        """Return underlying data (only for user convenience, not to be relied on)"""

    @property
    def meta(self):
        return helper.get_dict_attribute(self, "_meta")

    @meta.setter
    def meta(self, value):
        helper.set_dict_attribute(self, "_meta", value)

    @property
    @abstractmethod
    def n_points(self) -> int:
        """Return total number of points"""

    @abstractmethod
    def get_subset(self, indices: Container) -> Type['InputData']:
        """Return input data subset"""

    @classmethod
    def get_builder_kwargs(cls):
        return []

    def __repr__(self):  # pragma: no cover
        return f"{type(self).__name__}"


class InputDataComponents(InputData):
    """Extends the input data interface for point coordinates"""

    @property
    @abstractmethod
    def n_dim(self) -> int:
        """Return total number of dimensions"""

    @abstractmethod
    def get_component(self, point: int, dimension: int) -> float:
        """Return one component of point coordinates"""

    @abstractmethod
    def to_components_array(self) -> Type[np.ndarray]:
        """Return input data as NumPy array of shape (#points, #components)"""

    def __str__(self):
        return f"components of {self.n_points} points in {self.n_dim} dimensions"


class InputDataPairwiseDistances(InputData):
    """Extends the input data interface for inter-point distances"""

    @abstractmethod
    def get_distance(self, point_a: int, point_b: int) -> float:
        """Return the pairwise distance between two points"""

    def __str__(self):
        return f"distances of {self.n_points} points"


class InputDataPairwiseDistancesComputer(InputDataPairwiseDistances):
    """Extends the distance input data interface for computable distances"""

    @abstractmethod
    def compute_distances(self, input_data: Type["InputData"]) -> None:
        """Pre-compute pairwise distances"""


class InputDataNeighbourhoods(InputData):
    """Extends the input data interface for point neighbourhoods"""

    @abstractmethod
    def get_n_neighbours(self, point: int) -> int:
        """Return number of neighbours for point"""

    @abstractmethod
    def get_neighbour(self, point: int, member: int) -> int:
        """Return a member for point"""

    def __str__(self):
        return f"neighbourhoods of {self.n_points} points"

class InputDataNeighbourhoodsComputer(InputDataNeighbourhoods):
    """Extends the neighbourhood input data interface for computable neighbourhoods"""

    @abstractmethod
    def compute_neighbourhoods(
            self,
            input_data: Type["InputData"], r: float,
            is_sorted: bool = False, is_selfcounting: bool = True) -> None:
        """Pre-compute neighbourhoods at radius"""


cdef class InputDataExtInterface:
    """Defines the input data interface for Cython extension types"""

    cdef AVALUE _get_component(
            self, const AINDEX point, const AINDEX dimension) nogil: ...

    def get_component(self, point: int, dimension: int) -> int:
        return self._get_component(point, dimension)

    cdef AINDEX _get_n_neighbours(self, const AINDEX point) nogil: ...

    def get_n_neighbours(self, point: int) -> int:
        return self._get_n_neighbours(point)

    cdef AINDEX _get_neighbour(self, const AINDEX point, const AINDEX member) nogil: ...

    def get_neighbour(self, point: int, member: int) -> int:
        return self._get_neighbour(point, member)

    cdef AVALUE _get_distance(self, const AINDEX point_a, const AINDEX point_b) nogil: ...

    def get_distance(self, point_a: int, point_b: int) -> int:
        return self._get_distance(point_a, point_b)

    cdef void _compute_distances(self, InputDataExtInterface input_data) nogil: ...

    def compute_distances(self, InputDataExtInterface input_data):
        self._compute_distances(input_data)

    cdef void _compute_neighbourhoods(
            self,
            InputDataExtInterface input_data, AVALUE r,
            ABOOL is_sorted, ABOOL is_selfcounting) nogil: ...

    def compute_neighbourhoods(
            self,
            InputDataExtInterface input_data, AVALUE r,
            ABOOL is_sorted, ABOOL is_selfcounting):
        self._compute_neighbourhoods(input_data, r, is_sorted, is_selfcounting)

    @classmethod
    def get_builder_kwargs(cls):
        return []

    def __repr__(self):  # pragma: no cover
        return f"{type(self).__name__}"

    @property
    def meta(self):
        return helper.get_dict_attribute(self, "_meta")

    @meta.setter
    def meta(self, value):
        helper.set_dict_attribute(self, "_meta", value)


class Neighbours(ABC):
    """Defines the neighbours interface"""

    @abstractmethod
    def to_neighbours_array(self):
       """Return point indices as NumPy array"""

    @property
    @abstractmethod
    def n_points(self) -> int:
       """Return total number of points"""

    @abstractmethod
    def assign(self, member: int) -> None:
       """Add a member to this container"""

    @abstractmethod
    def reset(self) -> None:
       """Reset/empty this container"""

    @abstractmethod
    def enough(self, member_cutoff: int) -> bool:
        """Return True if there are enough points"""

    @abstractmethod
    def get_member(self, index: int) -> int:
       """Return indexable neighbours container"""

    @abstractmethod
    def contains(self, member: int) -> bool:
       """Return True if member is in neighbours container"""

    def __repr__(self):
        return f"{type(self).__name__}"

    @classmethod
    def get_builder_kwargs(cls):
        return []


class NeighboursGetter(ABC):
    """Defines the neighbours-getter interface"""

    @property
    @abstractmethod
    def is_sorted(self) -> bool:
       """Return True if neighbour indices are sorted"""

    @property
    @abstractmethod
    def is_selfcounting(self) -> bool:
       """Return True if points count as their own neighbour"""

    @abstractmethod
    def get(
            self,
            index: int,
            input_data: Type['InputData'],
            neighbours: Type['Neighbours'],
            cluster_params: Type['ClusterParameters']) -> None:
        """Collect neighbours for point in input data"""

    def get_other(
            self,
            index: int,
            input_data: Type['InputData'],
            other_input_data: Type['InputData'],
            neighbours: Type['Neighbours'],
            cluster_params: Type['ClusterParameters']) -> None:
        """Collect neighbours in input data for point in other input data"""

    def __repr__(self):
        return f"{type(self).__name__}"

    @classmethod
    def get_builder_kwargs(cls):
        return []


cdef class NeighboursExtInterface:

    cdef void _assign(self, const AINDEX member) nogil: ...
    cdef void _reset(self) nogil: ...
    cdef bint _enough(self, const AINDEX member_cutoff) nogil: ...
    cdef AINDEX _get_member(self, const AINDEX index) nogil: ...
    cdef bint _contains(self, const AINDEX member) nogil: ...

    def assign(self, member: int):
        self._assign(member)

    def reset(self):
        self._reset()

    def enough(self, member_cutoff: int):
        return self._enough(member_cutoff)

    def get_member(self, index: int):
        return self._get_member(index)

    def contains(self, member: int):
        return self._contains(member)

    def __str__(self):
        return f"{type(self).__name__}"

    @classmethod
    def get_builder_kwargs(cls):
        return []


cdef class NeighboursGetterExtInterface:

    cdef void _get(
            self,
            const AINDEX index,
            InputDataExtInterface input_data,
            NeighboursExtInterface neighbours,
            ClusterParameters cluster_params) nogil: ...

    cdef void _get_other(
            self,
            const AINDEX index,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data,
            NeighboursExtInterface neighbours,
            ClusterParameters cluster_params) nogil: ...

    def get(
            self,
            AINDEX index,
            InputDataExtInterface input_data,
            NeighboursExtInterface neighbours,
            ClusterParameters cluster_params):
        self._get(index, input_data, neighbours, cluster_params)


    def get_other(
            self,
            AINDEX index,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data,
            NeighboursExtInterface neighbours,
            ClusterParameters cluster_params):
        self._get_other(
            index, input_data, other_input_data,
            neighbours, cluster_params
            )

    def __repr__(self):
        return f"{type(self).__name__}"

    @classmethod
    def get_builder_kwargs(cls):
        return []


class DistanceGetter(ABC):
    """Defines the distance getter interface"""

    @abstractmethod
    def get_single(
            self,
            point_a: int, point_b: int,
            input_data: Type['InputData']) -> float:
        """Get distance between two points in input data"""

    @abstractmethod
    def get_single_other(
            self,
            point_a: int, point_b: int,
            input_data: Type['InputData'],
            other_input_data: Type['InputData']) -> float:
        """Get distance between two points in input data and other input data"""

    def __repr__(self):
        return f"{type(self).__name__}"

    @classmethod
    def get_builder_kwargs(cls):
        return []


cdef class DistanceGetterExtInterface:
    cdef AVALUE _get_single(
            self,
            const AINDEX point_a,
            const AINDEX point_b,
            InputDataExtInterface input_data) nogil: ...

    cdef AVALUE _get_single_other(
            self,
            const AINDEX point_a,
            const AINDEX point_b,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data) nogil: ...

    def get_single(
            self,
            AINDEX point_a,
            AINDEX point_b,
            InputDataExtInterface input_data):

        return self._get_single(point_a, point_b, input_data)

    def get_single_other(
            self,
            AINDEX point_a,
            AINDEX point_b,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data):

        return self._get_single_other(point_a, point_b, input_data, other_input_data)

    def __repr__(self):
        return f"{type(self).__name__}"

    @classmethod
    def get_builder_kwargs(cls):
        return []


class Metric(ABC):
    """Defines the metric-interface"""

    @abstractmethod
    def calc_distance(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData']) -> float:
        """Return distance between two points in input data"""

    @abstractmethod
    def calc_distance_other(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData'],
            other_input_data: Type['InputData']) -> float:
        """Return distance between two points in input data and other input data"""

    def __repr__(self):
        return f"{type(self).__name__}"


cdef class MetricExtInterface:
    """Defines the metric interface for extension types"""

    cdef AVALUE _calc_distance(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data) nogil: ...

    cdef AVALUE _calc_distance_other(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data) nogil: ...

    cdef AVALUE _adjust_radius(self, AVALUE radius_cutoff) nogil: ...

    def calc_distance(
            self,
            AINDEX index_a, AINDEX index_b,
            InputDataExtInterface input_data) -> float:

        return self._calc_distance(index_a, index_b, input_data)

    def calc_distance_other(
            self,
            AINDEX index_a, AINDEX index_b,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data) -> float:

        return self._calc_distance_other(
            index_a, index_b, input_data, other_input_data
            )

    def adjust_radius(self, AVALUE radius_cutoff) -> float:
        return self._adjust_radius(radius_cutoff)

    def __str__(self):
        return f"{type(self).__name__}"

    @classmethod
    def get_builder_kwargs(cls):
        return []


class SimilarityChecker(ABC):
    """Defines the similarity checker interface"""

    @abstractmethod
    def check(
            self,
            neighbours_a: Type["Neighbours"],
            neighbours_b: Type["Neighbours"],
            cluster_params: Type['ClusterParameters']) -> bool:
        """Return True if a and b have sufficiently many common neighbours"""

    @abstractmethod
    def get(
            self,
            neighbours_a: Type["Neighbours"],
            neighbours_b: Type["Neighbours"],
            cluster_params: Type['ClusterParameters']) -> int:
        """Return number of common neighbours"""

    def __repr__(self):
        return f"{type(self).__name__}"

    @classmethod
    def get_builder_kwargs(cls):
        return []


cdef class SimilarityCheckerExtInterface:
    """Defines the similarity checker interface for extension types"""

    cdef bint _check(
            self,
            NeighboursExtInterface neighbours_a,
            NeighboursExtInterface neighbours_b,
            ClusterParameters cluster_params) nogil: ...

    def check(
            self,
            NeighboursExtInterface neighbours_a,
            NeighboursExtInterface neighbours_b,
            ClusterParameters cluster_params):

        return self._check(neighbours_a, neighbours_b, cluster_params)

    cdef AINDEX _get(
            self,
            NeighboursExtInterface neighbours_a,
            NeighboursExtInterface neighbours_b,
            ClusterParameters cluster_params) nogil: ...

    def get(
            self,
            NeighboursExtInterface neighbours_a,
            NeighboursExtInterface neighbours_b,
            ClusterParameters cluster_params):

        return self._get(neighbours_a, neighbours_b, cluster_params)

    def __repr__(self):
        return f"{type(self).__name__}"

    @classmethod
    def get_builder_kwargs(cls):
        return []


class Queue(ABC):
    """Defines the queue interface"""

    @abstractmethod
    def push(self, value):
        """Put value into the queue"""

    @abstractmethod
    def pop(self):
        """Retrieve value from the queue"""

    @abstractmethod
    def is_empty(self) -> bool:
        """Return True if there are no values in the queue"""

    @abstractmethod
    def size(self) -> int:
        """Get number of items in the queue"""

    def __repr__(self):
        return f"{type(self).__name__}"

    @classmethod
    def get_builder_kwargs(cls):
        return []

cdef class QueueExtInterface:

    cdef void _push(self, const AINDEX value) nogil: ...
    cdef AINDEX _pop(self) nogil: ...
    cdef bint _is_empty(self) nogil: ...
    cdef AINDEX _size(self) nogil: ...

    def push(self, value: int):
        self._push(value)

    def pop(self) -> int:
        return self._pop()

    def is_empty(self) -> bool:
        return self._is_empty()

    def size(self) -> int:
        return self._size()

    def __repr__(self):
        return f"{type(self).__name__}"

    @classmethod
    def get_builder_kwargs(cls):
        return []


class PriorityQueue(ABC):
    """Defines the prioqueue interface"""

    @abstractmethod
    def reset(self) -> None:
        """Reset the queue"""

    @abstractmethod
    def size(self) -> int:
        """Get number of items in the queue"""

    @abstractmethod
    def push(self, a, b, weight) -> None:
        """Put values into the queue"""

    @abstractmethod
    def pop(self) -> (int, int, float):
        """Retrieve values from the queue"""

    @abstractmethod
    def is_empty(self) -> bool:
        """Return True if there are no values in the queue"""

    def __repr__(self):
        return f"{type(self).__name__}"

    @classmethod
    def get_builder_kwargs(cls):
        return []


cdef class PriorityQueueExtInterface:

    cdef void _reset(self) nogil: ...
    cdef void _push(self, const AINDEX a, const AINDEX b, const AVALUE weight) nogil: ...
    cdef (AINDEX, AINDEX, AVALUE) _pop(self) nogil: ...
    cdef bint _is_empty(self) nogil: ...
    cdef AINDEX _size(self) nogil: ...

    def push(self, a: int, b: int, weight: float):
        self._push(a, b, weight)

    def pop(self) -> (int, int, float):
        return self._pop()

    def is_empty(self) -> bool:
        return self._is_empty()

    def size(self) -> int:
        return self._size()

    def reset(self) -> None:
        return self._reset()

    def __repr__(self):
        return f"{type(self).__name__}"

    @classmethod
    def get_builder_kwargs(cls):
        return []


class InputDataComponentsSequence(InputDataComponents):

    def __init__(self, data: Sequence, *, meta=None):
        assert len(data) > 0, "Requires sequence of length > 0"
        n_dim = len(data[0])
        assert all(len(p) == n_dim for p in data), "Dimensionality inconsistent"

        self._data = data
        self._n_points = len(self._data)
        self._n_dim = n_dim

        _meta = {"access_components": True}
        if meta is not None:
            _meta.update(meta)
        self._meta = _meta

    @property
    def data(self):
        return self._data

    @property
    def n_points(self):
        return len(self._data)

    @property
    def n_dim(self) -> int:
        return len(self._data[0])

    def get_component(self, point: int, dimension: int) -> float:
        return self._data[point][dimension]

    def to_components_array(self) -> Type[np.ndarray]:
        return np.asarray(self._data)

    def get_subset(self, indices: Container) -> Type['InputData']:
        return type(self)([self._data[i] for i in indices])


cdef class InputDataExtComponentsMemoryview(InputDataExtInterface):
    """Implements the input data interface

    Stores point compenents as a 2D Cython typed memoryview.
    """

    def __cinit__(
            self, AVALUE[:, ::1] data not None, *, **kwargs):

        self._data = data
        self._n_points = self._data.shape[0]
        self._n_dim = self._data.shape[1]

    def __init__(self, data, *, meta=None):
        """
        Args:
            data: Raw input data. Needs to support the buffer protocol to be
                mapped to a AVALUE[:, ::1] typed memoryview.

        Keyword args:
            meta: Meta information dictionary.
        """

        _meta = {"access_components": True}
        if meta is not None:
            _meta.update(meta)
        self.meta = _meta

    @property
    def n_points(self):
        return self._data.shape[0]

    @property
    def n_dim(self):
        return self._data.shape[1]

    @property
    def data(self):
        return self._data

    def to_components_array(self):
        return np.asarray(self._data)

    cdef AVALUE _get_component(
            self, const AINDEX point, const AINDEX dimension) nogil:
        return self._data[point, dimension]

    def get_subset(
            self,
            object indices: Sequence) -> Type['InputDataExtComponentsMemoryview']:

            # np.array(self.to_components_array()[indices][:, indices], order="c")
            return type(self)(self.to_components_array()[indices])
            # Slow because it goes via numpy array

    def by_parts(self) -> Iterator:
        """Yield data by parts

        Returns:
            Generator of 2D :obj:`numpy.ndarray`s (parts)
        """

        assert self.n_points > 0

        edges = self.meta.get("edges", None)
        if not edges:
            edges = [self.n_points]

        data = self.data

        start = 0
        for end in edges:
            yield data[start:(start + end), :]
            start += end

    def __str__(self):
        return InputDataComponents.__str__(self)


class InputDataSklearnKDTree(InputDataComponents,InputDataNeighbourhoodsComputer):
    """Implements the input data interface

    Components stored as a NumPy array.  Neighbour queries delegated
    to pre-build KDTree.
    """

    def __init__(self, data: Type[np.ndarray], *, meta=None, **kwargs):
        self._data = data
        self._n_points = self._data.shape[0]
        self._n_dim = self._data.shape[1]

        self.build_tree(**kwargs)
        self.clear_cached()

        _meta = {
            "access_components": True,
            "access_neighbours": True
            }
        if meta is not None:
            _meta.update(meta)
        self._meta = _meta

    @property
    def n_points(self):
        return self._n_points

    @property
    def n_dim(self):
        return self._n_dim

    @property
    def data(self):
        return self.to_components_array()

    @property
    def n_neighbours(self):
        return self.to_n_neighbours_array()

    def to_components_array(self):
        return self._data

    def to_n_neighbours_array(self):
        return self._n_neighbours

    def get_component(self, point: int, dimension: int) -> float:
        return self._data[point, dimension]

    def get_n_neighbours(self, point: int) -> int:
        return self._n_neighbours[point]

    def get_neighbour(self, point: int, member: int) -> int:
        """Return a member for point"""
        return self._cached_neighbourhoods[point][member]

    def get_subset(self, indices: Container) -> Type['InputDataSklearnKDTree']:
        """Return input data subset"""

        return type(self)(self._data[indices])

    def build_tree(self, **kwargs):
        self._tree = sklearn.neighbors.KDTree(self._data, **kwargs)

    def compute_neighbourhoods(
            self,
            input_data: Type["InputData"],
            radius: float,
            is_sorted: bool = False,
            is_selfcounting: bool = True):

        self._cached_neighbourhoods = self._tree.query_radius(
            input_data.to_components_array(), r=radius,
            return_distance=False,
            )
        self._radius = radius

        if is_sorted:
            for n in self._cached_neighbourhoods:
                n.sort()

        if is_selfcounting is False:
            self._cached_neighbourhoods = np.array(
                [x[1:] for x in self._cached_neighbourhoods], dtype=object
            )

        self._n_neighbours = np.array([s.shape[0] for s in self._cached_neighbourhoods])

    def clear_cached(self):
        self._cached_neighbourhoods = None
        self._n_neighbours = None
        self._radius = None


cdef class InputDataExtDistancesMemoryview(InputDataExtInterface):
    """Implements the input data interface

    Stores inter-point distances as a 2D Cython typed memoryview.
    """

    def __cinit__(
            self, AVALUE[:, ::1] data not None, *, **kwargs):

        self._data = data
        self._n_points = self._data.shape[0]

    def __init__(self, data, *, meta=None):
        """
        Args:
            data: Raw input data. Needs to support the buffer protocol to be
                mapped to a AVALUE[:, ::1] typed memoryview.

        Keyword args:
            meta: Meta information dictionary.
        """

        _meta = {"access_distances": True}
        if meta is not None:
            _meta.update(meta)
        self.meta = _meta

    @property
    def n_points(self):
        return self._data.shape[0]

    @property
    def data(self):
        return self._data

    def to_distance_array(self):
        return np.asarray(self._data)

    cdef AVALUE _get_distance(
            self, const AINDEX point_a, const AINDEX point_b) nogil:
        return self._data[point_a, point_b]

    def get_subset(
            self,
            object indices: Sequence) -> Type['InputDataExtComponentsMemoryview']:

            return type(self)(np.array(self.to_distance_array()[indices][:, indices], order="c"))
            # Slow because it goes via numpy array

    def __str__(self):
        return InputDataPairwiseDistances.__str__(self)


cdef class InputDataExtDistancesLinearMemoryview(InputDataExtInterface):
    """Implements the input data interface

    Stores inter-point distances as 1D Cython typed memoryview
    """

    def __cinit__(self, AVALUE[::1] data not None, *, n_points=None, meta=None):
        self._data = data
        if n_points is None:
            n_points = int(0.5 * (csqrt(8 * self._data.shape[0] + 1) + 1))
        self._n_points = n_points

        _meta = {"access_distances": True}
        if meta is not None:
            _meta.update(meta)
        self.meta = _meta

    cdef AVALUE _get_distance(self, const AINDEX point_a, const AINDEX point_b) nogil:

        cdef AINDEX i, j, a, b

        if point_a == point_b:
            return 0

        if point_a > point_b:
            b = point_a
            a = point_b
        else:
            a = point_a
            b = point_b

        # Start of block d(a)
        i = a * (self._n_points - 1) - (a**2 - a) / 2
        j = b - a - 1  # Pos. within d(a) block

        return self._data[i + j]

    @property
    def n_points(self):
        return self._n_points

    @property
    def data(self):
        return self._data

    def to_distance_array(self):
        cdef AINDEX i, j
        cdef AVALUE d

        cdef AVALUE[:, ::1] distance_array = np.zeros((self._n_points, self._n_points), dtype=P_AVALUE)

        for i in range(0, self._n_points):
            for j in range(i + 1, self._n_points):
                d = self._get_distance(i, j)
                distance_array[i, j] = distance_array[j, i] = d

        return np.asarray(distance_array)

    def get_subset(
                self,
                object indices: Sequence) -> Type['InputDataExtComponentsMemoryview']:

            distance_subset = np.array(self.to_distance_array()[indices][:, indices], order="c")

            return type(self)(distance_subset[np.triu_indices_from(distance_subset, k=1)])
            # Slow because it goes via numpy array

    def __str__(self):
        return InputDataPairwiseDistances.__str__(self)


cdef class InputDataExtNeighbourhoodsMemoryview(InputDataExtInterface):
    """Implements the input data interface

    Neighbours of points stored using a Cython memoryview.
    """

    def __cinit__(
            self,
            AINDEX[:, ::1] data not None,
            AINDEX[::1] n_neighbours not None, *, meta=None):

        self._data = data
        self._n_points = self._data.shape[0]
        self._n_neighbours = n_neighbours

        _meta = {"access_neighbours": True}
        if meta is not None:
            _meta.update(meta)
        self.meta = _meta

    @property
    def n_points(self):
        return self._data.shape[0]

    @property
    def data(self):
        return self._data

    def to_neighbourhoods_array(self):
        cdef AINDEX i

        return np.array([
            s[:self._n_neighbours[i]]
            for i, s in enumerate(np.asarray(self._data))
            ], dtype=object)

    @property
    def n_neighbours(self):
        return self._n_neighbours

    def to_n_neighbours_array(self):
        return np.asarray(self._n_neighbours)

    cdef AINDEX _get_n_neighbours(self, const AINDEX point) nogil:
        return self._n_neighbours[point]

    cdef AINDEX _get_neighbour(self, const AINDEX point, const AINDEX member) nogil:
        """Return a member for point"""
        return self._data[point, member]

    def get_subset(self, indices: Sequence) -> Type['InputDataExtNeighbourhoodsMemoryview']:
        cdef AINDEX[::1] lengths
        cdef AINDEX i

        data_subset = np.asarray(self._data)[indices]
        data_subset = [
            [m for m in a if m in indices]
            for a in data_subset
        ]

        lengths = np.array([len(a) for a in data_subset])
        pad_to = np.asarray(lengths).max()

        for i, a in enumerate(data_subset):
            missing_elements = pad_to - lengths[i]
            a.extend([0] * missing_elements)

        return type(self)(np.asarray(data_subset, order="C", dtype=P_AINDEX), lengths)

    def __str__(self):
        return InputDataNeighbourhoods.__str__(self)


cdef class InputDataExtNeighbourhoodsVector(InputDataExtInterface):
    """Implements the input data interface

    Neighbours of points are stored using a C++ std::vector of vectors.
    """

    def __cinit__(
            self,
            data, *, meta=None):
        self._data = data
        self._n_points = self._data.size()
        self._n_neighbours = [len(x) for x in data]

        _meta = {"access_neighbours": True}
        if meta is not None:
            _meta.update(meta)
        self.meta = _meta

    @property
    def n_points(self):
        return self._data.size()

    @property
    def data(self):
        cdef AINDEX i, j
        cdef list as_list = []

        for i in range(self.n_points):
            as_list.append([])
            for j in range(self._n_neighbours[i]):
                as_list[i].append(<Py_ssize_t>self._data[i][j])
        return as_list

    def to_neighbourhoods_array(self):
        cdef AINDEX i, j
        cdef list as_list = []

        for i in range(self.n_points):
            as_list.append(np.empty(self._n_neighbours[i], dtype=P_AINDEX))
            for j in range(self._n_neighbours[i]):
                as_list[i][j] = self._data[i][j]
        return np.asarray(as_list, dtype=object)

    @property
    def n_neighbours(self):
        cdef AINDEX i
        cdef list as_list = []

        for i in range(self.n_points):
            as_list.append(<Py_ssize_t>self._n_neighbours[i])
        return as_list

    def to_n_neighbours_array(self):
        cdef AINDEX i
        cdef AINDEX[::1] as_array = np.empty(self.n_points, dtype=P_AINDEX)

        for i in range(self.n_points):
            as_array[i] = self._n_neighbours[i]

        return np.asarray(as_array)

    cdef AINDEX _get_n_neighbours(self, const AINDEX point) nogil:
        return self._n_neighbours[point]

    def get_n_neighbours(self, point: int) -> int:
        return self._get_n_neighbours(point)

    cdef AINDEX _get_neighbour(self, const AINDEX point, const AINDEX member) nogil:
        """Return a member for point"""
        return self._data[point][member]

    def get_neighbour(self, point: int, member: int) -> int:
        return self._get_neighbour(point, member)

    def get_subset(self, indices: Sequence) -> Type['InputDataExtNeighbourhoodsVector']:
        """Return input data subset"""
        cdef list lengths

        data_subset = [x for i, x in enumerate(self.data) if i in indices]
        data_subset = [
            [m for m in a if m in indices]
            for a in data_subset
        ]
        return type(self)(data_subset)

    def __str__(self):
        return InputDataNeighbourhoods.__str__(self)


class InputDataNeighbourhoodsSequence(InputDataNeighbourhoods):
    """Implements the input data interface

    Neighbours of points stored as a sequence.
    """

    def __init__(self, data: Sequence, *, meta=None):
        """

        Args:
            data: Any sequence of neighbour index sequences (needs to be
                sized, indexable, and iterable)

        Keyword args:
            meta: Meta-information dictionary.
        """

        self._data = data
        self._n_points = len(self._data)
        self._n_neighbours = [len(s) for s in self._data]

        _meta = {"access_neighbours": True}
        if meta is not None:
            _meta.update(meta)
        self._meta = _meta

    @property
    def data(self):
        return self._data

    @property
    def n_points(self):
        return len(self._data)

    @property
    def n_neighbours(self):
        return self._n_neighbours

    def to_n_neighbours_array(self):
        return np.asarray(self._n_neighbours)

    def to_neighbourhoods_array(self):
        cdef AINDEX i

        return np.array([np.asarray(s) for s in self._data], dtype=object)

    def get_n_neighbours(self, point: int) -> int:
        return self._n_neighbours[point]

    def get_neighbour(self, point: int, member: int) -> int:
        return self._data[point][member]

    def get_subset(self, indices: Container) -> Type['InputDataNeighbourhoodsSequence']:
        data_subset = [
            [m for m in s if m in indices]
            for i, s in enumerate(self._data)
            if i in indices
        ]

        return type(self)(data_subset)


InputDataComponents.register(InputDataExtComponentsMemoryview)
InputDataComponents.register(InputDataExtDistancesMemoryview)
InputDataPairwiseDistances.register(InputDataExtDistancesLinearMemoryview)
InputDataNeighbourhoods.register(InputDataExtNeighbourhoodsMemoryview)
InputDataNeighbourhoods.register(InputDataExtNeighbourhoodsVector)


class NeighboursList(Neighbours):
    """Implements the neighbours interface"""

    def __init__(self, neighbours=None):
        if neighbours is not None:
            self._neighbours = neighbours
            self._n_points = len(self._neighbours)
        else:
            self.reset()

    @property
    def n_points(self):
        return self._n_points

    @property
    def neighbours(self):
        return self._neighbours

    def to_neighbours_array(self):
        return np.asarray(self._neighbours)

    def assign(self, member: int):
        self._neighbours.append(member)
        self._n_points += 1

    def reset(self):
        self._neighbours = []
        self._n_points = 0

    def enough(self, member_cutoff: int) -> bool:
        if self._n_points > member_cutoff:
            return True
        return False

    def get_member(self, index: int) -> int:
        return self._neighbours[index]

    def contains(self, member: int) -> bool:
        if member in self._neighbours:
            return True
        return False


class NeighboursSet(Neighbours):
    """Implements the neighbours interface"""

    def __init__(self, neighbours=None):
        if neighbours is not None:
            self._neighbours = neighbours
            self._n_points = len(self._neighbours)
            self._query = 0
            self._iter = None
        else:
            self.reset()

    @property
    def n_points(self):
        return self._n_points

    @property
    def neighbours(self):
        return self._neighbours

    def to_neighbours_array(self):
        return np.asarray(list(self._neighbours))

    def assign(self, member: int):
        self._neighbours.add(member)
        self._n_points += 1

    def reset(self):
        self._neighbours = set()
        self._n_points = 0
        self._query = 0
        self._iter = None

    def enough(self, member_cutoff: int):
        if self._n_points > member_cutoff:
            return True
        return False

    def get_member(self, index: int) -> int:
        if (self._iter is None) or (index != self._query):
            self._iter = iter(self._neighbours)
            self._query = 0

        while self._query != index:
            _ = next(self._iter)
            self._query += 1

        self._query += 1
        return next(self._iter)

    def contains(self, member: int) -> bool:
        if member in self._neighbours:
            return True
        return False


cdef class NeighboursExtVector(NeighboursExtInterface):
    """Implements the neighbours interface

    Uses an underlying C++ std:vector.

    Keyword args:
        neighbours: A sequence of labels suitable to be cast to a vector.
        initial_size: Number of elements reserved for the size of vector.
    """

    def __cinit__(self, neighbours=None, *, AINDEX initial_size=1):
        self._initial_size = initial_size

        if neighbours is not None:
            self._neighbours = neighbours
            self._n_points = self._neighbours.size()
            self._neighbours.reserve(self._initial_size)
        else:
            self._reset()

    cdef void _assign(self, const AINDEX member) nogil:
        self._neighbours.push_back(member)
        self._n_points += 1

    cdef void _reset(self) nogil:
        self._neighbours.resize(0)
        self._neighbours.reserve(self._initial_size)
        self._n_points = 0

    cdef bint _enough(self, const AINDEX member_cutoff) nogil:
        if self._n_points > member_cutoff:
            return True
        return False

    cdef AINDEX _get_member(self, const AINDEX index) nogil:
        return self._neighbours[index]

    cdef bint _contains(self, const AINDEX member) nogil:

        if find(self._neighbours.begin(), self._neighbours.end(), member) == self._neighbours.end():
            return False
        return True

    def to_neighbours_array(self):
        cdef AINDEX i
        cdef AINDEX[::1] a = np.empty(self._n_points, dtype=P_AINDEX)

        for i in range(self._n_points):
            a[i] = self._neighbours[i]

        return np.asarray(a)

    @property
    def n_points(self):
        return self._neighbours.size()


cdef class NeighboursExtSet(NeighboursExtInterface):
    """Implements the neighbours interface

    Uses an underlying C++ std:set.

    Keyword args:
        neighbours: A sequence of labels suitable to be cast to a C++ set.
    """

    def __cinit__(self, neighbours=None):

        if neighbours is not None:
            self._neighbours = neighbours
            self._n_points = self._neighbours.size()
        else:
            self._reset()

    cdef void _assign(self, const AINDEX member) nogil:
        self._neighbours.insert(member)
        self._n_points += 1

    cdef void _reset(self) nogil:
        self._neighbours.clear()
        self._n_points = 0

    cdef bint _enough(self, const AINDEX member_cutoff) nogil:
        if self._n_points > member_cutoff:
            return True
        return False

    cdef AINDEX _get_member(self, const AINDEX index) nogil:
        cdef stdset[AINDEX].iterator it = self._neighbours.begin()
        cdef AINDEX i

        for i in range(index):
            preincrement(it)

        return dereference(it)

    cdef bint _contains(self, const AINDEX member) nogil:
        if self._neighbours.find(member) == self._neighbours.end():
            return False
        return True

    def to_neighbours_array(self):
        cdef stdset[AINDEX].iterator it = self._neighbours.begin()
        cdef AINDEX[::1] a = np.empty(self._n_points, dtype=P_AINDEX)
        cdef AINDEX i = 0

        while it != self._neighbours.end():
            a[i] = dereference(it)
            preincrement(it)
            i += 1

        return np.asarray(a)

    @property
    def n_points(self):
        return self._neighbours.size()


cdef class NeighboursExtUnorderedSet(NeighboursExtInterface):
    """Implements the neighbours interface

    Uses an underlying C++ std:unordered_set.

    Keyword args:
        neighbours: A sequence of labels suitable to be cast to a C++ set.
    """

    def __cinit__(self, neighbours=None):

        if neighbours is not None:
            self._neighbours = neighbours
            self._n_points = self._neighbours.size()
        else:
            self._reset()

    cdef void _assign(self, const AINDEX member) nogil:
        self._neighbours.insert(member)
        self._n_points += 1

    cdef void _reset(self) nogil:
        self._neighbours.clear()
        self._n_points = 0

    cdef bint _enough(self, const AINDEX member_cutoff) nogil:
        if self._n_points > member_cutoff:
            return True
        return False

    cdef AINDEX _get_member(self, const AINDEX index) nogil:
        cdef stduset[AINDEX].iterator it = self._neighbours.begin()
        cdef AINDEX i

        for i in range(index):
            preincrement(it)

        return dereference(it)

    cdef bint _contains(self, const AINDEX member) nogil:
        if self._neighbours.find(member) == self._neighbours.end():
            return False
        return True

    def to_neighbours_array(self):
        cdef stduset[AINDEX].iterator it = self._neighbours.begin()
        cdef AINDEX[::1] a = np.empty(self._n_points, dtype=P_AINDEX)
        cdef AINDEX i = 0

        while it != self._neighbours.end():
            a[i] = dereference(it)
            preincrement(it)
            i += 1

        return np.asarray(a)

    @property
    def n_points(self):
        return self._neighbours.size()


cdef class NeighboursExtVectorUnorderedSet(NeighboursExtInterface):
    """Implements the neighbours interface

    Uses a compination of an underlying C++ std:vector and a std:unordered_set.

    Keyword args:
        neighbours: A sequence of labels suitable to be cast to a C++ vector.
    """

    def __cinit__(self, neighbours=None, *, initial_size=1):
        cdef AINDEX member

        self._initial_size = initial_size

        if neighbours is not None:
            self._neighbours = neighbours
            self._n_points = self._neighbours.size()
            self._neighbours.reserve(self._initial_size)

            for member in self._neighbours:
                self._neighbours_view.insert(member)
        else:
            self._reset()

    cdef void _assign(self, const AINDEX member) nogil:
        self._neighbours.push_back(member)
        self._neighbours_view.insert(member)
        self._n_points += 1

    cdef void _reset(self) nogil:
        self._neighbours.resize(0)
        self._neighbours.reserve(self._initial_size)
        self._neighbours_view.clear()
        self._n_points = 0

    cdef bint _enough(self, const AINDEX member_cutoff) nogil:
        if self._n_points > member_cutoff:
            return True
        return False

    cdef AINDEX _get_member(self, const AINDEX index) nogil:
        return self._neighbours[index]

    cdef bint _contains(self, const AINDEX member) nogil:
        if self._neighbours_view.find(member) == self._neighbours_view.end():
            return False
        return True

    def to_neighbours_array(self):
        cdef AINDEX i
        cdef AINDEX[::1] a = np.empty(self._n_points, dtype=P_AINDEX)

        for i in range(self._n_points):
            a[i] = self._neighbours[i]

        return np.asarray(a)

    @property
    def n_points(self):
        return self._neighbours.size()


Neighbours.register(NeighboursExtVector)
Neighbours.register(NeighboursExtSet)
Neighbours.register(NeighboursExtUnorderedSet)
Neighbours.register(NeighboursExtVectorUnorderedSet)


class NeighboursGetterBruteForce(NeighboursGetter):
    """Implements the neighbours getter interface"""

    def __init__(
            self,
            distance_getter: Type["DistanceGetter"]):
        self._is_sorted = False
        self._is_selfcounting = True
        self._distance_getter = distance_getter

    def __str__(self):
        attr_str = ",".join([
            f"    dgetter={self._distance_getter}",
            f"\n    sorted={self._is_sorted}",
            f"\n    selfcounting={self._is_selfcounting}",
        ])

        return f"{type(self).__name__}(\n{attr_str}\n)"

    @classmethod
    def get_builder_kwargs(cls):
        return [("distance_getter", None)]

    @property
    def is_sorted(self) -> bool:
        return self._is_sorted

    @property
    def is_selfcounting(self) -> bool:
        return self._is_selfcounting

    def get(
            self,
            index: int,
            input_data: Type['InputData'],
            neighbours: Type['Neighbours'],
            cluster_params: Type['ClusterParameters']):

        cdef AINDEX i
        cdef AVALUE distance

        neighbours.reset()

        for i in range(input_data._n_points):
            distance = self._distance_getter.get_single(
                index, i, input_data
                )

            if distance <= cluster_params.get_fparam(0):
                neighbours.assign(i)

    def get_other(
            self,
            index: int,
            input_data: Type['InputData'],
            other_input_data: Type['InputData'],
            neighbours: Type['Neighbours'],
            cluster_params: Type['ClusterParameters']):

        cdef AINDEX i
        cdef AVALUE distance

        neighbours.reset()

        for i in range(input_data._n_points):
            distance = self._distance_getter.get_single_other(
                index, i, input_data, other_input_data
                )

            if distance <= cluster_params.get_fparam(0):
                neighbours.assign(i)


cdef class NeighboursGetterExtBruteForce(NeighboursGetterExtInterface):
    """Implements the neighbours getter interface
    This getter retrieves the neighbours of a point by comparing the
    distances (from a distance getter) between the point and all
    other points to the radius cutoff (:math:`r_{ij} \leq r`).
    The resulting neighbour containers are in general not sorted and
    include points as their own neighbour (self counting).
    Args:
        distance_getter: An object implementing the distance getter
            interface. Has to be a Cython extension type.
    """

    def __cinit__(
            self,
            DistanceGetterExtInterface distance_getter):
        self.is_sorted = False
        self.is_selfcounting = True
        self._distance_getter = distance_getter

    def __init__(self, distance_getter: Type["DistanceGetterExtInterface"]):
        pass

    def __str__(self):
        attr_str = ",".join([
            f"    dgetter={self._distance_getter}",
            f"\n    sorted={self.is_sorted}",
            f"\n    selfcounting={self.is_selfcounting}",
        ])

        return f"{type(self).__name__}(\n{attr_str}\n)"

    cdef void _get(
            self,
            const AINDEX index,
            InputDataExtInterface input_data,
            NeighboursExtInterface neighbours,
            ClusterParameters cluster_params) nogil:

        cdef AINDEX i
        cdef AVALUE distance

        neighbours._reset()

        for i in range(input_data._n_points):
            distance = self._distance_getter._get_single(
                index, i, input_data
                )

            if distance <= cluster_params.fparams[0]:
                neighbours._assign(i)

    cdef void _get_other(
            self,
            const AINDEX index,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data,
            NeighboursExtInterface neighbours,
            ClusterParameters cluster_params) nogil:

        cdef AINDEX i
        cdef AVALUE distance

        neighbours._reset()

        for i in range(input_data._n_points):
            distance = self._distance_getter._get_single_other(
                index, i, input_data, other_input_data
                )

            if distance <= cluster_params.fparams[0]:
                neighbours._assign(i)

    @classmethod
    def get_builder_kwargs(cls):
        return [("distance_getter", None)]


class NeighboursGetterLookup(NeighboursGetter):
    """Implements the neighbours getter interface"""

    def __init__(self, is_sorted=False, is_selfcounting=False):
        self._is_sorted = is_sorted
        self._is_selfcounting = is_selfcounting

    def __str__(self):
        attr_str = ", ".join([
            f"sorted={self._is_sorted}",
            f"selfcounting={self._is_selfcounting}",
        ])

        return f"{type(self).__name__}({attr_str})"

    @property
    def is_sorted(self) -> bool:
        return self._is_sorted

    @property
    def is_selfcounting(self) -> bool:
        return self._is_selfcounting

    def get(
            self,
            index: int,
            input_data: Type['InputData'],
            neighbours: Type['Neighbours'],
            cluster_params: Type['ClusterParameters']) -> None:

        cdef AINDEX i

        neighbours.reset()

        for i in range(input_data.get_n_neighbours(index)):
            neighbours.assign(input_data.get_neighbour(index, i))

    def get_other(
            self,
            index: int,
            input_data: Type['InputData'],
            other_input_data: Type['InputData'],
            neighbours: Type['Neighbours'],
            cluster_params: Type['ClusterParameters']):

        cdef AINDEX i

        neighbours.reset()

        for i in range(other_input_data.get_n_neighbours(index)):
            neighbours.assign(other_input_data.get_neighbour(index, i))


cdef class NeighboursGetterExtLookup(NeighboursGetterExtInterface):
    """Implements the neighbours getter interface"""

    def __cinit__(self, is_sorted=False, is_selfcounting=True):
        self.is_sorted = is_sorted
        self.is_selfcounting = is_selfcounting

    def __str__(self):
        attr_str = ", ".join([
            f"sorted={self.is_sorted}",
            f"selfcounting={self.is_selfcounting}",
        ])

        return f"{type(self).__name__}({attr_str})"

    cdef void _get(
            self,
            const AINDEX index,
            InputDataExtInterface input_data,
            NeighboursExtInterface neighbours,
            ClusterParameters cluster_params) nogil:

        cdef AINDEX i
        neighbours._reset()

        for i in range(input_data._get_n_neighbours(index)):
            neighbours._assign(input_data._get_neighbour(index, i))

    cdef void _get_other(
            self,
            const AINDEX index,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data,
            NeighboursExtInterface neighbours,
            ClusterParameters cluster_params) nogil:

        cdef AINDEX i

        neighbours._reset()

        for i in range(other_input_data._get_n_neighbours(index)):
            neighbours._assign(other_input_data._get_neighbour(index, i))


class NeighboursGetterRecomputeLookup(NeighboursGetter):
    """Implements the neighbours getter interface"""

    def __init__(self, is_sorted=False, is_selfcounting=True):
        self._is_sorted = is_sorted
        self._is_selfcounting = is_selfcounting

    def __str__(self):
        attr_str = ", ".join([
            f"sorted={self._is_sorted}",
            f"selfcounting={self._is_selfcounting}",
        ])

        return f"{type(self).__name__}({attr_str})"

    @property
    def is_sorted(self) -> bool:
        return self._is_sorted

    @property
    def is_selfcounting(self) -> bool:
        return self._is_selfcounting

    def get(
            self,
            index: int,
            input_data: Type['InputData'],
            neighbours: Type['Neighbours'],
            cluster_params: Type['ClusterParameters']) -> None:

        cdef AINDEX i

        if input_data._radius != cluster_params.get_fparam(0):
            input_data.compute_neighbourhoods(
                input_data,
                cluster_params.get_fparam(0),
                self._is_sorted,
                self._is_selfcounting
                )

        neighbours.reset()

        for i in range(input_data.get_n_neighbours(index)):
            neighbours.assign(input_data.get_neighbour(index, i))

    def get_other(
            self,
            index: int,
            input_data: Type['InputData'],
            other_input_data: Type['InputData'],
            neighbours: Type['Neighbours'],
            cluster_params: Type['ClusterParameters']):

        cdef AINDEX i

        if other_input_data._radius != cluster_params.get_fparam(0):
            other_input_data.compute_neighbourhoods(
                input_data,
                cluster_params.get_fparam(0),
                self._is_sorted,
                self._is_selfcounting
                )

        neighbours.reset()

        for i in range(other_input_data.get_n_neighbours(index)):
            neighbours.assign(other_input_data.get_neighbour(index, i))


NeighboursGetter.register(NeighboursGetterExtBruteForce)
NeighboursGetter.register(NeighboursGetterExtLookup)


class DistanceGetterMetric(DistanceGetter):
    """Implements the distance getter interface"""

    def __init__(self, metric: Type["Metric"]):
        self._metric = metric

    def __str__(self):
        attr_str = ", ".join([
            f"metric={self._metric}",
        ])

        return f"{type(self).__name__}({attr_str})"

    @classmethod
    def get_builder_kwargs(cls):
        return [("metric", None)]

    def  get_single(
            self,
            point_a: int,
            point_b: int,
            input_data: Type["InputData"]):

        return self._metric.calc_distance(point_a, point_b, input_data)

    def get_single_other(
            self,
            point_a: int,
            point_b: int,
            input_data: Type["InputData"],
            other_input_data: Type["InputData"]):

        return self._metric.calc_distance_other(
            point_a, point_b, input_data, other_input_data
            )


class DistanceGetterLookup(DistanceGetter):
    """Implements the distance getter interface"""

    def get_single(
            self,
            const AINDEX point_a,
            const AINDEX point_b,
            InputDataExtInterface input_data):

        return input_data.get_distance(point_a, point_b)

    def get_single_other(
            self,
            const AINDEX point_a,
            const AINDEX point_b,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data):

        return other_input_data.get_distance(point_a, point_b)


cdef class DistanceGetterExtMetric(DistanceGetterExtInterface):
    """Implements the distance getter interface"""

    def __cinit__(self, MetricExtInterface metric):
        self._metric = metric

    def __str__(self):
        attr_str = ", ".join([
            f"metric={self._metric}",
        ])

        return f"{type(self).__name__}({attr_str})"

    cdef AVALUE _get_single(
            self,
            const AINDEX point_a,
            const AINDEX point_b,
            InputDataExtInterface input_data) nogil:

        return self._metric._calc_distance(point_a, point_b, input_data)

    def get_single(
            self,
            AINDEX point_a,
            AINDEX point_b,
            InputDataExtInterface input_data):

        return self._get_single(point_a, point_b, input_data)

    cdef AVALUE _get_single_other(
            self,
            const AINDEX point_a,
            const AINDEX point_b,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data) nogil:

        return self._metric._calc_distance_other(
            point_a, point_b, input_data, other_input_data
            )

    def get_single_other(
            self,
            AINDEX point_a,
            AINDEX point_b,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data):

        return self._get_single_other(point_a, point_b, input_data, other_input_data)

    @classmethod
    def get_builder_kwargs(cls):
        return [("metric", None)]


cdef class DistanceGetterExtLookup(DistanceGetterExtInterface):
    """Implements the distance getter interface"""

    cdef AVALUE _get_single(
            self,
            const AINDEX point_a,
            const AINDEX point_b,
            InputDataExtInterface input_data) nogil:

        return input_data._get_distance(point_a, point_b)

    cdef AVALUE _get_single_other(
            self,
            const AINDEX point_a,
            const AINDEX point_b,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data) nogil:

        return other_input_data._get_distance(point_a, point_b)


DistanceGetter.register(DistanceGetterExtMetric)
DistanceGetter.register(DistanceGetterExtLookup)


class MetricDummy(Metric):
    """Implements the metric interface"""

    def calc_distance(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData']) -> float:
        return 0.

    def calc_distance_other(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData'],
            other_input_data: Type['InputData']) -> float:
        return 0.

    def adjust_radius(self, radius_cutoff: float) -> float:
        return radius_cutoff


cdef class MetricExtDummy(MetricExtInterface):
    """Implements the metric interface"""

    cdef AVALUE _calc_distance(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data) nogil:

        return 0.

    cdef AVALUE _calc_distance_other(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data) nogil:

        return 0.

    cdef AVALUE _adjust_radius(self, AVALUE radius_cutoff) nogil:
        return radius_cutoff


class MetricPrecomputed(Metric):
    """Implements the metric interface"""

    def calc_distance(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData']) -> float:

        return input_data.get_component(index_a, index_b)

    def calc_distance_other(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData'],
            other_input_data: Type['InputData']) -> float:

        return other_input_data.get_component(index_a, index_b)

    def adjust_radius(self, radius_cutoff: float) -> float:
        return radius_cutoff


cdef class MetricExtPrecomputed(MetricExtInterface):
    """Implements the metric interface"""

    cdef AVALUE _calc_distance(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data) nogil:

        return input_data._get_component(index_a, index_b)

    cdef AVALUE _calc_distance_other(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data) nogil:

        return other_input_data._get_component(index_a, index_b)

    cdef AVALUE _adjust_radius(self, AVALUE radius_cutoff) nogil:
        return radius_cutoff


class MetricEuclidean(Metric):
    """Implements the metric interface"""

    def calc_distance(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData']) -> float:

        cdef AVALUE total = 0
        cdef AINDEX i, n_dim = input_data._n_dim
        cdef AVALUE a, b

        for i in range(n_dim):
            a = input_data.get_component(index_a, i)
            b = input_data.get_component(index_b, i)
            total += cpow(a - b, 2)

        return csqrt(total)

    def calc_distance_other(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData'],
            other_input_data: Type['InputData']) -> float:

        cdef AVALUE total = 0
        cdef AINDEX i, n_dim = input_data._n_dim
        cdef AVALUE a, b

        for i in range(n_dim):
            a = other_input_data.get_component(index_a, i)
            b = input_data.get_component(index_b, i)
            total += cpow(a - b, 2)

        return csqrt(total)

    def adjust_radius(self, radius_cutoff: float) -> float:
        return radius_cutoff


cdef class MetricExtEuclidean(MetricExtInterface):
    """Implements the metric interface"""

    cdef AVALUE _calc_distance(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data) nogil:

        cdef AVALUE total = 0
        cdef AINDEX i, n_dim = input_data._n_dim
        cdef AVALUE a, b

        for i in range(n_dim):
            a = input_data._get_component(index_a, i)
            b = input_data._get_component(index_b, i)
            total += cpow(a - b, 2)

        return csqrt(total)

    cdef AVALUE _calc_distance_other(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data) nogil:

        cdef AVALUE total = 0
        cdef AINDEX i, n_dim = input_data._n_dim
        cdef AVALUE a, b

        for i in range(n_dim):
            a = other_input_data._get_component(index_a, i)
            b = input_data._get_component(index_b, i)
            total += cpow(a - b, 2)

        return csqrt(total)

    cdef AVALUE _adjust_radius(self, AVALUE radius_cutoff) nogil:
        return radius_cutoff


class MetricEuclideanReduced(Metric):
    """Implements the metric interface"""

    def calc_distance(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData']) -> float:

        cdef AVALUE total = 0
        cdef AINDEX i, n_dim = input_data._n_dim
        cdef AVALUE a, b

        for i in range(n_dim):
            a = input_data.get_component(index_a, i)
            b = input_data.get_component(index_b, i)
            total += cpow(a - b, 2)

        return total

    def calc_distance_other(
            self,
            index_a: int, index_b: int,
            input_data: Type['InputData'],
            other_input_data: Type['InputData']) -> float:

        cdef AVALUE total = 0
        cdef AINDEX i, n_dim = input_data._n_dim
        cdef AVALUE a, b

        for i in range(n_dim):
            a = other_input_data.get_component(index_a, i)
            b = input_data.get_component(index_b, i)
            total += cpow(a - b, 2)

        return total

    def adjust_radius(self, radius_cutoff: float) -> float:
        return radius_cutoff**2


cdef class MetricExtEuclideanReduced:
    """Implements the metric interface"""

    cdef AVALUE _calc_distance(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data) nogil:

        cdef AVALUE total = 0
        cdef AINDEX i, n_dim = input_data._n_dim
        cdef AVALUE a, b

        for i in range(n_dim):
            a = input_data._get_component(index_a, i)
            b = input_data._get_component(index_b, i)
            total += cpow(a - b, 2)

        return total

    cdef AVALUE _calc_distance_other(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data) nogil:

        cdef AVALUE total = 0
        cdef AINDEX i, n_dim = input_data._n_dim
        cdef AVALUE a, b

        for i in range(n_dim):
            a = other_input_data._get_component(index_a, i)
            b = input_data._get_component(index_b, i)
            total += cpow(a - b, 2)

        return total

    cdef AVALUE _adjust_radius(self, AVALUE radius_cutoff) nogil:
        return cpow(radius_cutoff, 2)


cdef class MetricExtEuclideanPeriodicReduced:
    """Implements the metric interface"""

    def __cinit__(self, bounds):
        self._bounds = bounds

    cdef AVALUE _calc_distance(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data) nogil:

        cdef AVALUE total = 0
        cdef AINDEX i, n_dim = input_data._n_dim
        cdef AVALUE a, b, distance, bound

        for i in range(n_dim):
            a = input_data._get_component(index_a, i)
            b = input_data._get_component(index_b, i)

            bound = self._bounds[i]
            distance = cfabs(a - b)

            if bound > 0:
                distance = distance % bound
                if distance > (bound / 2):
                    distance = bound - distance

            total +=  cpow(distance, 2)

        return total

    cdef AVALUE _calc_distance_other(
            self,
            const AINDEX index_a, const AINDEX index_b,
            InputDataExtInterface input_data,
            InputDataExtInterface other_input_data) nogil:

        cdef AVALUE total = 0
        cdef AINDEX i, n_dim = input_data._n_dim
        cdef AVALUE a, b, distance, bound

        for i in range(n_dim):
            a = other_input_data._get_component(index_a, i)
            b = input_data._get_component(index_b, i)

            bound = self._bounds[i]
            distance = cfabs(a - b)

            if bound > 0:
                distance = distance % bound
                if distance > (bound / 2):
                    distance = bound - distance

            total += cpow(distance, 2)

        return total

    cdef AVALUE _adjust_radius(self, AVALUE radius_cutoff) nogil:
        return cpow(radius_cutoff, 2)


Metric.register(MetricExtDummy)
Metric.register(MetricExtPrecomputed)
Metric.register(MetricExtEuclidean)
Metric.register(MetricExtEuclideanReduced)
Metric.register(MetricExtEuclideanPeriodicReduced)


class SimilarityCheckerContains(SimilarityChecker):
    r"""Implements the similarity checker interface
    Strategy:
        Loops over members of one neighbours container and checks
        if they are contained in the other neighbours container.  Breaks
        early when similarity criterion is reached.
        The performance and time-complexity of the check depends on the
        used neighbour containers.  Worst case time
        complexity is :math:`\mathcal{O}(n * m)` with :math:`n` and
        :math:`m` being the lengths of the neighbours containers if the
        containment check is performed by iteration.  Worst
        case time complexity is :math:`\mathcal{O}(n)` if containment
        check can be performed as lookup in linear time.  Note that
        no switching of the neighbours containers is done to ensure
        that the first container is the one with the shorter length
        (compare
        :obj:`commonnn._types.SimilarityCheckerSwitchContains`).
    """

    def check(
            self,
            neighbours_a: Type["Neighbours"],
            neighbours_b: Type["Neighbours"],
            cluster_params: Type['ClusterParameters']) -> bool:

        cdef AINDEX na = neighbours_a._n_points

        cdef AINDEX c = cluster_params.get_iparam(0)
        cdef AINDEX member_a, member_index_a
        cdef AINDEX common = 0

        if c == 0:
            return True

        for member_index_a in range(na):
            member_a = neighbours_a.get_member(member_index_a)
            if neighbours_b.contains(member_a):
                common += 1
                if common == c:
                    return True
                continue
        return False

    def get(
            self,
            neighbours_a: Type["Neighbours"],
            neighbours_b: Type["Neighbours"],
            cluster_params: Type['ClusterParameters']) -> int:
        """Return number of common neighbours"""

        cdef AINDEX na = neighbours_a._n_points

        cdef AINDEX member_a, member_index_a
        cdef AINDEX common = 0

        for member_index_a in range(na):
            member_a = neighbours_a.get_member(member_index_a)
            if neighbours_b.contains(member_a):
                common += 1

        return common


class SimilarityCheckerSwitchContains(SimilarityChecker):
    r"""Implements the similarity checker interface
    Strategy:
        Loops over members of one neighbours container and checks
        if they are contained in the other neighbours container.  Breaks
        early when similarity criterion is reached.  The performance
        and time-complexity of the check depends on the
        used neighbour containers.  Worst case time
        complexity is :math:`\mathcal{O}(n * m)` with :math:`n` and
        :math:`m` being the lengths of the neighbours containers if the
        containment check is performed by iteration.  Worst
        case time complexity is :math:`\mathcal{O}(n)` if containment
        check can be performed as lookup in linear time.  Note that a
        switching of the neighbours containers is done to ensure
        that the first container is the one with the shorter length
        (compare
        :obj:`commonnn._types.SimilarityCheckerContains`).
    """

    def check(
            self,
            neighbours_a: Type["Neighbours"],
            neighbours_b: Type["Neighbours"],
            cluster_params: Type['ClusterParameters']) -> bool:

        cdef AINDEX na = neighbours_a._n_points
        cdef AINDEX nb = neighbours_b._n_points

        cdef AINDEX c = cluster_params.get_iparam(0)
        cdef AINDEX member_a, member_index_a
        cdef AINDEX common = 0

        if c == 0:
            return True

        if nb < na:
            neighbours_a, neighbours_b = neighbours_b, neighbours_a
            na, nb = nb, na

        for member_index_a in range(na):
            member_a = neighbours_a.get_member(member_index_a)
            if neighbours_b.contains(member_a):
                common += 1
                if common == c:
                    return True
                continue
        return False

    def get(
            self,
            neighbours_a: Type["Neighbours"],
            neighbours_b: Type["Neighbours"],
            cluster_params: Type['ClusterParameters']) -> int:
        """Return number of common neighbours"""

        cdef AINDEX na = neighbours_a._n_points
        cdef AINDEX nb = neighbours_b._n_points

        cdef AINDEX member_a, member_index_a
        cdef AINDEX common = 0

        if nb < na:
            neighbours_a, neighbours_b = neighbours_b, neighbours_a
            na, nb = nb, na

        for member_index_a in range(na):
            member_a = neighbours_a.get_member(member_index_a)
            if neighbours_b.contains(member_a):
                common += 1

        return common


cdef class SimilarityCheckerExtContains(SimilarityCheckerExtInterface):
    r"""Implements the similarity checker interface
    Strategy:
        Loops over members of one neighbours container and checks
        if they are contained in the other neighbours container.  Breaks
        early when similarity criterion is reached.
        The performance and time-complexity of the check depends on the
        used neighbour containers.
        Worst case time
        complexity is :math:`\mathcal{O}(n * m)` with :math:`n` and
        :math:`m` being the lengths of the neighbours containers if the
        containment check is performed by iteration.  Worst
        case time complexity is :math:`\mathcal{O}(n)` if containment
        check can be performed as lookup in linear time.  Note that
        no switching of the neighbours containers is done to ensure
        that the first container is the one with the shorter length
        (compare
        :obj:`commonnn._types.SimilarityCheckerExtSwitchContains`).
    """

    cdef bint _check(
            self,
            NeighboursExtInterface neighbours_a,
            NeighboursExtInterface neighbours_b,
            ClusterParameters cluster_params) nogil:

        cdef AINDEX c = cluster_params.iparams[0]

        if c == 0:
            return True

        cdef AINDEX na = neighbours_a._n_points
        cdef AINDEX member_a, member_index_a
        cdef AINDEX common = 0

        for member_index_a in range(na):
            member_a = neighbours_a._get_member(member_index_a)
            if neighbours_b._contains(member_a):
                common += 1
                if common == c:
                    return True
                continue
        return False

    cdef AINDEX _get(
            self,
            NeighboursExtInterface neighbours_a,
            NeighboursExtInterface neighbours_b,
            ClusterParameters cluster_params) nogil:

        cdef AINDEX na = neighbours_a._n_points
        cdef AINDEX member_a, member_index_a
        cdef AINDEX common = 0

        for member_index_a in range(na):
            member_a = neighbours_a._get_member(member_index_a)
            if neighbours_b._contains(member_a):
                common += 1

        return common


cdef class SimilarityCheckerExtSwitchContains(SimilarityCheckerExtInterface):
    r"""Implements the similarity checker interface
    Strategy:
        Loops over members of one neighbours container and checks
        if they are contained in the other neighbours container.  Breaks
        early when similarity criterion is reached.
        The performance and time-complexity of the check depends on the
        used neighbour containers.
        Worst case time
        complexity is :math:`\mathcal{O}(n * m)` with :math:`n` and
        :math:`m` being the lengths of the neighbours containers if the
        containment check is performed by iteration.  Worst
        case time complexity is :math:`\mathcal{O}(n)` if containment
        check can be performed as lookup in linear time.  Note that
        switching of the neighbours containers is done to ensure
        that the first container is the one with the shorter length
        (compare
        :class:`~commonnn._types.SimilarityCheckerExtContains`).
    """

    cdef bint _check(
            self,
            NeighboursExtInterface neighbours_a,
            NeighboursExtInterface neighbours_b,
            ClusterParameters cluster_params) nogil:


        cdef AINDEX c = cluster_params.iparams[0]

        if c == 0:
            return True

        cdef AINDEX na = neighbours_a._n_points
        cdef AINDEX nb = neighbours_b._n_points

        cdef AINDEX member_a, member_index_a
        cdef AINDEX common = 0

        if nb < na:
            with gil:
                neighbours_a, neighbours_b = neighbours_b, neighbours_a
                na, nb = nb, na

        for member_index_a in range(na):
            member_a = neighbours_a._get_member(member_index_a)
            if neighbours_b._contains(member_a):
                common += 1
                if common == c:
                    return True
                continue
        return False

    cdef AINDEX _get(
            self,
            NeighboursExtInterface neighbours_a,
            NeighboursExtInterface neighbours_b,
            ClusterParameters cluster_params) nogil:

        cdef AINDEX na = neighbours_a._n_points
        cdef AINDEX nb = neighbours_b._n_points

        cdef AINDEX member_a, member_index_a
        cdef AINDEX common = 0

        if nb < na:
            with gil:
                neighbours_a, neighbours_b = neighbours_b, neighbours_a
                na, nb = nb, na

        for member_index_a in range(na):
            member_a = neighbours_a._get_member(member_index_a)
            if neighbours_b._contains(member_a):
                common += 1

        return common


cdef class SimilarityCheckerExtScreensorted(SimilarityCheckerExtInterface):
    r"""Implements the similarity checker interface
    Strategy:
        Loops over members of two neighbour containers alternatingly
        and checks if neighbours are contained in both containers.
        Requires that the containers are sorted ascendingly to return
        the correct result. Sorting will neither be checked nor enforced.
        Breaks
        early when similarity criterion is reached.
        The performance of the check depends on the
        used neighbour containers.
        Worst case time
        complexity is :math:`\mathcal{O}(n + m)` with :math:`n` and
        :math:`m` being the lengths of the neighbours containers.
    """

    cdef bint _check(
            self,
            NeighboursExtInterface neighbours_a,
            NeighboursExtInterface neighbours_b,
            ClusterParameters cluster_params) nogil:

        cdef AINDEX c = cluster_params.iparams[0]

        if c == 0:
            return True

        cdef AINDEX na = neighbours_a._n_points
        cdef AINDEX nb = neighbours_b._n_points

        if (na == 0) or (nb == 0):
            return False

        cdef AINDEX member_index_a = 0, member_index_b = 0
        cdef AINDEX member_a, member_b
        cdef AINDEX common = 0

        member_a = neighbours_a._get_member(member_index_a)
        member_b = neighbours_b._get_member(member_index_b)

        while True:
            if member_a == member_b:
                common += 1
                if common == c:
                    return True

                member_index_a += 1
                member_index_b += 1

                if (member_index_a == na) or (member_index_b == nb):
                    break

                member_a = neighbours_a._get_member(member_index_a)
                member_b = neighbours_b._get_member(member_index_b)
                continue

            if member_a < member_b:
                member_index_a += 1
                if (member_index_a == na):
                    break
                member_a = neighbours_a._get_member(member_index_a)
                continue

            member_index_b += 1
            if (member_index_b == nb):
                break
            member_b = neighbours_b._get_member(member_index_b)

        return False

    cdef AINDEX _get(
            self,
            NeighboursExtInterface neighbours_a,
            NeighboursExtInterface neighbours_b,
            ClusterParameters cluster_params) nogil:

        cdef AINDEX na = neighbours_a._n_points
        cdef AINDEX nb = neighbours_b._n_points

        if (na == 0) or (nb == 0):
            return 0

        cdef AINDEX member_index_a = 0, member_index_b = 0
        cdef AINDEX member_a, member_b
        cdef AINDEX common = 0

        member_a = neighbours_a._get_member(member_index_a)
        member_b = neighbours_b._get_member(member_index_b)

        while True:
            if member_a == member_b:
                common += 1

                member_index_a += 1
                member_index_b += 1

                if (member_index_a == na) or (member_index_b == nb):
                    break

                member_a = neighbours_a._get_member(member_index_a)
                member_b = neighbours_b._get_member(member_index_b)
                continue

            if member_a < member_b:
                member_index_a += 1
                if (member_index_a == na):
                    break
                member_a = neighbours_a._get_member(member_index_a)
                continue

            member_index_b += 1
            if (member_index_b == nb):
                break
            member_b = neighbours_b._get_member(member_index_b)

        return common


SimilarityChecker.register(SimilarityCheckerExtContains)
SimilarityChecker.register(SimilarityCheckerExtSwitchContains)
SimilarityChecker.register(SimilarityCheckerExtScreensorted)


class QueueFIFODeque(Queue):
    """Implements the queue interface"""

    def __init__(self):
       self._queue = deque()

    def push(self, value):
        """Append value to back/right end"""
        self._queue.append(value)

    def pop(self):
        return self._queue.popleft()

    def is_empty(self) -> bool:
        if self._queue:
            return False
        return True

    def size(self) -> int:
        return len(self._queue)


cdef class QueueExtLIFOVector(QueueExtInterface):
    """Implements the queue interface"""

    def __cinit__(self, values=None):
        if values is None:
            values = []

        self._queue = values

    cdef inline void _push(self, const AINDEX value) nogil:
        """Append value to back/right end"""
        self._queue.push_back(value)

    cdef inline AINDEX _pop(self) nogil:
        """Retrieve value from back/right end"""

        cdef AINDEX value = self._queue.back()
        self._queue.pop_back()

        return value

    cdef inline bint _is_empty(self) nogil:
        """Return True if there are no values in the queue"""
        return self._queue.empty()

    cdef inline AINDEX _size(self) nogil:
        return self._queue.size()


cdef class QueueExtFIFOQueue(QueueExtInterface):
    """Implements the queue interface"""

    cdef inline void _push(self, const AINDEX value) nogil:
        """Append value to back/right end"""
        self._queue.push(value)

    cdef inline AINDEX _pop(self) nogil:
        """Retrieve value from front/left end"""

        cdef AINDEX value = self._queue.front()
        self._queue.pop()

        return value

    cdef inline bint _is_empty(self) nogil:
        """Return True if there are no values in the queue"""
        return self._queue.empty()

    cdef inline AINDEX _size(self) nogil:
        return self._queue.size()


Queue.register(QueueExtLIFOVector)
Queue.register(QueueExtFIFOQueue)


class PriorityQueueMaxHeap(PriorityQueue):
    """Defines the prioqueue interface"""

    def __init__(self):
       self.reset()

    def reset(self) -> None:
        self._queue = []

    def push(self, a, b, weight) -> None:
        """Put values into the queue"""
        heapq.heappush(self._queue, (-weight, (a, b)))

    def pop(self) -> (int, int, float):
        """Retrieve values from the queue"""
        weight, (a, b) = heapq.heappop(self._queue)
        return a, b, -weight

    def is_empty(self) -> bool:
        """Return True if there are no values in the queue"""

        if self._queue:
            return False
        return True

    def size(self):
        return len(self._queue)
