from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Container, Iterator, Sequence
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

    @property
    @abstractmethod
    def neighbours(self):
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

    def __str__(self):
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

    def __str__(self):
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

    def __str__(self):
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

    def __str__(self):
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

    def __str__(self):
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

    def __str__(self):
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

    def __str__(self):
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

    def __str__(self):
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

    def __str__(self):
        return f"{type(self).__name__}"

    @classmethod
    def get_builder_kwargs(cls):
        return []

cdef class QueueExtInterface:

    cdef void _push(self, const AINDEX value) nogil: ...
    cdef AINDEX _pop(self) nogil: ...
    cdef bint _is_empty(self) nogil: ...

    def push(self, value: int):
        self._push(value)

    def pop(self) -> int:
        return self._pop()

    def is_empty(self) -> bool:
        return self._is_empty()

    def __str__(self):
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

    def __str__(self):
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

    def __str__(self):
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
