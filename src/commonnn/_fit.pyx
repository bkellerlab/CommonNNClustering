from abc import ABC, abstractmethod
from collections import deque
import copy
from types import GeneratorType
from typing import Any, Optional, Type, Union
from typing import Container, Iterable, List, Tuple, Sequence
import heapq
import warnings
import weakref

try:
    import networkx as nx
    NX_FOUND = True
except ModuleNotFoundError as error:
    print("Optional dependency module not found: ", error)  # pragma: no cover
    NX_FOUND = False  # pragma: no cover

import numpy as np
from loguru import logger

from commonnn import report
from commonnn._primitive_types import P_AINDEX, P_AVALUE, P_ABOOL
from commonnn._bundle import reset_hierarchy_levels, leafs_to_labels
from commonnn._bundle cimport check_children, Bundle
from commonnn._types import (
    InputData,
    InputDataComponents,
    InputDataPairwiseDistances,
    InputDataPairwiseDistancesComputer,
    InputDataNeighbourhoods,
    InputDataNeighbourhoodsComputer,
    NeighboursGetter,
    Neighbours,
    DistanceGetter,
    Metric,
    SimilarityChecker,
    Queue,
    PriorityQueue
    )

from cython.operator cimport dereference, preincrement


class Fitter(ABC):
    """Defines the fitter interface"""

    @abstractmethod
    def fit(self, Bundle bundle, *, **kwargs) -> (float, Type["ClusterParameters"]):
        """Generic clustering outer function"""

    @abstractmethod
    def _fit(
            self,
            object input_data,
            Labels labels,
            ClusterParameters cluster_params) -> None:
        """Generic clustering inner function"""

    @abstractmethod
    def make_parameters(
            self, **kwargs) -> Type["ClusterParameters"]:
        """Create fitter specific cluster parameters"""

    def __repr__(self):
        return f"{type(self).__name__}"


class FitterCommonNN(Fitter):
    """Further differentiates the fitter interface for CommonNN clusterings"""

    _parameter_type = CommonNNParameters
    _record_type = report.CommonNNRecord

    def fit(
            self, Bundle bundle, *, purge=True, info=True,
            **kwargs) -> (float, Type["ClusterParameters"]):

        cdef set old_label_set, new_label_set

        if (bundle._labels is None) or purge or (
                not bundle._labels.meta.get("frozen", False)):

            bundle._labels = Labels(
                np.zeros(bundle._input_data.n_points, order="C", dtype=P_AINDEX)
                )
            old_label_set = set()
            current_start = 1
        else:
            old_label_set = bundle._labels.to_set()
            if "start_label" not in kwargs:
                kwargs["start_label"] = max(old_label_set) + 1

        cluster_params = self.make_parameters(**kwargs)
        original_params = self.make_parameters(original=True, **kwargs)

        _, execution_time = report.timed(self._fit)(
            bundle._input_data, bundle._labels, cluster_params
            )

        if info:
            new_label_set = bundle._labels.to_set()
            params = {
                k: (original_params.radius_cutoff, original_params.similarity_cutoff)
                for k in new_label_set - old_label_set
                if k != 0
                }
            meta = {
                "params": params,
                "reference": weakref.proxy(bundle),
                "origin": "fit"
            }
            old_params = bundle._labels.meta.get("params", {})
            old_params.update(meta["params"])
            meta["params"] = old_params
            bundle._labels._meta = meta

        return execution_time, original_params

    def make_parameters(
            self, similarity_offset=0, original=False, **kwargs) -> Type["ClusterParameters"]:

        cluster_params = self._parameter_type.from_mapping(kwargs)
        cluster_params.similarity_cutoff -= similarity_offset

        assert cluster_params.similarity_cutoff >= 0

        if original:
            return cluster_params

        try:
            used_metric = self._neighbours_getter._distance_getter._metric
        except AttributeError:
            pass
        else:
            cluster_params.radius_cutoff = used_metric.adjust_radius(cluster_params.radius_cutoff)

        try:
            is_selfcounting = self._neighbours_getter.is_selfcounting
        except AttributeError:
            pass
        else:
            if is_selfcounting:
                cluster_params._support_cutoff += 1
                cluster_params.similarity_cutoff += 2

        return cluster_params

    def get_fit_signature(self):
        fparams = []
        for pn in self._parameter_type._fparam_names:
            # TODO/TEST: Not tested because there is no default float parameter yet
            if pn in self._parameter_type._defaults:
                p = self._parameter_type._defaults[pn]
                if isinstance(p, str):
                    p = "<" + self._parameter_type._fparam_names[int(p.strip("<>"))] + ">"
                fparams.append(f"{pn}: Optional[float] = {p}")
            else:
                fparams.append(f"{pn}: float")

        iparams = []
        for pn in self._parameter_type._iparam_names:
            if pn in self._parameter_type._defaults:
                p = self._parameter_type._defaults[pn]
                if isinstance(p, str):
                    p = "<" + self._parameter_type._iparam_names[int(p.strip("<>"))] + ">"
                iparams.append(f"{pn}: Optional[int] = {p}")
            else:
                iparams.append(f"{pn}: int")

        return f'fit({", ".join(fparams)}, {", ".join(iparams)})'


class HierarchicalFitter(ABC):
    """Defines the hierarchical fitter interface"""

    @abstractmethod
    def fit(self, Bundle bundle, *, **kwargs) -> None:
        """Generic clustering"""

    def __repr__(self):
        return f"{type(self).__name__}"


cdef class FitterExtInterface:
    def fit(self, Bundle bundle, *, **kwargs) -> (float, Type["ClusterParameters"]): ...

    def _fit(
            self,
            InputDataExtInterface input_data,
            Labels labels,
            ClusterParameters cluster_params) -> None:

        self._fit_inner(input_data, labels, cluster_params)

    cdef void _fit_inner(
            self,
            InputDataExtInterface input_data,
            Labels labels,
            ClusterParameters cluster_params) nogil: ...

    def make_parameters(
            self, **kwargs) -> Type["ClusterParameters"]: ...

    def __repr__(self):
        return f"{type(self).__name__}"


cdef class FitterExtCommonNNInterface(FitterExtInterface):

    _parameter_type = CommonNNParameters
    _record_type = report.CommonNNRecord

    def fit(self, Bundle bundle, *, **kwargs) -> (float, Type["ClusterParameters"]):
        return FitterCommonNN.fit(self, bundle, **kwargs)

    def make_parameters(
            self, **kwargs) -> Type["ClusterParameters"]:
        return FitterCommonNN.make_parameters(self, **kwargs)

    def get_fit_signature(self):
        return FitterCommonNN.get_fit_signature(self)


class Predictor(ABC):
    """Defines the predictor interface"""

    @abstractmethod
    def predict(self, Bundle bundle, Bundle other, *, **kwargs):
        """Generic prediction"""

    @abstractmethod
    def _predict(
            self,
            input_data: Type['InputData'],
            predictand_input_data: Type['InputData'],
            labels: Type['Labels'],
            predictand_labels: Type['Labels'],
            cluster_params: Type['ClusterParameters']):
        """Generic prediction"""

    @abstractmethod
    def make_parameters(
            self, **kwargs) -> Type["ClusterParameters"]:
        """Create fitter specific cluster parameters"""

    def __repr__(self):
        return f"{type(self).__name__}"


class PredictorCommonNN(Predictor):

    _parameter_type = CommonNNParameters

    def predict(
            self, Bundle bundle, Bundle other, *,
            clusters=None, purge=True, info=True, **kwargs):

        if (other._labels is None) or purge or (
                not other._labels.meta.get("frozen", False)):
            other._labels = Labels(
                np.zeros(other._input_data.n_points, order="C", dtype=P_AINDEX)
                )

        cluster_params = self.make_parameters(**kwargs)
        original_params = self.make_parameters(original=True, **kwargs)

        if clusters is None:
           clusters = bundle._labels.to_set() - {0}

        other._labels.consider_set = clusters

        _, execution_time = report.timed(self._predict)(
            bundle._input_data, other._input_data,
            bundle._labels, other._labels,
            cluster_params
        )

        if info:
            params = {
                k: (original_params.radius_cutoff, original_params.similarity_cutoff)
                for k in clusters
                if k != 0
                }
            meta = {
                "params": params,
                "reference": weakref.proxy(bundle),
                "origin": "predict",
            }
            old_params = other._labels.meta.get("params", {})
            old_params.update(meta["params"])
            meta["params"] = old_params
            other._labels._meta = meta

        other._labels.meta["frozen"] = True

    def make_parameters(
            self, **kwargs) -> Type["ClusterParameters"]:
        return FitterCommonNN.make_parameters(self, **kwargs)


    def get_fit_signature(self):
        return FitterCommonNN.get_fit_signature(self)


class FitterCommonNNBFS(FitterCommonNN):
    """Concrete implementation of the fitter interface

    Args:
        neighbours_getter: Any object implementing the neighbours getter
            interface.
        neighbours: Any object implementing the neighbours
            interface.
        neighbour_neighbourss: Any object implementing the neighbours
            interface.
        similarity_checker: Any object implementing the similarity checker
            interface.
        queue: Any object implementing the queue interface.
    """

    def __init__(
            self,
            neighbours_getter: Type["NeighboursGetter"],
            neighbours: Type["Neighbours"],
            neighbour_neighbours: Type["Neighbours"],
            similarity_checker: Type["SimilarityChecker"],
            queue: Type["Queue"]):

        self._neighbours_getter = neighbours_getter
        self._neighbours = neighbours
        self._neighbour_neighbours = neighbour_neighbours
        self._similarity_checker = similarity_checker
        self._queue = queue

    def __str__(self):

        attr_str = ", ".join([
            f"ngetter={self._neighbours_getter}",
            f"na={self._neighbours}",
            f"nb={self._neighbour_neighbours}",
            f"checker={self._similarity_checker}",
            f"queue={self._queue}",
        ])

        return f"{type(self).__name__}({attr_str})"

    @classmethod
    def get_builder_kwargs(cls):
        return [
            ("neighbours_getter", None),
            ("neighbours", None),
            ("neighbour_neighbours","neighbours"),
            ("similarity_checker", None),
            ("queue", None),
            ]

    def _fit(
            self,
            object input_data,
            Labels labels,
            ClusterParameters cluster_params) -> None:
        """Generic common-nearest-neighbour clustering

        Uses a breadth-first-search (BFS) approach to grow clusters.

        Args:
            input_data: Data source implementing the input data
                interface.
            labels: Instance of :obj:`commonnn._types.Labels`.
            cluster_params: Instance of
                :obj:`commonnn._types.ClusterParameters`.
        """

        cdef AINDEX _support_cutoff = cluster_params.get_iparam(1)
        cdef AINDEX n, m, current
        cdef AINDEX init_point, point, member, member_index
        cdef AINDEX* _labels = &labels._labels[0]
        cdef ABOOL* _consider = &labels._consider[0]

        n = input_data._n_points
        current = cluster_params.get_iparam(2)

        for init_point in range(n):
            if _consider[init_point] == 0:
                continue
            _consider[init_point] = 0

            self._neighbours_getter.get(
                init_point,
                input_data,
                self._neighbours,
                cluster_params
                )

            if not self._neighbours.enough(_support_cutoff):
                continue

            _labels[init_point] = current

            while True:

                m = self._neighbours._n_points

                for member_index in range(m):
                    member = self._neighbours.get_member(member_index)

                    if _consider[member] == 0:
                        continue

                    self._neighbours_getter.get(
                        member,
                        input_data,
                        self._neighbour_neighbours,
                        cluster_params
                        )

                    if not self._neighbour_neighbours.enough(_support_cutoff):
                        _consider[member] = 0
                        continue

                    if self._similarity_checker.check(
                            self._neighbours,
                            self._neighbour_neighbours,
                            cluster_params):
                        _consider[member] = 0
                        _labels[member] = current
                        self._queue.push(member)

                if self._queue.is_empty():
                    break

                point = self._queue.pop()
                self._neighbours_getter.get(
                    point,
                    input_data,
                    self._neighbours,
                    cluster_params
                    )

            current += 1


class FitterCommonNNBFSDebug(FitterCommonNN):
    """Concrete implementation of the fitter interface

    Yields/prints information during the clustering.

    Args:
        neighbours_getter: Any extension type
            implementing the neighbours getter
            interface.
        neighbours: Any extension type implementing the neighbours
            interface.
        neighbour_neighbourss: Any extension type implementing the neighbours
            interface.
        similarity_checker: Any extension type implementing the similarity checker
            interface.
        queue: Any extension type implementing the queue interface. Used
            during the clustering procedure.
    """

    def __init__(
            self,
            neighbours_getter: Type["NeighboursGetter"],
            neighbours: Type["Neighbours"],
            neighbour_neighbours: Type["Neighbours"],
            similarity_checker: Type["SimilarityChecker"],
            queue: Type["Queue"],
            verbose=True,
            yielding=True):

        self._neighbours_getter = neighbours_getter
        self._neighbours = neighbours
        self._neighbour_neighbours = neighbour_neighbours
        self._similarity_checker = similarity_checker
        self._queue = queue
        self._verbose = verbose
        self._yielding = yielding

    def __str__(self):

        attr_str = ", ".join([
            f"ngetter={self._neighbours_getter}",
            f"na={self._neighbours}",
            f"nb={self._neighbour_neighbours}",
            f"checker={self._similarity_checker}",
            f"queue={self._queue}",
            f"verbose={self._verbose}",
            f"yielding={self._yielding}",
        ])

        return f"{type(self).__name__}({attr_str})"

    @classmethod
    def get_builder_kwargs(cls):
        return [
            ("neighbours_getter", None),
            ("neighbours", None),
            ("neighbour_neighbours","neighbours"),
            ("similarity_checker", None),
            ("queue", None),
            ]

    def _fit(
            self,
            object input_data,
            Labels labels,
            ClusterParameters cluster_params) -> None:
        """Generic common-nearest-neighbour clustering

        Uses a breadth-first-search (BFS) approach to grow clusters.

        Args:
            input_data: Data source implementing the input data
                interface.
            labels: Instance of :obj:`commonnn._types.Labels`.
            cluster_params: Instance of
                :obj:`commonnn._types.ClusterParameters`.
        """

        deque(self._fit_debug(input_data, labels, cluster_params), maxlen=0)

    def _fit_debug(
            self,
            object input_data,
            Labels labels,
            ClusterParameters cluster_params) -> GeneratorType:
        """Generic common-nearest-neighbour clustering in debug mode

        Uses a breadth-first-search (BFS) approach to grow clusters.

        Args:
            input_data: Data source implementing the input data
                interface.
            labels: Instance of :obj:`commonnn._types.Labels`.
            cluster_params: Instance of
                :obj:`commonnn._types.ClusterParameters`.
        """

        cdef AINDEX _support_cutoff = cluster_params.get_iparam(1)
        cdef AINDEX n, m, current
        cdef AINDEX init_point, point, member, member_index
        cdef AINDEX* _labels = &labels._labels[0]
        cdef ABOOL* _consider = &labels._consider[0]

        n = input_data._n_points
        current = cluster_params.get_iparam(2)

        if self._verbose:
            print(f"CommonNN clustering - {type(self).__name__}")
            print("=" * 80)
            print(f"{n} points")
            print(
                *(
                    f"{k:<29}: {v}"
                    for k, v in cluster_params.to_dict().items()
                ),
                sep="\n"
            )

        for init_point in range(n):

            if self._verbose:
                print()
                print(f"New source: {init_point}")

            if _consider[init_point] == 0:
                if self._verbose:
                    print("    ... already visited\n")
                continue
            _consider[init_point] = 0

            self._neighbours_getter.get(
                init_point,
                input_data,
                self._neighbours,
                cluster_params
                )

            if not self._neighbours.enough(_support_cutoff):
                if self._verbose:
                    print("    ... not enough neighbours\n")
                continue

            _labels[init_point] = current
            if self._verbose:
                print(f"    ... new cluster {current}")

            if self._yielding:
                yield {
                    "reason": "assigned_source",
                    "init_point": init_point,
                    "point": None,
                    "member": None,
                    }

            while True:

                m = self._neighbours._n_points
                if self._verbose:
                    print(f"    ... loop over {m} neighbours")

                for member_index in range(m):
                    member = self._neighbours.get_member(member_index)

                    if self._verbose:
                        print(f"        ... current neighbour {member}")

                    if _consider[member] == 0:
                        if self._verbose:
                            print(f"        ... already visited\n")
                        continue

                    self._neighbours_getter.get(
                        member,
                        input_data,
                        self._neighbour_neighbours,
                        cluster_params
                        )

                    if not self._neighbour_neighbours.enough(_support_cutoff):
                        _consider[member] = 0
                        if self._verbose:
                            print("        ... not enough neighbours\n")
                        continue

                    if self._similarity_checker.check(
                            self._neighbours,
                            self._neighbour_neighbours,
                            cluster_params):

                        if self._verbose:
                            print("        ... successful check!\n")

                        _consider[member] = 0
                        _labels[member] = current

                        if self._yielding:
                            yield {
                                "reason": "assigned_neighbour",
                                "init_point": init_point,
                                "point": point,
                                "member": member,
                                }
                        self._queue.push(member)

                if self._queue.is_empty():
                    if self._verbose:
                        print(f"    ... finished cluster {current}")
                        print("=" * 80)
                    break

                point = self._queue.pop()

                if self._verbose:
                    print(f"    ... Next point: {point}")

                self._neighbours_getter.get(
                    point,
                    input_data,
                    self._neighbours,
                    cluster_params
                    )

            current += 1


cdef class FitterExtCommonNNBFS(FitterExtCommonNNInterface):
    """Concrete implementation of the fitter interface

    Realises a CommonNN clustering using a breadth-first-search

    Args:
        neighbours_getter: Any extension type
            implementing the neighbours getter
            interface.
        neighbours: Any extension type implementing the neighbours
            interface.
        neighbour_neighbourss: Any extension type implementing the neighbours
            interface.
        similarity_checker: Any extension type implementing the similarity checker
            interface.
        queue: Any extension type implementing the queue interface. Used
            during the clustering procedure.
    """

    def __cinit__(
            self,
            NeighboursGetterExtInterface neighbours_getter,
            NeighboursExtInterface neighbours,
            NeighboursExtInterface neighbour_neighbours,
            SimilarityCheckerExtInterface similarity_checker,
            QueueExtInterface queue):
        self._neighbours_getter = neighbours_getter
        self._neighbours = neighbours
        self._neighbour_neighbours = neighbour_neighbours
        self._similarity_checker = similarity_checker
        self._queue = queue

    def __str__(self):

        attr_str = ", ".join([
            f"ngetter={self._neighbours_getter}",
            f"na={self._neighbours}",
            f"nb={self._neighbour_neighbours}",
            f"checker={self._similarity_checker}",
            f"queue={self._queue}",
        ])

        return f"{type(self).__name__}({attr_str})"

    @classmethod
    def get_builder_kwargs(cls):
        return [
            ("neighbours_getter", None),
            ("neighbours", None),
            ("neighbour_neighbours","neighbours"),
            ("similarity_checker", None),
            ("queue", None),
            ]

    cdef void _fit_inner(
            self,
            InputDataExtInterface input_data,
            Labels labels,
            ClusterParameters cluster_params) nogil:
        """Generic common-nearest-neighbours clustering

        Uses a breadth-first-search (BFS) approach to grow clusters.

        Args:
            input_data: Data source implementing the input data
                interface.
            labels: Instance of :obj:`commonnn._types.Labels`.
            cluster_params: Instance of
                :obj:`commonnn._types.ClusterParameters`.
        """

        cdef AINDEX _support_cutoff = cluster_params.iparams[1]
        cdef AINDEX n, m, current
        cdef AINDEX init_point, point, member, member_index
        cdef AINDEX* _labels = &labels._labels[0]
        cdef ABOOL* _consider = &labels._consider[0]

        n = input_data._n_points
        current = cluster_params.iparams[2]

        for init_point in range(n):
            if _consider[init_point] == 0:
                continue
            _consider[init_point] = 0

            self._neighbours_getter._get(
                init_point,
                input_data,
                self._neighbours,
                cluster_params
                )

            if not self._neighbours._enough(_support_cutoff):
                continue

            _labels[init_point] = current

            while True:

                m = self._neighbours._n_points

                for member_index in range(m):
                    member = self._neighbours._get_member(member_index)

                    if _consider[member] == 0:
                        continue

                    self._neighbours_getter._get(
                        member,
                        input_data,
                        self._neighbour_neighbours,
                        cluster_params
                        )

                    if not self._neighbour_neighbours._enough(_support_cutoff):
                        _consider[member] = 0
                        continue

                    if self._similarity_checker._check(
                            self._neighbours,
                            self._neighbour_neighbours,
                            cluster_params):
                        _consider[member] = 0
                        _labels[member] = current
                        self._queue._push(member)

                if self._queue._is_empty():
                    break

                point = self._queue._pop()
                self._neighbours_getter._get(
                    point,
                    input_data,
                    self._neighbours,
                    cluster_params
                    )

            current += 1


Fitter.register(FitterExtCommonNNBFS)


class HierarchicalFitterCommonNNMSTPrim(HierarchicalFitter):
    """Concrete implementation of the hierarchical fitter interface

    Builds a minimum spanning tree on the density estimate
    using Prim's algorithm and
    creates a cluster hierarchy via single linkage clustering.

    Note:
        This is still experimental.
    """

    _parameter_type = RadiusParameters

    def __init__(
            self,
            neighbours_getter: Type["NeighboursGetter"],
            neighbours: Type["Neighbours"],
            neighbour_neighbours: Type["Neighbours"],
            similarity_checker: Type["SimilarityChecker"],
            priority_queue: Type["PriorityQueue"],
            priority_queue_tree: Type["PriorityQueue"]):
        """
        Args:
            neighbours_getter: Any object implementing the neighbours getter
                interface.
            neighbours: Any object implementing the neighbours
                interface.
            neighbour_neighbours: Any object implementing the neighbours
                interface.
            similarity_checker: Any object implementing the similarity checker
                interface.
            priority_queue:
                Any object implementing the prioqueue interface (max heap).
            priority_queue_tree:
                Any object implementing the prioqueue interface (max heap).
        """

        self._neighbours_getter = neighbours_getter
        self._neighbours = neighbours
        self._neighbour_neighbours = neighbour_neighbours
        self._similarity_checker = similarity_checker
        self._priority_queue = priority_queue
        self._priority_queue_tree = priority_queue_tree
        self._artifacts = {}

    def __str__(self):

        attr_str = ", ".join([
            f"ngetter={self._neighbours_getter}",
            f"na={self._neighbours}",
            f"nb={self._neighbour_neighbours}",
            f"checker={self._similarity_checker}",
            f"prioq={self._priority_queue}",
            f"prioq_tree={self._priority_queue_tree}"
        ])

        return f"{type(self).__name__}({attr_str})"

    @classmethod
    def get_builder_kwargs(cls):
        return [
            ("neighbours_getter", None),
            ("neighbours", None),
            ("neighbour_neighbours","neighbours"),
            ("similarity_checker", None),
            ("priority_queue", None),
            ("priority_queue_tree", "priority_queue")
            ]

    def make_parameters(self, **kwargs) -> Type["ClusterParameters"]:
        cluster_params = self._parameter_type.from_mapping(kwargs)

        try:
            used_metric = self._neighbours_getter._distance_getter._metric
        except AttributeError:
            pass
        else:
            cluster_params.radius_cutoff = used_metric.adjust_radius(cluster_params.radius_cutoff)

        return cluster_params

    def fit(
            self,
            Bundle bundle, *,
            info=True,
            member_cutoff=10,
            scipy_hierarchy=True,
            bundle_hierarchy=True,
            make_labels=True,
            **kwargs) -> None:
        """Orchestrates hierarchical clustering
        
        Args:
            bundle: Bundle object containing the input data and labels.
            info: If True, store the parameters in the labels meta dictionary.
            member_cutoff: Minimum number of members for clusters to be
                considered valid. Only considered for hierarchy building
                if `bundle_hierarchy=True`.
            scipy_hierarchy: If `True`, build a SciPy-compatible linkage hierarchy
                Z-matrix from MST edges that will be put into `_artifacts["Z"]`.
            bundle_hierarchy: If `True`, build a bundle hierarchy from the
                MST edges or (if present) a previously computed Scipy Z-matrix.
                Note that currently, we only recommend the latter option for
                which `scipy_hierarchy=True` is required.
            make_labels: If `True`, create root labels from all leaf nodes
                after a bundle hierarchy has been built (`bundle_hierarchy=True`).
        """

        cdef AINDEX n_points = bundle._input_data.n_points

        logger.info(f"Started hierarchical CommonNN clustering - {type(self).__name__}")

        bundle._labels = Labels(
            np.ones(n_points, order="C", dtype=P_AINDEX)
            )

        cluster_params = self.make_parameters(**kwargs)

        _, execution_time = report.timed(self._fit)(
            bundle._input_data, bundle._labels, cluster_params
            )

        if scipy_hierarchy:
            self._make_scipy_hierarchy(n_points)
            logger.info(f"Computed Scipy Z matrix")

        if bundle_hierarchy:
            if scipy_hierarchy:
                Z = self._artifacts.get("Z")
                if Z is not None:
                    self.scipy_to_bundle_hierarchy(
                        bundle,
                        Z=Z,
                        member_cutoff=member_cutoff
                        )
                    logger.info(f"Built bundle hierarchy from Scipy Z matrix")
                else:
                    raise LookupError(
                        "Scipy Z matrix not found. "
                        )
            else:
                self._make_bundle_hierarchy(bundle, n_points, member_cutoff=member_cutoff)
                logger.info(f"Built bundle hierarchy")

            reset_hierarchy_levels(bundle)
            if make_labels:
                leafs_to_labels(bundle, n_points=n_points)
                logger.info(f"Converted all leafs to root labels")

        if info:
            meta = {
                "params": {},
                "reference": weakref.proxy(bundle),
                "origin": "fit_hierarchical"
            }
            bundle._labels._meta = meta

        return execution_time, cluster_params

    def _fit(
            self,
            object input_data,
            Labels labels,
            ClusterParameters cluster_params) -> None:

        self._make_mst(input_data, labels, cluster_params)
        logger.info(f"Built MST")

    def _make_mst(
            self,
            object input_data,
            Labels labels,
            ClusterParameters cluster_params) -> None:

        cdef AINDEX n_points, point, n_members, member, member_index, a, b
        cdef AVALUE weight
        cdef ABOOL* _consider = &labels._consider[0]

        self._priority_queue.reset()
        self._priority_queue_tree.reset()

        n_points = input_data.n_points
        for point in range(n_points):
            if _consider[point] == 0:
                continue
            _consider[point] = 0

            self._neighbours_getter.get(
                point,
                input_data,
                self._neighbours,
                cluster_params
                )

            n_members = self._neighbours._n_points
            if (n_members - self._neighbours_getter.is_selfcounting) == 0:
                logger.warning(f"Point {point} has no neighbours")

            for member_index in range(n_members):
                member = self._neighbours.get_member(member_index)

                if _consider[member] == 0:
                    continue

                self._neighbours_getter.get(
                    member,
                    input_data,
                    self._neighbour_neighbours,
                    cluster_params
                    )

                weight = self._similarity_checker.get(
                    self._neighbours,
                    self._neighbour_neighbours,
                    cluster_params
                    )

                self._priority_queue.push(point, member, weight)

            while not self._priority_queue.is_empty():
                a, b, weight = self._priority_queue.pop()

                if _consider[b] == 0:
                    continue
                _consider[b] = 0

                self._priority_queue_tree.push(a, b, weight)

                self._neighbours_getter.get(
                    b,
                    input_data,
                    self._neighbours,
                    cluster_params
                    )

                n_members = self._neighbours._n_points
                for member_index in range(n_members):
                    member = self._neighbours.get_member(member_index)

                    if _consider[member] == 0:
                        continue

                    self._neighbours_getter.get(
                        member,
                        input_data,
                        self._neighbour_neighbours,
                        cluster_params
                        )

                    weight = self._similarity_checker.get(
                        self._neighbours,
                        self._neighbour_neighbours,
                        cluster_params
                        )

                    self._priority_queue.push(b, member, weight)

    def _make_scipy_hierarchy(self, AINDEX n_points) -> None:
        """Build a SciPy-compatible linkage matrix Z from MST edges"""

        cdef stdvector[AINDEX] top_roots = []
        cdef AINDEX n_edges = self._priority_queue_tree.size()
        cdef AVALUE[:, ::1] Z = np.zeros((n_points - 1, 4), dtype=P_AVALUE)
        cdef AINDEX[::1] parents_indicator = np.arange(n_points * 2 - 1, dtype=P_AINDEX)
        cdef AINDEX[::1] seen_indicator = np.zeros(n_points, dtype=P_AINDEX)
        cdef AINDEX i, a, b, new_id, root_a, root_b, index
        cdef AVALUE weight, size_a, size_b

        logger.debug(f"{n_edges} edges in MST")

        i = 0
        while not self._priority_queue_tree.is_empty():

            a, b, weight = self._priority_queue_tree.pop()
            seen_indicator[a] = 1
            seen_indicator[b] = 1

            root_a = get_root(a, parents_indicator)
            root_b = get_root(b, parents_indicator)

            logger.debug(f"Iteration {i}: Points {a} (cluster {root_a}) and {b} (cluster {root_b}) connect at weight={weight}")
            
            if root_a >= n_points:
                size_a = Z[root_a - n_points, 3]
            else:
                size_a = 1
                
            if root_b >= n_points:
                size_b = Z[root_b  - n_points, 3]
            else:
                size_b = 1

            # Record the merge in Z
            Z[i, 0] = min(root_a, root_b)
            Z[i, 1] = max(root_a, root_b)
            Z[i, 2] = weight
            Z[i, 3] = size_a + size_b

            new_id = n_points + i
            parents_indicator[root_a] = new_id
            parents_indicator[root_b] = new_id

            i += 1

        # Merge disjoint components
        for index in range(n_points + i):
            if parents_indicator[index] == index:
                top_roots.push_back(index)

        if top_roots.size() > 1:
            logger.info(f"Found {top_roots.size()} disjoint components")
            root_a = top_roots.back()
            top_roots.pop_back()
            while top_roots.size() > 0:
                root_b =  top_roots.back()
                top_roots.pop_back()

                if root_a >= n_points:
                    size_a = Z[root_a - n_points, 3]
                else:
                    size_a = 1
                    
                if root_b >= n_points:
                    size_b = Z[root_b - n_points, 3]
                else:
                    size_b = 1

                Z[i, 0] = min(root_a, root_b)
                Z[i, 1] = max(root_a, root_b)
                Z[i, 2] = 0
                Z[i, 3] = size_a + size_b
                
                new_id = n_points + i
                parents_indicator[root_a] = new_id
                parents_indicator[root_b] = new_id
                root_a = new_id

                i += 1

        self._artifacts["Z"] = np.array(Z)

    def scipy_to_bundle_hierarchy(self, Bundle bundle, *, AVALUE[:, ::1] Z, AINDEX member_cutoff=10) -> None:
        """Build a hierarchy of bundles from a SciPy-compatible linkage matrix Z
        
        Args:
            bundle: Root bundle
            Z: SciPy-compatible linkage matrix
        
        Keyword args:
            member_cutoff: Minimum number of members in a bundle to
                consider it as independent cluster. Clusters with fewer
                members are merged into their parent cluster or respectively
                split off as noise.
        """

        cdef Bundle parent_bundle, bundle_a, bundle_b, bundle_x
        cdef dict top_bundles = {}
        cdef AINDEX n_points = Z.shape[0] + 1
        cdef AINDEX i, new_id
        cdef AVALUE INFTY = np.inf
        cdef AVALUE w, prev_weight = INFTY
        cdef bint needs_folding = False

        for i in range(Z.shape[0]):
            new_id = i + n_points

            a = <AINDEX>Z[i, 0]
            b = <AINDEX>Z[i, 1]
            w = Z[i, 2]

            if w < prev_weight:
                # Check top bundles only if weight has decreased
                # Fold only if weight has been the same during previous iterations
                for bundle_x in top_bundles.values():
                    if not bundle_x._checked:
                        check_children(bundle_x, member_cutoff, needs_folding=needs_folding)
                        bundle_x._checked = True
                needs_folding = False
            else:
                needs_folding = True

            # Get top bundles or make singleton bundles on-the-fly
            if a < n_points:
                bundle_a = Bundle(alias=f"{a}", graph={a})
                bundle_a._size = 1
                bundle_a._lambda = INFTY
            else:
                bundle_a = top_bundles[a]

            if b < n_points:
                bundle_b = Bundle(alias=f"{b}", graph={b})
                bundle_b._size = 1
                bundle_b._lambda = INFTY
            else:
                bundle_b = top_bundles[b]

            # logger.debug(f"Iteration {i}: {bundle_a.alias} (size={bundle_a._size}) and {bundle_b.alias} (size={bundle_b._size}) connect at weight={w}")

            # Make parent bundle and update top bundles
            parent_bundle = Bundle(alias=f"{new_id}", graph=bundle_a.graph | bundle_b.graph, children={1: bundle_a, 2: bundle_b})
            parent_bundle._lambda = w
            parent_bundle._size = bundle_a._size + bundle_b._size
            top_bundles[new_id] = parent_bundle

            if a >= n_points:
                del top_bundles[a]
            if b >= n_points:
                del top_bundles[b]
                
            prev_weight = w

        # Check remaining top bundles (should only be root if Z is complete)
        for bundle_x in top_bundles.values():
            if not bundle_x._checked:
                check_children(bundle_x, member_cutoff, needs_folding=needs_folding)
                bundle_x._checked = True

        bundle._children = parent_bundle._children
        bundle._lambda = parent_bundle._lambda
        bundle._size = parent_bundle._size
        for child in bundle._children.values():
            child._parent = weakref.proxy(bundle)


    def _make_bundle_hierarchy(self, Bundle bundle, AINDEX n_points, AINDEX member_cutoff=10) -> None:
        """Build a hierarchy of bundles from MST edges
        
        Note:
            This function should not be used in production at the moment
            and will undergo changes in the future. Use :meth:`scipy_to_bundle_hierarchy`
            after :meth:`_make_scipy_hierarchy` instead.
        
        Note:
            This function consumes MST edges from the priority queue
            (:attr:`_priority_queue_tree`) filled by :meth:`_make_mst`
            and can only be called once before the MST needs to be rebuilt.
            
        Args:
            bundle: Root bundle
            n_points: Number of points in the dataset

        Keyword args:
            member_cutoff: Minimum number of members in a bundle to
                consider it as independent cluster. Clusters with fewer
                members are merged into their parent cluster or respectively
                split off as noise.
        """

        warnings.warn(
            "This function is deprecated and will be removed/changed in the future. "
            "Use `scipy_to_bundle_hierarchy` after `_make_scipy_hierarchy` instead "
            "for consistency and performance.",
            DeprecationWarning
            )

        if not NX_FOUND:
            raise ModuleNotFoundError("No module named 'networkx'")

        cdef AVALUE INFTY = np.inf
        cdef AINDEX[::1] parent_indicator = np.arange(n_points)
        cdef AINDEX[::1] root_clabel_indicator = np.arange(n_points)
        cdef dict top_bundles = {}
        cdef ABOOL[::1] checked = np.zeros(n_points - 1, dtype=bool)
        cdef Bundle top, left, right
        cdef AVALUE weight, current_weight = INFTY
        cdef bint needs_folding = False
        cdef AINDEX i, a, b, ra, rb, la, lb, bundle_index

        i = -1
        for i in range(self._priority_queue_tree.size()):
            a, b, weight = self._priority_queue_tree.pop()

            ra = get_root(a, parent_indicator)
            rb = get_root(b, parent_indicator)
            la = root_clabel_indicator[ra]
            lb = root_clabel_indicator[rb]

            if (weight < current_weight):
                for bundle_index, top in top_bundles.items():
                    if checked[bundle_index]:
                        continue
                    check_children(top, member_cutoff, needs_folding)
                    checked[bundle_index] = True
                needs_folding = False
            else:
                needs_folding = True

            top = top_bundles[i] = Bundle(
                alias=f"{i + n_points}",
                graph=nx.Graph([(a, b, {"weight": weight})]),
            )
            top._lambda = weight

            label = 1
            if (la >= n_points):
                left = top_bundles[la - n_points]
                top._graph.add_edges_from(left._graph.edges(data=True))
                children = top.children
                children[label] = left
                top._children = children
                del top_bundles[la - n_points]
                label += 1

            if (lb >= n_points):
                right = top_bundles[lb - n_points]
                top._graph.add_edges_from(right._graph.edges(data=True))
                children = top.children
                children[label] = right
                top._children = children
                del top_bundles[lb - n_points]

            parent_indicator[ra] = rb
            root_clabel_indicator[rb] = i + n_points
            current_weight = weight

        for bundle_index, top in top_bundles.items():
            if checked[bundle_index]:
                continue
            check_children(top, member_cutoff, needs_folding)

        if i < (n_points - 2):

            bundle._lambda = -INFTY
            bundle._graph = nx.Graph()

            count = 1
            for top in top_bundles.values():
                if len(top._graph) >= member_cutoff:
                    bundle._graph.add_edges_from(bundle._graph.edges(data=True))
                    children = bundle.children
                    children[count] = top
                    bundle._children =  children
                    count += 1

            for child in bundle._children.values():
                child._parent = weakref.proxy(bundle)

        else:
            top = top_bundles[n_points - 2]
            bundle._graph = top._graph
            bundle._lambda = top._lambda


cdef class HierarchicalFitterExtCommonNNMSTPrim:

    _parameter_type = RadiusParameters

    def __cinit__(
            self,
            NeighboursGetterExtInterface neighbours_getter,
            NeighboursExtInterface neighbours,
            NeighboursExtInterface neighbour_neighbours,
            SimilarityCheckerExtInterface similarity_checker,
            PriorityQueueExtInterface priority_queue,
            PriorityQueueExtInterface priority_queue_tree):
        self._neighbours_getter = neighbours_getter
        self._neighbours = neighbours
        self._neighbour_neighbours = neighbour_neighbours
        self._similarity_checker = similarity_checker
        self._priority_queue = priority_queue
        self._priority_queue_tree = priority_queue_tree
        self._artifacts = {}

    def __str__(self):
        return HierarchicalFitterCommonNNMSTPrim.__str__(self)

    @classmethod
    def get_builder_kwargs(cls):
        return HierarchicalFitterCommonNNMSTPrim.get_builder_kwargs()

    def make_parameters(self, **kwargs) -> Type["ClusterParameters"]:
        return HierarchicalFitterCommonNNMSTPrim.make_parameters(self, **kwargs)

    def fit(self, bundle, *, info=True, member_cutoff=10, scipy_hierarchy=True, bundle_hierarchy=True, **kwargs) -> (float, Type["ClusterParameters"]):
        HierarchicalFitterCommonNNMSTPrim.fit(self, bundle, info=info, member_cutoff=member_cutoff, scipy_hierarchy=scipy_hierarchy, bundle_hierarchy=bundle_hierarchy, **kwargs)

    def _fit(
            self,
            InputDataExtInterface input_data,
            Labels labels,
            ClusterParameters cluster_params) -> None:

        self._make_mst(input_data, labels, cluster_params)
        logger.info(f"Built MST")

    cdef void _make_mst(
            self,
            InputDataExtInterface input_data,
            Labels labels,
            ClusterParameters cluster_params) nogil:

        cdef AINDEX n_points, point, n_members, member, member_index, a, b
        cdef AVALUE weight
        cdef ABOOL* _consider = &labels._consider[0]
        cdef (AINDEX, AINDEX, AVALUE) edge

        self._priority_queue._reset()
        self._priority_queue_tree._reset()

        n_points = input_data._n_points
        for point in range(n_points):
            if _consider[point] == 0:
                continue
            _consider[point] = 0

            self._neighbours_getter._get(
                point,
                input_data,
                self._neighbours,
                cluster_params
                )

            n_members = self._neighbours._n_points
            for member_index in range(n_members):
                member = self._neighbours._get_member(member_index)

                if _consider[member] == 0:
                    continue

                self._neighbours_getter._get(
                    member,
                    input_data,
                    self._neighbour_neighbours,
                    cluster_params
                    )

                weight = self._similarity_checker._get(
                    self._neighbours,
                    self._neighbour_neighbours,
                    cluster_params
                    )

                self._priority_queue._push(point, member, weight)

            while not self._priority_queue._is_empty():
                edge = self._priority_queue._pop()
                a = edge[0]
                b = edge[1]
                weight = edge[2]

                if _consider[b] == 0:
                    continue
                _consider[b] = 0

                self._priority_queue_tree._push(a, b, weight)

                self._neighbours_getter._get(
                    b,
                    input_data,
                    self._neighbours,
                    cluster_params
                    )

                n_members = self._neighbours._n_points
                for member_index in range(n_members):
                    member = self._neighbours._get_member(member_index)

                    if _consider[member] == 0:
                        continue

                    self._neighbours_getter._get(
                        member,
                        input_data,
                        self._neighbour_neighbours,
                        cluster_params
                        )

                    weight = self._similarity_checker._get(
                        self._neighbours,
                        self._neighbour_neighbours,
                        cluster_params
                        )

                    self._priority_queue._push(b, member, weight)

    cpdef void _make_scipy_hierarchy(self, AINDEX n_points):
        """Build a SciPy-compatible linkage matrix Z from MST edges"""

        cdef stdvector[AINDEX] top_roots = []
        cdef AVALUE[:, ::1] Z = np.zeros((n_points - 1, 4), dtype=P_AVALUE)
        cdef AINDEX[::1] parents_indicator = np.arange(n_points * 2 - 1, dtype=P_AINDEX)

        self._make_scipy_hierarchy_inner(Z, parents_indicator, top_roots)
        self._artifacts["Z"] = np.array(Z)

    cdef void _make_scipy_hierarchy_inner(self, AVALUE[:, ::1] Z, AINDEX[::1] parents_indicator, stdvector[AINDEX] top_roots) nogil:
        """Build a SciPy-compatible linkage matrix Z from MST edges"""

        cdef AVALUE* _Z = &Z[0, 0]
        cdef AINDEX* _parents_indicator = &parents_indicator[0]
        cdef AINDEX i, _i, index, new_id, n_points, a, b, root_a, root_b
        cdef AVALUE weight, size_a, size_b
        cdef (AINDEX, AINDEX, AVALUE) edge

        n_points = Z.shape[0] + 1
        i = 0
        _i = 0
        while not self._priority_queue_tree._is_empty():

            edge = self._priority_queue_tree._pop()
            a = edge[0]
            b = edge[1]
            weight = edge[2]

            root_a = get_root(a, parents_indicator)
            root_b = get_root(b, parents_indicator)
            
            if root_a >= n_points:
                index = (root_a - n_points) * 4 + 3
                size_a = _Z[index]
            else:
                size_a = 1

            if root_b >= n_points:
                index = (root_b - n_points) * 4 + 3
                size_b = _Z[index]
            else:
                size_b = 1

            # Record the merge in Z
            _Z[_i] = min(root_a, root_b)
            _Z[_i + 1] = max(root_a, root_b)
            _Z[_i + 2] = weight
            _Z[_i + 3] = size_a + size_b

            new_id = n_points + i
            _parents_indicator[root_a] = new_id
            _parents_indicator[root_b] = new_id

            i += 1
            _i = i * 4

        # Merge disjoint components
        for index in range(n_points + i):
            if parents_indicator[index] == index:
                top_roots.push_back(index)

        if top_roots.size() > 1:
            root_a = top_roots.back()
            top_roots.pop_back()
            while top_roots.size() > 0:
                root_b =  top_roots.back()
                top_roots.pop_back()

                if root_a >= n_points:
                    index = (root_a - n_points) * 4 + 3
                    size_a = _Z[index]
                else:
                    size_a = 1
                    
                if root_b >= n_points:
                    index = (root_b - n_points) * 4 + 3
                    size_b = _Z[index]
                else:
                    size_b = 1

                # Record the merge in Z
                _Z[_i] = min(root_a, root_b)
                _Z[_i + 1] = max(root_a, root_b)
                _Z[_i + 2] = 0
                _Z[_i + 3] = size_a + size_b
                
                new_id = n_points + i
                parents_indicator[root_a] = new_id
                parents_indicator[root_b] = new_id
                root_a = new_id

                i += 1
                _i = i * 4

    def scipy_to_bundle_hierarchy(self, Bundle bundle, *, AVALUE[:, ::1] Z, AINDEX member_cutoff=10) -> None:
        HierarchicalFitterCommonNNMSTPrim.scipy_to_bundle_hierarchy(self, bundle, Z=Z, member_cutoff=member_cutoff)

    def _make_bundle_hierarchy(self, Bundle bundle, AINDEX n_points, AINDEX member_cutoff=10) -> None:
        HierarchicalFitterCommonNNMSTPrim._make_bundle_hierarchy(self, bundle, n_points, member_cutoff=member_cutoff)

HierarchicalFitter.register(HierarchicalFitterExtCommonNNMSTPrim)


class HierarchicalFitterRepeat(HierarchicalFitter):

    def __init__(
            self,
            fitter: Type["Fitter"]):
        self._fitter = fitter

    def __str__(self):

        attr_str = ", ".join([
            f"fitter={self._fitter}"
        ])

        return f"{type(self).__name__}({attr_str})"

    @classmethod
    def get_builder_kwargs(cls):
        return [
            ("fitter", None),
            ]

    def fit(
            self,
            Bundle bundle,
            *,
            radius_cutoffs,
            similarity_cutoffs,
            sort_by_size=True,
            member_cutoff=None,
            max_clusters=None,
            info=True,
            **kwargs) -> None:

        if not isinstance(radius_cutoffs, Iterable):
            radius_cutoffs = [radius_cutoffs]
        radius_cutoffs = [float(x) for x in radius_cutoffs]

        if not isinstance(similarity_cutoffs, Iterable):
            similarity_cutoffs = [similarity_cutoffs]
        similarity_cutoffs = [int(x) for x in similarity_cutoffs]

        if len(radius_cutoffs) == 1:
            radius_cutoffs *= len(similarity_cutoffs)

        if len(similarity_cutoffs) == 1:
            similarity_cutoffs *= len(radius_cutoffs)

        cdef AINDEX step, n_steps = len(radius_cutoffs)
        assert n_steps == len(similarity_cutoffs)

        cdef AINDEX n, n_points = bundle._input_data._n_points

        cdef Labels current_labels, previous_labels = Labels(
            np.ones(n_points, order="C", dtype=P_AINDEX)
            )
        cdef AINDEX c_label, p_label

        cdef stdumap[AINDEX, stdvector[AINDEX]] parent_labels_map
        cdef stdumap[AINDEX, stdvector[AINDEX]].iterator p_it
        cdef stdvector[AINDEX] labels_vector
        cdef Labels labels_container
        cdef AINDEX* _labels
        cdef AINDEX lvs
        cdef AINDEX lvi

        cdef dict terminal_bundles = {1: bundle}
        cdef dict new_terminal_bundles

        for step in range(n_steps):
            logger.info(
                f"Running step {step:<5} "
                f"(r = {radius_cutoffs[step]}, "
                f"c = {similarity_cutoffs[step]})"
            )

            new_terminal_bundles = {}

            current_labels = Labels(
                np.zeros(n_points, order="C", dtype=P_AINDEX)
                )

            cluster_params = self._fitter.make_parameters(
                radius_cutoff=radius_cutoffs[step],
                similarity_cutoff = similarity_cutoffs[step],
                **kwargs
                )

            self._fitter._fit(
                bundle._input_data,
                current_labels,
                cluster_params
                )

            if sort_by_size:
                current_labels.sort_by_size(member_cutoff, max_clusters)

            parent_labels_map.clear()

            for n in range(n_points):
                p_label = previous_labels._labels[n]
                if p_label == 0:
                    continue
                c_label = current_labels._labels[n]

                parent_labels_map[p_label].push_back(c_label)

            p_it = parent_labels_map.begin()
            while (p_it != parent_labels_map.end()):
                p_label = dereference(p_it).first

                parent_bundle = terminal_bundles[p_label]
                labels_vector =  dereference(p_it).second
                lvs = labels_vector.size()
                labels_container = Labels.from_length(lvs)
                _labels = &labels_container._labels[0]
                parent_bundle._labels = labels_container
                for lvi in range(lvs):
                    _labels[lvi] = labels_vector[lvi]

                if info:
                    params = {
                        k: (radius_cutoffs[step], similarity_cutoffs[step])
                        for k in parent_bundle._labels.to_set()
                        if k != 0
                    }
                    meta = parent_bundle._labels.meta
                    meta.update({
                        "params": params,
                        "reference": weakref.proxy(parent_bundle),
                        "origin": "fit"
                    })
                    parent_bundle._labels._meta = meta

                parent_bundle.isolate(isolate_input_data=False)

                for c_label, child_bundle in parent_bundle._children.items():
                    if c_label == 0:
                        continue
                    new_terminal_bundles[c_label] = child_bundle

                preincrement(p_it)

            terminal_bundles = new_terminal_bundles
            previous_labels = current_labels


class PredictorCommonNNFirstmatch(PredictorCommonNN):

    def __init__(
            self,
            neighbours_getter: Type["NeighboursGetter"],
            neighbours_getter_other: Type["NeighboursGetter"],
            neighbours: Type["Neighbours"],
            neighbour_neighbours: Type["Neighbours"],
            similarity_checker: Type["SimilarityChecker"]):
        self._neighbours_getter = neighbours_getter
        self._neighbours_getter_other = neighbours_getter_other
        self._neighbours = neighbours
        self._neighbour_neighbours = neighbour_neighbours
        self._similarity_checker = similarity_checker

    def __str__(self):

        attr_str = ", ".join([
            f"ngetter={self._neighbours_getter}",
            f"ngetter_other={self._neighbours_getter_other}",
            f"na={self._neighbours}",
            f"nb={self._neighbour_neighbours}",
            f"checker={self._similarity_checker}",
        ])

        return f"{type(self).__name__}({attr_str})"

    @classmethod
    def get_builder_kwargs(cls):
        return [
            ("neighbours_getter", None),
            ("neighbours_getter_other", "neighbours_getter"),
            ("neighbours", None),
            ("neighbour_neighbours","neighbours"),
            ("similarity_checker", None),
            ]

    def _predict(
            self,
            object input_data,
            object predictand_input_data,
            Labels labels,
            Labels predictand_labels,
            ClusterParameters cluster_params):
        """Generic cluster label prediction"""

        cdef AINDEX _support_cutoff = cluster_params.get_iparam(1)
        cdef AINDEX n, m, point, member, member_index, label

        cdef AINDEX* _labels = &labels._labels[0]
        cdef AINDEX* _predictand_labels = &predictand_labels._labels[0]
        cdef ABOOL* _consider = &predictand_labels._consider[0]
        cdef stduset[AINDEX] _consider_set = predictand_labels._consider_set

        n = predictand_input_data._n_points

        for point in range(n):
            if _consider[point] == 0:
                continue

            self._neighbours_getter_other.get_other(
                point,
                input_data,
                predictand_input_data,
                self._neighbours,
                cluster_params
            )

            if not self._neighbours.enough(_support_cutoff):
                continue

            m = self._neighbours._n_points
            for member_index in range(m):
                member = self._neighbours.get_member(member_index)
                label = _labels[member]

                if _consider_set.find(label) == _consider_set.end():
                    continue

                self._neighbours_getter.get(
                    member,
                    input_data,
                    self._neighbour_neighbours,
                    cluster_params
                    )

                if not self._neighbour_neighbours.enough(_support_cutoff):
                    continue

                if self._similarity_checker.check(
                        self._neighbours,
                        self._neighbour_neighbours,
                        cluster_params):
                    _consider[point] = 0
                    _predictand_labels[point] = label
                    break

        return

# TODO: Create union find and/or graph type
cdef inline AINDEX get_root(AINDEX p, AINDEX[::1] parent_indicator) except -1 nogil:
    cdef AINDEX parent
    parent = parent_indicator[p]
    while parent != p:
        p = parent
        parent = parent_indicator[p]
    return p