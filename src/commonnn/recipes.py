from collections.abc import Iterable

import numpy as np

from commonnn._primitive_types import P_AINDEX, P_AVALUE
from commonnn import _types, _fit


def prepare_pass(data):
    """Dummy preparation hook

    Use if no preparation of input data is desired.

    Args:
        data: Input data that should be prepared.

    Returns:
        (data,), {}
    """
    return (data,), {}


def prepare_to_array(data, **kwargs):
    return (np.array(data, **kwargs),), {}


def prepare_square_array_to_linear(data):
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    indices = np.triu_indices_from(data, k=1)
    data = data[indices]
    return (data,), {}


def prepare_components_array_from_parts(data):
    r"""Prepare input data points

    Use when point components are passed as sequence of parts, e.g. as

        >>> input_data, meta = prepare_points_parts([[[0, 0],
        ...                                           [1, 1]],
        ...                                          [[2, 2],
        ...                                           [3,3]]])
        >>> input_data
        array([[0, 0],
               [1, 1],
               [2, 2],
               [3, 3]])
        >>> meta
        {"edges": [2, 2]}

    Recognised data formats are:
        * Sequence of length *d*:
            interpreted as 1 point with *d* components.
        * 2D Sequence (sequence of sequences all of same length) with
            length *n* (rows) and width *d* (columns):
            interpreted as *n* points with *d* components.
        * Sequence of 2D sequences all of same width:
            interpreted as parts (groups) of points.

    The returned input data format is compatible with:
        * `commonnn._types.InputDataExtPointsMemoryview`

    Args:
        data: Input data that should be prepared.

    Returns:
        * Formatted input data (NumPy array of shape
            :math:`\sum n_\mathrm{part}, d`)
        * Dictionary of meta-information

    Notes:
        Does not catch deeper nested formats.
    """

    try:
        d1 = len(data)
    except TypeError as error:
        raise error

    finished = False

    all_d2_equal = all_d3_equal = False
    if d1 == 0:
        # Empty sequence
        data = [np.array([[]])]
        finished = True

    if not finished:
        try:
            d2 = [len(x) for x in data]
            all_d2_equal = (len(set(d2)) == 1)
        except TypeError:
            # 1D Sequence
            data = [np.array([data])]
            finished = True

    if not finished:
        try:
            d3 = [len(y) for x in data for y in x]
            all_d3_equal = (len(set(d3)) == 1)
        except TypeError:
            if not all_d2_equal:
                raise ValueError(
                    "Dimension mismatch"
                )
            # 2D Sequence of sequences of same length
            data = [np.asarray(data)]
            finished = True

    if not finished:
        if not all_d3_equal:
            raise ValueError(
                "Dimension mismatch"
            )
        # Sequence of 2D sequences of same width
        data = [np.asarray(x) for x in data]
        finished = True

    meta = {}

    meta["edges"] = [x.shape[0] for x in data]

    data_args = (np.asarray(np.vstack(data), order="C", dtype=P_AVALUE),)
    data_kwargs = {"meta": meta}

    return data_args, data_kwargs


def prepare_padded_neighbourhoods_array(data):
    """Prepare neighbourhood information by padding

    Args:
        data: Expects a sequence of sequences with neighbour indices.

    Returns:
        Data as a 2D NumPy array of shape (#points, max. number of neighbours)
        and a 1D array with the actual number of neighbours for each point (data
        args). Also returns meta information (data kwargs).
    """

    n_neighbours = [len(s) for s in data]
    pad_to = max(n_neighbours)

    data = [
        np.pad(a, (0, pad_to - n_neighbours[i]), mode="constant", constant_values=0)
        for i, a in enumerate(data)
    ]

    meta = {}

    data_args = (
        np.asarray(data, order="C", dtype=P_AINDEX),
        np.asarray(n_neighbours, dtype=P_AINDEX)
    )

    data_kwargs = {"meta": meta}

    return data_args, data_kwargs


class Builder:

    default_recipe = "coordinates"
    default_preparation_hook = staticmethod(prepare_pass)

    def __init__(self, recipe=None, **kwargs):

        if recipe is None:
            recipe = self.default_recipe

        if isinstance(recipe, str):
            recipe = get_registered_recipe(recipe)

        self.recipe = recipe
        self.recipe.update(kwargs)

        self.recipe = {
            k.replace("__", "."): v
            for k, v in self.recipe.items()
        }

        self.recipe = {
            ".".join(
                COMPONENT_ALT_KW_MAP.get(kw, kw)
                for kw in k.split(".")
            ): v
            for k, v in self.recipe.items()
        }

    def make_component(
            self, component_kw, alternative=None, prev_kw=None):
        if prev_kw is None:
            prev_kw = []

        full_kw = ".".join([*prev_kw, component_kw])

        if alternative is not None:
            alternative = full_kw.rsplit(".", 1)[0] + f".{alternative}"

        if full_kw in self.recipe:
            component_details = self.recipe[full_kw]
        elif (alternative is not None) & (alternative in self.recipe):
            component_details = self.recipe[alternative]
        else:
            return object

        args = ()
        kwargs = {}

        _component_kw = COMPONENT_KW_TYPE_ALIAS_MAP.get(
            component_kw, component_kw
        )

        if component_details is None:
            return None

        if isinstance(component_details, str):
            component_type = COMPONENT_NAME_TYPE_MAP[_component_kw][
                component_details
            ]

        elif isinstance(component_details, tuple):
            component_type, args, kwargs = component_details
            if isinstance(component_type, str):
                component_type = COMPONENT_NAME_TYPE_MAP[_component_kw][
                    component_type
                ]

        else:
            component_type = component_details

        if hasattr(component_type, "get_builder_kwargs"):
            for component_kw, alternative in (
                    component_type.get_builder_kwargs()):
                component = self.make_component(
                    component_kw, alternative, prev_kw=full_kw.split(".")
                )
                if component is not object:
                    kwargs[component_kw] = component

        return component_type(*args, **kwargs)

    def make_input_data(self, data, preparation_hook=None, details=None):

        if preparation_hook is None:
            preparation_hook = self.recipe.get(
                "preparation_hook", self.default_preparation_hook
            )

        if isinstance(preparation_hook, str):
            preparation_hook = COMPONENT_NAME_TYPE_MAP["preparation_hook"][
                preparation_hook
            ]

        if details is None:
            try:
                details = self.recipe["input_data"]
            except KeyError:
                raise LookupError("Input data type details not found")

        args = ()
        kwargs = {}

        if isinstance(details, str):
            component_type = COMPONENT_NAME_TYPE_MAP["input_data"][details]

        elif isinstance(details, tuple):
            component_type, args, kwargs = details
            if isinstance(component_type, str):
                component_type = COMPONENT_NAME_TYPE_MAP["input_data"][component_type]

        else:
            component_type = details

        data_args, data_kwargs = preparation_hook(data)
        args = (*data_args, *args)

        if "meta" not in data_kwargs:
            data_kwargs["meta"] = {}
        data_kwargs["meta"].update(kwargs.get("meta", {}))

        kwargs.update(data_kwargs)

        # TODO: Remove no cover pragma if an input data type with builder kwargs is implemented
        if hasattr(component_type, "get_builder_kwargs"):
            for component_kw, alternative in (
                    component_type.get_builder_kwargs()):
                component = self.make_component(  # pragma: no cover
                    component_kw, alternative, prev_kw=["input_data"]
                )
                if component is not object:  # pragma: no cover
                    kwargs[component_kw] = component

        return component_type(*args, **kwargs)


REGISTERED_RECIPES = {
    "none": {},
    "coordinates": {
        "input_data": "components_mview",
        "preparation_hook": "components_array_from_parts",
        "fitter": "bfs",
        "fitter.ngetter": "brute_force",
        "fitter.na": "vuset",
        "fitter.checker": "switch",
        "fitter.queue": "fifo",
        "fitter.ngetter.dgetter": "metric",
        "fitter.ngetter.dgetter.metric": "euclidean_r",
    },
    "distances": {
        "input_data": "components_mview",
        "preparation_hook": "components_array_from_parts",
        "fitter": "bfs",
        "fitter.ngetter": "brute_force",
        "fitter.na": "vuset",
        "fitter.checker": "switch",
        "fitter.queue": "fifo",
        "fitter.ngetter.dgetter": "metric",
        "fitter.ngetter.dgetter.metric": "precomputed",
    },
    "neighbourhoods": {
        "input_data": "neighbourhoods_mview",
        "preparation_hook": "padded_neighbourhoods_array",
        "fitter": "bfs",
        "fitter.ngetter": "lookup",
        "fitter.na": "vuset",
        "fitter.checker": "switch",
        "fitter.queue": "fifo",
    },
    "sorted_neighbourhoods": {
        "input_data": "neighbourhoods_mview",
        "preparation_hook": "padded_neighbourhoods_array",
        "fitter": "bfs",
        "fitter.ngetter": ("lookup", (), {"is_sorted": True}),
        "fitter.na": "vector",
        "fitter.checker": "screen",
        "fitter.queue": "fifo",
    },
    "coordinates_mst": {
        "input_data": "components_mview",
        "preparation_hook": "components_array_from_parts",
        "hfitter": "mst",
        "hfitter.ngetter": "brute_force",
        "hfitter.na": "vuset",
        "hfitter.checker": "switch",
        "hfitter.prioq": "maxheap",
        "hfitter.ngetter.dgetter": "metric",
        "hfitter.ngetter.dgetter.metric": "euclidean_r",
    },
}


def get_registered_recipe(key):
    return {k: v for k, v in REGISTERED_RECIPES[key.lower()].items()}


# Provides alternative (short) identifiers for types
COMPONENT_ALT_KW_MAP = {
    "prep": "preparation_hook",
    "input": "input_data",
    "data": "input_data",
    "n": "neighbours",
    "na": "neighbours",
    "nb": "neighbour_neighbours",
    "getter": "neighbours_getter",
    "ogetter": "neighbours_getter_other",
    "ngetter": "neighbours_getter",
    "ongetter": "neighbours_getter_other",
    "dgetter": "distance_getter",
    "checker": "similarity_checker",
    "q": "queue",
    "pq": "priority_queue",
    "prioq": "priority_queue",
    "pqt": "priority_queue_tree",
    "prioq_tree": "priority_queue_tree",
    "hfitter": "hierarchical_fitter"
}

# Provides alternative equivalent types
COMPONENT_KW_TYPE_ALIAS_MAP = {
    "neighbour_neighbours": "neighbours",
    "neighbours_getter_other": "neighbours_getter",
    "priority_queue_tree": "priority_queue"
}

# Provides name identifiers for types
COMPONENT_NAME_TYPE_MAP = {
    "preparation_hook": {
        "pass": prepare_pass,
        "components_array_from_parts": prepare_components_array_from_parts,
        "padded_neighbourhoods_array": prepare_padded_neighbourhoods_array,
        "array": prepare_to_array,
    },
    "input_data": {
        "components_mview": _types.InputDataExtComponentsMemoryview,
        "distances_mview": _types.InputDataExtDistancesMemoryview,
        "neighbourhoods_mview": _types.InputDataExtNeighbourhoodsMemoryview
    },
    "neighbours_getter": {
        "brute_force": _types.NeighboursGetterExtBruteForce,
        "lookup": _types.NeighboursGetterExtLookup,
    },
    "distance_getter": {
        "metric": _types.DistanceGetterExtMetric,
        "lookup": _types.DistanceGetterExtLookup,
    },
    "neighbours": {
        "vector": _types.NeighboursExtVector,
        "uset": _types.NeighboursExtUnorderedSet,
        "vuset": _types.NeighboursExtVectorUnorderedSet,
    },
    "metric": {
        "dummy": _types.MetricExtDummy,
        "precomputed": _types.MetricExtPrecomputed,
        "euclidean": _types.MetricExtEuclidean,
        "euclidean_r": _types.MetricExtEuclideanReduced,
        "euclidean_periodic_r": _types.MetricExtEuclideanPeriodicReduced,
        "euclidean_reduced": _types.MetricExtEuclideanReduced,
        "euclidean_periodic_reduced": _types.MetricExtEuclideanPeriodicReduced,
    },
    "similarity_checker": {
        "contains": _types.SimilarityCheckerExtContains,
        "switch": _types.SimilarityCheckerExtSwitchContains,
        "screen": _types.SimilarityCheckerExtScreensorted,
    },
    "queue": {
        "fifo": _types.QueueExtFIFOQueue
    },
    "priority_queue": {
        "maxheap": _types.PriorityQueueMaxHeap
    },
    "fitter": {
        "bfs": _fit.FitterExtCommonNNBFS,
        "bfs_debug": _fit.FitterCommonNNBFSDebug
    },
    "hierarchical_fitter": {
        "repeat": _fit.HierarchicalFitterRepeat,
        "mst": _fit.HierarchicalFitterCommonNNMSTPrim,
    },
    "predictor": {
        "firstmatch": _fit.PredictorCommonNNFirstmatch
    }
}
