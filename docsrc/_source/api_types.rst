.. _sec_api_types:

_types
=======

This module provides a set of types that can be used as building block
in the aggregation of a :class:`~cnnclustering.cluster.Clustering` object.


Go to:

   * :ref:`Cluster parameters <sec_api_cnnclustering_types_cluster_params>`
   * :ref:`Cluster labels <sec_api_cnnclustering_types_labels>`
   * :ref:`Input data <sec_api_cnnclustering_types_input_data>`
   * :ref:`Neighbours <sec_api_cnnclustering_types_neighbours>`
   * :ref:`Neighbours getter <sec_api_cnnclustering_types_neighbours_getter>`
   * :ref:`Distance getter <sec_api_cnnclustering_types_distance_getter>`
   * :ref:`Metrics <sec_api_cnnclustering_types_metric>`
   * :ref:`Similarity Checker <sec_api_cnnclustering_types_similarity_checker>`
   * :ref:`Queues <sec_api_cnnclustering_types_queue>`


.. _sec_api_cnnclustering_types_cluster_params:

Cluster parameters
------------------

.. autoclass:: cnnclustering._types.ClusterParameters

   :members: to_dict


.. _sec_api_cnnclustering_types_labels:

Cluster labels
--------------

.. autoclass:: cnnclustering._types.Labels

   :members: from_sequence, sort_by_size


.. _sec_api_cnnclustering_types_input_data:

Input data
----------

Types used as input data to a clustering have to adhere to the input
data interface which is defined through
:class:`~cnnclustering._types.InputDataExtInterface` for Cython extension
types. For pure Python types the input data interface is defined through
the abstract base class :class:`~cnnclustering._types.InputDataInputData`
and the specialised abstract classes

   * :class:`~cnnclustering._types.InputData`
   * :class:`~cnnclustering._types.InputDataComponents`
   * :class:`~cnnclustering._types.InputDataPairwiseDistances`
   * :class:`~cnnclustering._types.InputDataPairwiseDistancesComputer`
   * :class:`~cnnclustering._types.InputDataNeighbourhoods`
   * :class:`~cnnclustering._types.InputDataNeighbourhoodsComputer`

|

.. autoclass:: cnnclustering._types.InputDataExtInterface
   :members:

.. autoclass:: cnnclustering._types.InputData
   :members:

.. autoclass:: cnnclustering._types.InputDataComponents
   :members:

.. autoclass:: cnnclustering._types.InputDataPairwiseDistances
   :members:

.. autoclass:: cnnclustering._types.InputDataPairwiseDistancesComputer
   :members:

.. autoclass:: cnnclustering._types.InputDataNeighbourhoods
   :members:

.. autoclass:: cnnclustering._types.InputDataNeighbourhoodsComputer
   :members:

.. autoclass:: cnnclustering._types.InputDataExtComponentsMemoryview
   :members:

.. autoclass:: cnnclustering._types.InputDataExtDistancesLinearMemoryview
   :members:

.. autoclass:: cnnclustering._types.InputDataExtNeighbourhoodsMemoryview
   :members:

.. autoclass:: cnnclustering._types.InputDataNeighbourhoodsSequence
   :members:

.. autoclass:: cnnclustering._types.InputDataSklearnKDTree
   :members:


.. _sec_api_cnnclustering_types_neighbours:

Neighbour containers
--------------------

.. autoclass:: cnnclustering._types.NeighboursExtInterface
   :members:

.. autoclass:: cnnclustering._types.Neighbours
   :members:

.. autoclass:: cnnclustering._types.NeighboursExtVector
   :members:

.. autoclass:: cnnclustering._types.NeighboursExtCPPSet
   :members:

.. autoclass:: cnnclustering._types.NeighboursExtCPPUnorderedSet
   :members:

.. autoclass:: cnnclustering._types.NeighboursExtVectorCPPUnorderedSet
   :members:

.. autoclass:: cnnclustering._types.NeighboursList
   :members:

.. autoclass:: cnnclustering._types.NeighboursSet
   :members:


.. _sec_api_cnnclustering_types_neighbours_getter:

Neighbours getter
-----------------

.. autoclass:: cnnclustering._types.NeighboursGetterExtInterface
   :members:

.. autoclass:: cnnclustering._types.NeighboursGetter
   :members:

.. autoclass:: cnnclustering._types.NeighboursGetterExtBruteForce
   :members:

.. autoclass:: cnnclustering._types.NeighboursGetterExtLookup
   :members:

.. autoclass:: cnnclustering._types.NeighboursGetterBruteForce
   :members:

.. autoclass:: cnnclustering._types.NeighboursGetterLookup
   :members:

.. autoclass:: cnnclustering._types.NeighboursGetterRecomputeLookup
   :members:


.. _sec_api_cnnclustering_types_distance_getter:

Distance getter
---------------

.. autoclass:: cnnclustering._types.DistanceGetterExtInterface
   :members:

.. autoclass:: cnnclustering._types.DistanceGetter
   :members:

.. autoclass:: cnnclustering._types.DistanceGetterExtMetric
   :members:

.. autoclass:: cnnclustering._types.DistanceGetterExtLookup
   :members:

.. autoclass:: cnnclustering._types.DistanceGetterMetric
   :members:

.. autoclass:: cnnclustering._types.DistanceGetterLookup
   :members:


.. _sec_api_cnnclustering_types_metric:

Metrics
-------

.. autoclass:: cnnclustering._types.MetricExtInterface
   :members:

.. autoclass:: cnnclustering._types.Metric
   :members:

.. autoclass:: cnnclustering._types.MetricExtDummy
   :members:

.. autoclass:: cnnclustering._types.MetricExtPrecomputed
   :members:

.. autoclass:: cnnclustering._types.MetricExtEuclidean
   :members:

.. autoclass:: cnnclustering._types.MetricExtEuclideanReduced
   :members:

.. autoclass:: cnnclustering._types.MetricExtEuclideanPeriodicReduced
   :members:

.. autoclass:: cnnclustering._types.MetricDummy
   :members:

.. autoclass:: cnnclustering._types.MetricEuclidean
   :members:

.. autoclass:: cnnclustering._types.MetricEuclideanReduced
   :members:


.. _sec_api_cnnclustering_types_similarity_checker:

Similarity checker
------------------

.. autoclass:: cnnclustering._types.SimilarityCheckerExtInterface
   :members:

.. autoclass:: cnnclustering._types.SimilarityChecker
   :members:

.. autoclass:: cnnclustering._types.SimilarityCheckerExtContains
   :members:

.. autoclass:: cnnclustering._types.SimilarityCheckerExtSwitchContains
   :members:

.. autoclass:: cnnclustering._types.SimilarityCheckerExtScreensorted
   :members:

.. autoclass:: cnnclustering._types.SimilarityCheckerContains
   :members:

.. autoclass:: cnnclustering._types.SimilarityCheckerSwitchContains
   :members:


.. _sec_api_cnnclustering_types_queue:


|

Queues
------

Queues can be optionally used by a fitter, e.g.

   * :class:`~cnnclustering._fit.FitterExtBFS`
   * :class:`~cnnclustering._fit.FitterBFS`

|

.. autoclass:: cnnclustering._types.QueueExtInterface
   :members:

|

.. autoclass:: cnnclustering._types.Queue
   :members:

|

.. autoclass:: cnnclustering._types.QueueExtLIFOVector
   :members:

|

.. autoclass:: cnnclustering._types.QueueExtFIFOQueue
   :members:

|

.. autoclass:: cnnclustering._types.QueueFIFODeque
   :members: