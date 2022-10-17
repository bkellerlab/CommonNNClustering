.. _sec_api_cluster:

.. module:: cnnclustering

cluster
=======

The functionality of this module is primarily exposed and bundled by the
:class:`~cnnclustering.cluster.Clustering` class. An instance of this
class aggregates various types (defined in :mod:`~cnnclustering._types`
:ref:`here <sec_api_types>`).


Go to:

   * :ref:`Clustering <sec_api_cnnclustering_cluster_Clustering>`
   * :ref:`Records and summary <sec_api_cnnclustering_cluster_records>`


.. _sec_api_cnnclustering_cluster_Clustering:

Clustering
----------

.. autoclass:: cnnclustering.cluster.Clustering
   :members:

.. autoclass:: cnnclustering.cluster.ClusteringBuilder
   :members:


.. _sec_api_cnnclustering_cluster_records:

Records and summary
-------------------

.. autoclass:: cnnclustering.cluster.Record
   :members:

.. autoclass:: cnnclustering.cluster.Summary
   :members:

.. autofunction:: cnnclustering.cluster.make_typed_DataFrame

.. autofunction:: cnnclustering.cluster.timed
