Summary of changes
==================

v0.0.5 (dev)
------------

 * Add `Clustering.to_dlabels()` and `Clustering.to_dmapping()` convenience functions

v0.0.4
------

  * Improved "automatic" hierarchical clustering (PR: 1)
    * Added Cython extension type `HierarchicalFitterExtCommonNNMSTPrim`
    * Added recipes:
       * `"coordinates_mst/_debug"`
       * `"sorted_neighbourhoods_mst/_debug"`
    * Reviewed bundle hierarchy building via SciPy Z-matrix
    * Added supporting functionality in the `_bundle` module
  * Annotation option in `cluster.Clustering.tree`
  * General performance improvements by declaring `cdef` functions `noexcept`
  * Added convenience function `recipes.sorted_neighbourhoods_from_coordinates`
  * Fixed processing of `CFLAGS` in `setup.py` to allow install on Windows (should also help with compilation on Mac)
  * Added content related to fully hierarchical clustering to the docs (scikit-learn data, MD example)

v0.0.3
------

Final release after completed migration from [janjoswig/CommonNNClustering](https://github.com/janjoswig/CommonNNClustering). No changes tracked yet.