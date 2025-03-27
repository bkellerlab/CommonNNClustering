Summary of changes
==================

v0.0.4
------

  * General performance improvements by declaring `cdef` functions `noexcept`
  * Added convenience function `recipes.sorted_neighbourhoods_from_coordinates`
  * Improved "automatic" hierarchical clustering (PR:)
    * Added Cython extension type `HierarchicalFitterExtCommonNNMSTPrim`
    * Added recipes:
       * `"coordinates_mst/_debug"`
       * `"sorted_neighbourhoods_mst/_debug"`
    * Added convenience funtionality for hierarchical cluster result inspection
  * Fixed processing of `CFLAGS` in `setup.py` to allow install on Windows (should also help with compilation on Mac)

v0.0.3
------

Final release after completed migration from [janjoswig/CommonNNClustering](https://github.com/janjoswig/CommonNNClustering). No changes tracked yet.