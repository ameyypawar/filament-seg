"""Solar Filament Segmentation Challenge 2026 (IEEE Big Data Cup).

Package layout:

* :mod:`filament_seg.config`      paths and dataset constants
* :mod:`filament_seg.rle`         mask <-> COCO RLE <-> submission CSV
* :mod:`filament_seg.metrics`     local Panoptic Quality + rubric diagnostics
* :mod:`filament_seg.data`        MAGFiLO annotations and leak-free splits
* :mod:`filament_seg.disk`        solar disk detection, limb-darkening removal
* :mod:`filament_seg.postprocess` binary mask -> filament instances
* :mod:`filament_seg.baseline`    model-free detector (pipeline validation)
"""

__version__ = "0.1.0"
