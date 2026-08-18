"""Built-in stages, split by family.

Importing this package imports every family module, which is what registers the
stages under ``kind="stage"``. :meth:`segmentator.pipeline.Pipeline.from_config`
imports it once for exactly that side effect.
"""

from . import (  # noqa: F401
    color,
    keypoints,
    mask,
    motion,
    preprocess,
    structure,
    texture,
)
