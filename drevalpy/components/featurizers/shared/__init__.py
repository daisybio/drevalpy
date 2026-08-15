"""Featurizers whose implementation is identical on both entity sides.

Each module here declares one implementation and binds it to the cell-line and
drug registries with ``register_for_sides``; the generated per-side subclasses are
injected into the defining module's namespace. ``register_native_components`` in
``drevalpy/registry/_builtins.py`` scans this directory once, not once per side.
"""
