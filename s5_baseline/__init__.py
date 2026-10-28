"""Sequence-regression adapter around the official S5 implementation.

The S5 core (JAX/Flax) is used unmodified from the upstream repository at a
pinned commit; see ``official_imports.py`` and ``scripts/fetch_official_s5.sh``.
Local code is limited to the regression adapter, data views, the optimizer
recipe, training, and selection utilities.
"""
