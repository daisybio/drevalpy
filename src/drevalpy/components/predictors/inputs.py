"""What a predictor reads from each side's featurizer: nothing, a matrix, or named blocks.

A predictor declares ``sample_inputs`` and ``drug_inputs``, each an :class:`Inputs`:

* ``Inputs.none()`` - the predictor uses no features on that side (feature-free);
* ``Inputs.matrix()`` - any numeric output, flattened into one design matrix;
* ``Inputs.blocks(*specs)`` - the featurizer must emit every spec; a spec is a
  :class:`~drevalpy.types.data.batch.feature_block.BlockSpec` or a tuple of
  alternative specs of which one must be emitted. ``Inputs.blocks()`` with no spec
  takes the featurizer's blocks as they are.

:func:`predictor_inputs` reads the declaration of a predictor class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from drevalpy.types.data.batch.feature_block import BlockSpec
from drevalpy.types.data.feature_source import Side

InputSpec = BlockSpec | tuple[BlockSpec, ...]


@dataclass(frozen=True)
class Inputs:
    """One side's input declaration of a predictor."""

    kind: Literal["none", "matrix", "blocks"]
    #: Required blocks of a ``"blocks"`` declaration; empty for the other kinds.
    specs: tuple[InputSpec, ...] = ()

    @classmethod
    def none(cls) -> Inputs:
        """Declare that the side uses no features.

        :returns: Feature-free declaration.
        """
        return cls("none")

    @classmethod
    def matrix(cls) -> Inputs:
        """Declare that the side's numeric output is read as one design matrix.

        :returns: Matrix declaration.
        """
        return cls("matrix")

    @classmethod
    def blocks(cls, *specs: InputSpec) -> Inputs:
        r"""Declare the named blocks the side's featurizer must emit.

        :param \*specs: Required blocks; a tuple of specs means one of them.
        :returns: Block declaration.
        """
        return cls("blocks", tuple(specs))


def predictor_inputs(cls: type[Any], side: Side) -> Inputs:
    """Return what predictor class *cls* reads from *side*.

    :param cls: Predictor class.
    :param side: Entity side.
    :returns: The class's ``sample_inputs`` / ``drug_inputs`` declaration.
    :raises TypeError: If *cls* does not declare that side's inputs as an :class:`Inputs`.
    """
    attr = f"{side}_inputs"
    declared = getattr(cls, attr, None)
    if not isinstance(declared, Inputs):
        msg = f"{cls.__name__}.{attr} must be an Inputs (Inputs.none(), .matrix() or .blocks(...)), got {declared!r}"
        raise TypeError(msg)
    return declared
