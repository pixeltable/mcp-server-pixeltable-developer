"""Schema-type resolver for `pixeltable_create_table`.

The resolver must:
  - accept all base types (string + dict forms, case-insensitive)
  - accept Required[T] / Array[T] / Required[Array[T]]
  - reject anything else (no silent fallback to String)
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:Field name \"schema\".*shadows.*:UserWarning"
)


def _resolver():
    from mcp_server_pixeltable_stio.core.tables import _resolve_pxt_type

    return _resolve_pxt_type


@pytest.mark.parametrize(
    "spec",
    [
        "String", "string", "Int", "int", "Float", "Bool", "Json", "Image",
        "Video", "Audio", "Document", "Timestamp", "Date",
    ],
)
def test_base_types(spec):
    import pixeltable as pxt

    resolved = _resolver()(spec)
    # The annotated type carries its base in pxt.<canonical>; just confirm we got something truthy.
    assert resolved is not None
    # Round-trip through pxt.<name> (case-insensitive) so we know the resolver picked the right one.
    canonical = spec[:1].upper() + spec[1:].lower() if spec[0].islower() else spec
    assert resolved == getattr(pxt, canonical)


def test_required_wrapper():
    import pixeltable as pxt

    assert _resolver()("Required[String]") == pxt.Required[pxt.String]


def test_array_wrapper():
    import pixeltable as pxt

    assert _resolver()("Array[Float]") == pxt.Array[pxt.Float]


def test_required_array():
    import pixeltable as pxt

    assert _resolver()("Required[Array[Int]]") == pxt.Required[pxt.Array[pxt.Int]]


def test_dict_form_basic():
    import pixeltable as pxt

    assert _resolver()({"type": "String"}) == pxt.String


def test_dict_form_required_array():
    import pixeltable as pxt

    spec = {"type": "Array", "element_type": "Float", "required": True}
    assert _resolver()(spec) == pxt.Required[pxt.Array[pxt.Float]]


@pytest.mark.parametrize("spec", ["Pixel", "Required[Mystery]", "Array[NotAType]", ""])
def test_unknown_raises(spec):
    with pytest.raises(ValueError):
        _resolver()(spec)


def test_none_raises():
    with pytest.raises(ValueError):
        _resolver()(None)
