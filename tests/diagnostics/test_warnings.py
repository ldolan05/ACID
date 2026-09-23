import warnings


import ACID_code
from ACID_code.diagnostics.warnings import ACIDDroppedDataWarning, ACIDWarning


def test_acid_warning_categories_are_public_and_filterable():
    assert issubclass(ACIDDroppedDataWarning, ACIDWarning)
    assert ACID_code.ACIDWarning is ACIDWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ACIDWarning)
        warnings.warn("problematic input", ACIDDroppedDataWarning)

    assert len(caught) == 1
    assert caught[0].category is ACIDDroppedDataWarning


def test_acid_warnings_use_default_filter():
    # Pytest deliberately replaces filters installed during package import, so
    # exercise ACID's configuration function in a locally scoped filter list.
    with warnings.catch_warnings():
        ACID_code.configure_warnings()
        assert any(
            action == "default" and category is ACIDWarning
            for action, _message, category, _module, _lineno in warnings.filters
        )

