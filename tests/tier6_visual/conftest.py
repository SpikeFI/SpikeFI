"""Tier 6 setup: forces the non-interactive Agg backend before pyplot is
ever touched, and closes every figure after each test so the tier never
opens a window and never leaks figures across tests.
"""


from collections.abc import Generator

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _close_all_figures() -> Generator[None, None, None]:
    """Closes every open figure once a test finishes, regardless of how
    many plotting calls it made, so figures never accumulate across tests."""
    yield
    plt.close('all')
