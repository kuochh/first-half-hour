"""Shared test setup: the official-close size and hash check is for the real CSV only."""

import pytest

from first_half_hour import config


@pytest.fixture(autouse=True, scope="session")
def _stub_official_close():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "OFFICIAL_CLOSE_VERIFY", False)
        yield
