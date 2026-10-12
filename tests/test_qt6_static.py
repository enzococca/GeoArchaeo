"""Il codice del plugin deve superare il controllo statico Qt6 del repository QGIS."""
import os
import sys

import pytest

pytest.importorskip('qgis.core')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qt6_enum_check  # noqa: E402


def test_no_qt6_enum_issues(capsys):
    total = qt6_enum_check.main()
    assert total == 0, capsys.readouterr().out
