"""Tests for pyleoclim.utils.plotting"""

import matplotlib.pyplot as plt
import pytest

from pyleoclim.utils import plotting


class TestUtilsPlottingSavefig:
    @pytest.mark.parametrize("name, settings, expected", [
        ("fig", {}, "fig.pdf"),                       # no suffix, no format: pdf (#730)
        ("fig", {"format": "png"}, "fig.png"),        # no suffix: follow "format" (#730)
        ("fig.png", {}, "fig.png"),                   # suffix kept
        ("fig.svg", {}, "fig.svg"),                   # any format matplotlib supports is kept
        ("fig.v2", {}, "fig.v2.pdf"),                 # unrecognized suffix: a suffix is added
    ])
    def test_savefig_t0(self, tmp_path, capsys, name, settings, expected):
        ''' the file is saved where the message says it is, and nowhere else '''
        fig, ax = plt.subplots()
        plotting.savefig(fig, path=tmp_path / name, settings=settings)
        assert [p.name for p in tmp_path.iterdir()] == [expected]
        assert f'"{tmp_path / expected}"' in capsys.readouterr().out

    def test_savefig_t1(self, tmp_path):
        ''' path given in settings, inside a directory that doesn't exist yet '''
        fig, ax = plt.subplots()
        plotting.savefig(fig, settings={"path": tmp_path / "new" / "fig"}, verbose=False)
        assert (tmp_path / "new" / "fig.pdf").exists()

    def test_savefig_t2(self):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError):
            plotting.savefig(fig)
        plt.close(fig)
