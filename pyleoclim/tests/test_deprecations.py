''' Tripwires for scheduled deprecations.

Each test here passes while a deprecation is still within its grace period, and
fails as soon as the package version reaches the release in which it must be
completed. The failure message is the checklist for that release, so the work
cannot slip through unnoticed. Delete a test once its checklist is done.

Naming conventions follow test_core_*.py.
'''

import pathlib
import tomllib

import pyleoclim as pyleo
import pyleoclim.utils.tsbase as tsbase
from packaging.version import Version


def _source_version():
    '''Version declared in pyproject.toml, falling back to the installed metadata.

    The source tree is authoritative: an editable install keeps the version it was
    installed with, so its metadata can lag behind a version bump.
    '''
    pyproject = pathlib.Path(__file__).parents[2] / 'pyproject.toml'
    if pyproject.exists():
        with open(pyproject, 'rb') as f:
            return Version(tomllib.load(f)['project']['version'])
    return Version(pyleo.__version__)


def _reached(version, target):
    '''True once version is target or later, counting target's pre-releases
    (2.0.0b0 already counts as 2.0.0), so the work lands before the final release.'''
    return version.release >= Version(target).release


IMPLICIT_COMMON_TIME_CHECKLIST = '''
Pyleoclim {target}: implicit common_time() alignment must now raise (#697, #699-#701).

  1. series.py, _conform_pair(): when common_time_kwargs is None and the axes differ,
     raise ValueError (naming common_time_kwargs={{}} and
     MultipleSeries([ts1, ts2]).common_time() as the fixes) instead of warning.
  2. multipleseries.py MultipleSeries.correlation() and ensembleseries.py
     EnsembleSeries.correlation(): turn the warn-once check before the loop into a
     raise-once check. Also narrow the bare `except:` in MultipleSeries.correlation(),
     which would otherwise turn every mismatched pair into a silent NaN.
  3. tsbase.py: delete warn_implicit_common_time(), IMPLICIT_COMMON_TIME_REMOVAL,
     and DataConformationWarning (class and __all__ entry).
  4. Docstrings: rewrite the `common_time_kwargs` entry in Series.correlation(),
     .causality(), .wavelet_coherence(), MultipleSeries.correlation() and
     EnsembleSeries.correlation() to say None raises when the axes differ.
  5. Tests: turn the *_implicit_alignment_warns* tests (test_core_Series.py,
     test_core_MultipleSeries.py, test_core_EnsembleSeries.py) into pytest.raises
     tests; add common_time_kwargs={{}} to the TestMultipleSeriesCorrelation tests
     that rely on implicit alignment (test_correlation_t1 and _t2).
  6. Release notes: list it as a breaking change. Then delete this test.
'''


def test_implicit_common_time_deprecation_expired():
    target = tsbase.IMPLICIT_COMMON_TIME_REMOVAL
    version = _source_version()
    assert not _reached(version, target), (
        f'Version is {version}.' + IMPLICIT_COMMON_TIME_CHECKLIST.format(target=target))


def test_reached_counts_prereleases():
    assert _reached(Version('2.0.0b0'), '2.0.0')
    assert _reached(Version('2.1.0'), '2.0.0')
    assert not _reached(Version('1.9.9'), '2.0.0')
    assert not _reached(Version('1.3.2b0'), '2.0.0')
