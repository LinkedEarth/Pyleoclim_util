"""Tests for how pyleoclim objects display themselves (repr, str, and Jupyter's _repr_html_)

Displaying an object must return text, never print it as a side effect (#685).
"""

import pytest
import pyleoclim as pyleo


class TestSeriesDisplay:
    def test_repr_t0(self, pinkseries, capsys):
        ''' repr returns the metadata and data, and prints nothing '''
        r = repr(pinkseries)
        assert capsys.readouterr().out == ''
        assert not r.startswith('None')
        assert pinkseries.label in r

    def test_repr_html_t0(self, pinkseries, capsys):
        h = pinkseries._repr_html_()
        assert capsys.readouterr().out == ''
        assert 'pyleoclim.Series' in h
        assert '<table' in h
        assert pinkseries.label in h

    def test_repr_html_t1(self, pinkgeoseries):
        ''' GeoSeries show their coordinates '''
        h = pinkgeoseries._repr_html_()
        assert 'pyleoclim.GeoSeries' in h
        for key in ['lat', 'lon']:
            assert f'>{key}<' in h

    def test_repr_html_t2(self, pinkseries):
        ''' metadata are HTML-escaped '''
        ts = pinkseries.copy()
        ts.label = '<b>x & y</b>'
        h = ts._repr_html_()
        assert '&lt;b&gt;x &amp; y&lt;/b&gt;' in h


class TestMultipleSeriesDisplay:
    def test_repr_html_t0(self, pinkseries, capsys):
        ts2 = pinkseries.copy()
        ts2.label = 'other'
        ms = pyleo.MultipleSeries([pinkseries, ts2], label='pair')
        h = ms._repr_html_()
        assert capsys.readouterr().out == ''
        assert 'pyleoclim.MultipleSeries' in h
        assert 'pair' in h
        assert '(2 series)' in h

    def test_to_pandas_t0(self, pinkseries):
        ''' series sharing a label are all kept '''
        ens = pyleo.EnsembleSeries([pinkseries.copy() for _ in range(3)])
        assert ens.to_pandas().shape[1] == 3
        assert ens.to_pandas(paleo_style=True).shape[1] == 3

    def test_to_pandas_t1(self, pinkseries):
        ''' duplicate labels get numbered suffixes; unique labels are left alone '''
        ts2 = pinkseries.copy()
        ts2.label = 'other'
        ms = pyleo.MultipleSeries([pinkseries, pinkseries.copy(), ts2])
        assert list(ms.to_pandas().columns) == ['pink noise_0', 'pink noise_1', 'other']

    def test_to_pandas_t2(self, pinkseries):
        ''' paleo_style index name includes the time unit, as for a Series '''
        ms = pyleo.MultipleSeries([pinkseries, pinkseries.copy()])
        assert ms.to_pandas(paleo_style=True).index.name == pinkseries.to_pandas(paleo_style=True).index.name


class TestEnsembleSeriesDisplay:
    def test_repr_t0(self, pinkseries):
        ''' small ensembles show one column per member '''
        ens = pyleo.EnsembleSeries([pinkseries.copy() for _ in range(3)])
        assert 'pink noise_2' in repr(ens)
        assert 'pink noise_2' in ens._repr_html_()

    def test_repr_t1(self, pinkseries, capsys):
        ''' large ensembles show a summary with quantiles '''
        ens = pyleo.EnsembleSeries([pinkseries.copy() for _ in range(10)])
        r, h = repr(ens), ens._repr_html_()
        assert capsys.readouterr().out == ''
        for out in (r, h):
            assert '50%' in out and 'pink noise_9' not in out
        assert '10 members' in h

    def test_repr_t2(self, pinkseries):
        ''' quantiles come from quantiles(): along time for age ensembles, omitted if neither axis is shared '''
        ages = [pinkseries.copy() for _ in range(10)]
        for i, ts in enumerate(ages):
            ts.time = ts.time + 0.1 * i
        ens = pyleo.EnsembleSeries(ages)
        assert 'Time quantiles' in repr(ens)
        ens.series_list[1].value = ens.series_list[1].value + 1
        r, h = repr(ens), ens._repr_html_()
        assert 'Quantiles not shown' in r and 'Quantiles not shown' in h


class TestResultsDisplay:
    def test_corr_str_t0(self, capsys):
        soi = pyleo.utils.load_dataset('SOI')
        nino = pyleo.utils.load_dataset('NINO3')
        corr = soi.correlation(nino, number=2, common_time_kwargs={})
        capsys.readouterr()
        s = str(corr)
        assert capsys.readouterr().out == ''
        assert 'p-value' in s

    def test_correns_str_t0(self, pinkseries, capsys):
        ens = pyleo.EnsembleSeries([pinkseries.copy() for _ in range(3)])
        corr = ens.correlation(pinkseries, number=2)
        capsys.readouterr()
        s = str(corr)
        assert capsys.readouterr().out == ''
        assert 'p-value' in s and 'Ensemble size: 3' in s

    def test_psd_str_t0(self, pinkseries, capsys):
        psd = pinkseries.interp().spectral(method='mtm')
        capsys.readouterr()
        s = str(psd)
        assert capsys.readouterr().out == ''
        assert 'Frequency' in s and 'Length:' in s

    def test_scalogram_str_t0(self, pinkseries, capsys):
        scal = pinkseries.interp().wavelet()
        capsys.readouterr()
        s = str(scal)
        assert capsys.readouterr().out == ''
        assert 'frequency' in s and 'amplitude' in s

    def test_corr_repr_t0(self):
        soi = pyleo.utils.load_dataset('SOI')
        nino = pyleo.utils.load_dataset('NINO3')
        corr = soi.correlation(nino, number=2, common_time_kwargs={})
        assert repr(corr).startswith('pyleoclim.Corr')
        assert 'pyleoclim.Corr' in corr._repr_html_() and '<table' in corr._repr_html_()

    def test_correns_str_t1(self, pinkseries):
        ''' large ensembles are truncated '''
        ens = pyleo.EnsembleSeries([pinkseries.copy() for _ in range(15)])
        corr = ens.correlation(pinkseries, number=2)
        s = str(corr)
        assert '[15 rows]' in s and len(s.splitlines()) < 20
        assert repr(corr).startswith('pyleoclim.CorrEns')
        assert '<table' in corr._repr_html_()

    def test_psd_repr_t0(self, pinkseries):
        ''' long PSDs are truncated '''
        psd = pinkseries.interp().spectral(method='mtm')
        for out in (str(psd), repr(psd)):
            assert len(out.splitlines()) < 20
        assert repr(psd).startswith('pyleoclim.PSD') and 'mtm' in repr(psd)
        assert '<table' in psd._repr_html_()

    def test_scalogram_repr_t0(self, pinkseries):
        ''' scalograms show a summary, not their values '''
        scal = pinkseries.interp().wavelet()
        for out in (str(scal), repr(scal), scal._repr_html_()):
            assert 'frequency' in out and 'amplitude' in out
        assert len(str(scal).splitlines()) < 10
        assert repr(scal).startswith('pyleoclim.Scalogram')
