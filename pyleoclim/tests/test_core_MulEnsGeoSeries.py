''' Tests for pyleoclim.core.ui.MulEnsGeoSeries

Naming rules:
1. class: Test{filename}{Class}{method} with appropriate camel case
2. function: test_{method}_t{test_id}

Notes on how to test locally:
0. Make sure [pytest](https://docs.pytest.org) has been installed: `pip install pytest`
1. execute `pytest {directory_path}` in terminal to perform all tests in all testing files inside the specified directory
2. execute `pytest {file_path}` in terminal to perform all tests in the specified file
3. execute `pytest {file_path}::{TestClass}::{test_method}` in terminal to perform a specific test class/method inside the specified file
4. after `pip install pytest-xdist`, one may execute "pytest -n 4" to test in parallel with number of workers specified by `-n`
5. for more details, see https://docs.pytest.org/en/stable/usage.html
'''

import pytest
import pyleoclim as pyleo

class TestUIMulEnsGeoSeriesMCPCA():
    '''Tests for the MulEnsGeoSeries.mcpca function'''
    def test_mcpca_t0(self,ensemblegeoseries_basic):
        ens1 = ensemblegeoseries_basic
        ens2 = ensemblegeoseries_basic
        m_ens = pyleo.MulEnsGeoSeries([ens1,ens2])
        _ = m_ens.mcpca(nsim=10)

    def test_mcpca_t1(self,ensemblegeoseries_nans):
        ens1 = ensemblegeoseries_nans
        ens2 = ensemblegeoseries_nans
        m_ens = pyleo.MulEnsGeoSeries([ens1,ens2])
        _ = m_ens.mcpca(nsim=10)

class TestUIMulEnsGeoSeriesStackplot():
    @pytest.mark.parametrize('labels', [None, 'auto', ['soi','nino']])
    def test_StackPlot_t0(self, ensemblegeoseries_basic, labels):
        ens1 = ensemblegeoseries_basic
        ens2 = ensemblegeoseries_basic
        m_ens = pyleo.MulEnsGeoSeries([ens1,ens2])
        fig, ax = m_ens.stackplot(labels=labels)
        pyleo.closefig(fig)
    
    @pytest.mark.parametrize('plot_kwargs', [{'curve_clr':'red'},[{'qs':[.1,.2,.3,.4,.5]},{'plot_legend':'True'}]])
    def test_StackPlot_t1(self, ensemblegeoseries_basic, plot_kwargs):
        ens1 = ensemblegeoseries_basic
        ens2 = ensemblegeoseries_basic
        m_ens = pyleo.MulEnsGeoSeries([ens1,ens2])
        fig, ax = m_ens.stackplot(plot_kwargs=plot_kwargs)
        pyleo.closefig(fig)
        
    @pytest.mark.parametrize('ylims', ['spacious', 'auto'])
    def test_StackPlot_t2(self, ensemblegeoseries_basic, ylims):
        ens1 = ensemblegeoseries_basic
        ens2 = ensemblegeoseries_basic
        m_ens = pyleo.MulEnsGeoSeries([ens1,ens2])
        fig, ax = m_ens.stackplot(ylims=ylims)
        pyleo.closefig(fig)
        
    @pytest.mark.parametrize('yticks_minor', [True, False])
    def test_StackPlot_t3(self, ensemblegeoseries_basic, yticks_minor):
        ens1 = ensemblegeoseries_basic
        ens2 = ensemblegeoseries_basic
        m_ens = pyleo.MulEnsGeoSeries([ens1,ens2])
        fig, ax = m_ens.stackplot(yticks_minor=yticks_minor)
        pyleo.closefig(fig)
        
    @pytest.mark.parametrize('xticks_minor', [True, False])
    def test_StackPlot_t4(self, ensemblegeoseries_basic, xticks_minor):
        ens1 = ensemblegeoseries_basic
        ens2 = ensemblegeoseries_basic
        m_ens = pyleo.MulEnsGeoSeries([ens1,ens2])
        fig, ax = m_ens.stackplot(xticks_minor=xticks_minor)
        pyleo.closefig(fig)