''' Tests for pyleoclim.core.ui.EnsMultivarDecomp

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

class TestUIEnsMultivarDecompScreeplot():
    '''Tests for the EnsMultivarDecomp.screeplot function'''
    def test_screeplot_t0(self,ensemblegeoseries_basic):
        ens1 = ensemblegeoseries_basic
        ens2 = ensemblegeoseries_basic
        m_ens = pyleo.MulEnsGeoSeries([ens1,ens2])
        mcpca = m_ens.mcpca(nsim=10)
        fig,_ = mcpca.screeplot()
        pyleo.closefig(fig)

class TestUIEnsMultivarDecompModeplot():
    '''Tests for the EnsMultivarDecomp.modeplot function'''
    def test_modeplot_t0(self,ensemblegeoseries_basic):
        ens1 = ensemblegeoseries_basic
        ens2 = ensemblegeoseries_basic
        m_ens = pyleo.MulEnsGeoSeries([ens1,ens2])
        mcpca = m_ens.mcpca(nsim=10)
        fig,_ = mcpca.modeplot()
        pyleo.closefig(fig)