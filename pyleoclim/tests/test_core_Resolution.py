''' Tests for pyleoclim.core.ui.Resolution

Naming rules:
1. class: Test{filename}{Class}{method} with appropriate camel case
2. function: test_{method}_t{test_id}

Notes on how to test:
0. Make sure [pytest](https://docs.pytest.org) has been installed: `pip install pytest`
1. execute `pytest {directory_path}` in terminal to perform all tests in all testing files inside the specified directory
2. execute `pytest {file_path}` in terminal to perform all tests in the specified file
3. execute `pytest {file_path}::{TestClass}::{test_method}` in terminal to perform a specific test class/method inside the specified file
4. after `pip install pytest-xdist`, one may execute "pytest -n 4" to test in parallel with number of workers specified by `-n`
5. for more details, see https://docs.pytest.org/en/stable/usage.html
'''

import matplotlib.pyplot as plt

class TestUIResolutionDescribe:
    """Tests for Resolution.describe()"""

    def test_describe_t0(self,unevenly_spaced_series):
        resolution = unevenly_spaced_series.resolution()
        resolution.describe()

class TestUIResolutionPlot:
    """Tests for Resolution.plot()"""

    def test_plot_t0(self,unevenly_spaced_series):
        resolution = unevenly_spaced_series.resolution()
        resolution.plot()
        plt.close()

class TestUIHistPlot:
    """Tests for Resolution.plot()"""

    def test_histplot_t0(self,unevenly_spaced_series):
        resolution = unevenly_spaced_series.resolution()
        resolution.histplot()
        plt.close()

class TestUIDashboard:
    """Tests for Resolution.dashboard()"""

    def test_dashboard_t0(self,unevenly_spaced_series):
        resolution = unevenly_spaced_series.resolution()
        resolution.dashboard()
        plt.close()