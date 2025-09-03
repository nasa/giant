


r"""
This package provides utilities for visually inspecting star identification and attitude estimation results.

In general, the only functions a user will directly interface with from this module are the :func:`show_id_results`
which shows the results of performing star identification and attitude estimation, :func:`residual_histograms` which
shows histograms of the residuals, :func:`plot_residuals_vs_magnitude` which generates a scatter plot of residuals
as a function of star magnitude, and :func:`plot_residuals_vs_temperature` which generates a scatter plot of residuals
as a function of camera temperature.  The other contents of this model are used for manual outlier inspection, which is
typically done by using the :meth:`~.StellarOpNav.review_outliers` method.
"""

from giant.stellar_opnav.visualizers.residual_histograms import residual_histograms
from giant.stellar_opnav.visualizers.residuals_vs_magnitude import residuals_vs_magnitude
from giant.stellar_opnav.visualizers.residuals_vs_temperature import residuals_vs_temperature
from giant.stellar_opnav.visualizers.show_id_results import show_id_results
from giant.stellar_opnav.visualizers.show_outliers import show_outlier, OutlierCallback
