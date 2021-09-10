# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


"""
This package provides classes for creating/using point spread function is GIANT.

In GIANT a Point Spread Function (PSF) represents the way a camera spreads out light from a point source across multiple
pixels.  This is analogous to the modulation transfer function (MTF) but in the spatial domain instead of the frequency
domain.

PSFs are used extensively throughout GIANT.  They are used to locate the centroid of stars and unresolved objects in
:mod:`.stellar_opnav`, :mod:`.unresolved`, and :mod:`.calibration`.  They are also used in :mod:`.relative_opnav` to
"blur" templates to make them more closely resemble what is actually captured by the camera before attempting
cross-correlation.

There are many different ways to model PSFs for cameras, but one of the most popular and common is to use a
2D Gaussian function.  Giant provides implementations for using Gaussian functions as the PSF in classes
:class:`.Gaussian`, :class:`.GeneralizedGaussian`, :class:`.IterativeGaussian`, :class:`.IterativeGeneralizedGaussian`,
:class:`.IterativeGaussianWBackground` and :class:`IterativeGeneralizedGaussianWBackground`.
These fully implemented classes can generally be used as is throughout GIANT for most cameras.  In some cases however a
camera may have a PSF that is not well modeled by a Gaussian function, in which case, the
:mod:`~giant.point_spread_functions.point_spread_functions` module provides some abstract base classes and common
functionality that can make creating a new PSF easier.

If you are just starting out, we recommend that you begin with one of the provided PSF models as these are generally
sufficient and are easy to use.
"""

from .psf_meta import (PointSpreadFunction, SizedPSF,
                       InitialGuessIterativeNonlinearLSTSQPSF,
                       InitialGuessIterativeNonlinearLSTSQPSFwBackground,
                       IterativeNonlinearLSTSQwBackground,
                       IterativeNonlinearLSTSQPSF, KernelBasedCallPSF, KernelBasedApply1DPSF)

from .gaussians import (Gaussian, GeneralizedGaussian, IterativeGaussian, IterativeGeneralizedGaussian,
                        IterativeGaussianWBackground, IterativeGeneralizedGaussianWBackground)

from .moments import Moment


__all__ = ['PointSpreadFunction', 'SizedPSF', 'InitialGuessIterativeNonlinearLSTSQPSF',
           'InitialGuessIterativeNonlinearLSTSQPSFwBackground', 'IterativeNonlinearLSTSQwBackground',
           'IterativeNonlinearLSTSQPSF', 'KernelBasedCallPSF', 'KernelBasedApply1DPSF', 'Gaussian',
           'GeneralizedGaussian', 'IterativeGaussian', 'IterativeGeneralizedGaussianWBackground',
           'IterativeGeneralizedGaussian', 'IterativeGaussianWBackground', 'Moment']
