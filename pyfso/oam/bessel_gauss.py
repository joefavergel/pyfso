# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy import special

from pyfso.core.utils import cart2pol
from pyfso.oam.base import BaseBeam


class BesselGauss(BaseBeam):
    def __init__(
        self,
        n_samples: int = 1023,
        grid_length: int = 5.0e-3,
        wave_length: float = 532e-9,
        angular_momentum: float = 5.0,
        beam_weast: float = (0.57e-3)/2.0,
        z_coor: float = 0.0,
        kind: str = 'J'
    ):
    """Bessel-Gauss Beam with Jv(z) Function

    Definition and craation of Bessel-Gauss beam with the first kind of real order and
    complex argument of the Bessel function.

    Attributes
    ----------
    n_samples : int, deafult = 1023
        Financial data series attribute as encapsulated data structure.
        One-dimensional ndarray with axis labels (including time series).

    grid_length : int, deafult = 5.0e-3
        Time interval equivalent to span parameter for the long term EWMA to
        compute MACD line.

    wave_length: float, deafult = 532e-9
        Time interval equivalent to span parameter for the short term EWMA to
        compute MACD line.

    angular_momentum: float, deafult = 5.0
        Time interval equivalent to span parameter for the MACD line EWMA to
        compute signal line.

    beam_weast: float, deafult = (0.57e-3)/2.0
        Divide by decaying adjustment factor in beginning periods to account
        for imbalance in relative weightings (viewing EWMA as a moving
        average).

    z_coor: float, deafult = 0.0
        `True` value defines `min_periods` equal to zero for the moving averages
        (MAs), otherwise `min_periods` will be defined equal to the MA `period`.
        In short, `fillna` as `True` will not leave values in `NaN`, while in
        `False` it will.

    .. note::
        Suggested default values for the MACD indicator are
        ``{'st_period': 12, 'lt_period': 26, 'sl_period': 9}``. Nevertheless,
        for the purpose of replicating the TAS Navigator indicator the default
        values were set as
        ``{'st_period': 13, 'lt_period': 25, 'sl_period': 5}``.
        _More details can be found at: TAS Navigator Official Documentation_
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from techindicators.technical.momentum import MACD
    Now, `MACD` is instantiated from attribute dictionary.
    >>> kwargs = dict(fts=pd.Series(range(0, 30)))
    >>> macd = MACD(**kwargs)
    >>> print(type(macd))
    <class 'techindicators.technical.momentum.MACD'>
    Notice that the `fts` attribute was initialized with data series that
    corresponds to a list of integers from 0 to (`n_rows - 1`) and the rest of
    the attributes are assigned with the default values. Now, the `macd`
    object contains the transformations over `fts` attribute like properties.
    >>> print(macd.macd[0:5])
    0    0.000000
    1    0.018462
    2    0.049057
    3    0.091531
    4    0.145528
    dtype: float64
    >>> print(macd.signal[0:5])
    0    0.000000
    1    0.011077
    2    0.029068
    3    0.055014
    4    0.089761
    dtype: float64
    The `build` method allows to export the `MACD` results as `pandas`
    dataframe at any time.
    >>> print(macd.build().head(5))
           macd  macd_signal  macd_diff
    0  0.000000     0.000000   0.000000
    1 -0.003277    -0.001966  -0.001311
    2 -0.001569    -0.001778   0.000209
    3  0.001592    -0.000378   0.001970
    4  0.006760     0.002362   0.004398
    The `MACD` attributes can be reassigned at any time.
    >>> macd.sl_period = 9
    >>> print(macd.build().head(5))
           macd  macd_signal  macd_diff
    0  0.000000     0.000000   0.000000
    1 -0.003277    -0.001821  -0.001457
    2 -0.001569    -0.001718   0.000148
    3  0.001592    -0.000597   0.002188
    4  0.006760     0.001592   0.005168
    In this last output it can be seen that having reassigned the `sl_period`
    parameter changes the default value from `5` to `9` for the period in the
    EWMA applied on the MACD line, making the signal line and difference series
    changes but the MACD line stays the same.
    References
    ----------
    .. [1] Gerald Appel, E. D. (2008). Understanding MACD (Moving Average
       Convergence Divergence). 2008.
    .. [2] Achelis, S. B. (n.d.). MACD: Technical Anassslysis from
        A to Z. Retrieved January 29, 2021, from
        ..https://www.metastock.com/customer/resources/taaz/?p=70
    .. versionadded:: 0.1
    """

        self.N = N
        self.L = L
        self.dl = L/N
        self.lamb = lamb
        self.m = m
        self.wo = wo
        self.z = z

    def create(self):
        self.x = np.arange(-self.N/2,self.N/2 + 1,1)*self.dl
        self.y = np.arange(self.N/2,-self.N/2 - 1,-1)*self.dl
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.phi, self.rho = cart2pol(self.X, self.Y)
        self.phi = np.rot90(self.phi)

        self.k = 2*np.pi/self.lamb
        j = 0.0 + 1.0j
        self.zr = (np.pi*(self.wo**2))/self.lamb
        self.w = self.wo*np.sqrt(1+(self.z/self.zr)**2)
        self.kt = self.k/500

        self.E = np.multiply(
            special.jv(self.m, self.kt*self.rho),
            np.multiply(np.exp(j*self.m*self.phi),
            np.exp(np.multiply(-self.rho,self.rho)/(self.w**2)))
        )

    def compute_intensity(self):
        intensity = np.multiply(self.E, np.conj(self.E))
        return intensity
    
    def plot_oam(self):
        plt.figure(figsize=(20, 20))
        intensity = self.compute_intensity()
        plt.imshow(np.real(intensity), extent=[0, 1, 0, 1], cmap='inferno')
        plt.title(r'Bessel-Gauss Beam to First Diffraction Order with $m={am}$'.format(
            am=self.m
        ))
        plt.xlim(0.3,0.7)
        plt.ylim(0.3,0.7)
