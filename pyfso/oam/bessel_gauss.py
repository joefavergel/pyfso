# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy import special

from pyfso.oam.base import BaseBeam


class BesselGauss(BaseBeam):
    def __init__(
        self,
        n_samples: int = 1023,
        grid_length: int = 5.0e-3,
        wave_length: float = 532e-9,
        angular_momentum: float = 5.0,
        beam_waist: float = (0.57e-3) / 2.0,
        z_coor: float = 0.0,
        kind: str = "J",
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

        References
        ----------
        .. [1] Gerald Appel, E. D. (2008). Understanding MACD (Moving Average
        Convergence Divergence). 2008.
        .. [2] Achelis, S. B. (n.d.). MACD: Technical Anassslysis from
            A to Z. Retrieved January 29, 2021, from
            ..https://www.metastock.com/customer/resources/taaz/?p=70
        .. versionadded:: 0.1

        """
        self.n_samples = n_samples
        self.grid_length = grid_length
        self.wave_length = wave_length
        self.angular_momentum = angular_momentum
        self.beam_waist = beam_waist
        self.z_coor = z_coor
        self.kind = kind
        super().__init__()

    def create(self):
        j = 0.0 + 1.0j
        self.E = np.multiply(
            special.jv(self.angular_momentum, self.kt * self.rho),
            np.multiply(
                np.exp(j * self.angular_momentum * self.phi),
                np.exp(np.multiply(-self.rho, self.rho) / (self.beam_width ** 2)),
            ),
        )

    def compute_intensity(self):
        intensity = np.multiply(self.E, np.conj(self.E))
        return intensity

    def plot_oam(self):
        plt.figure(figsize=(20, 20))
        intensity = self.compute_intensity()
        plt.imshow(np.real(intensity), extent=[0, 1, 0, 1], cmap="inferno")
        plt.title(
            r"Bessel-Gauss Beam to First Diffraction Order with $m={am}$".format(
                am=self.angular_momentum
            )
        )
        plt.xlim(0.3, 0.7)
        plt.ylim(0.3, 0.7)
