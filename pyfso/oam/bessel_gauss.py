# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy import special

from pyfso.oam.base import BaseBeam

plt.style.use("science")


class BesselGauss(BaseBeam):
    def __init__(
        self,
        n_samples: int = 1023,
        grid_length: int = 5.0e-3,
        wave_length: float = 532e-9,
        angular_momentum: float = 5.0,
        beam_waist: float = (0.57e-3) / 2.0,
        z_coor: float = 0.0,
        kind: str = "Jv",
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
        if self.kind == "Jv":
            self.E = np.multiply(
                special.jv(self.angular_momentum, self.kt * self.rho),
                np.multiply(
                    np.exp(j * self.angular_momentum * self.phi),
                    np.exp(np.multiply(-self.rho, self.rho) / (self.beam_width ** 2)),
                ),
            )
        elif self.kind == "Iv":
            n = 1.000293  # refractive index
            d = 1.0
            thetaG = 0.0002  # beam divergence in [rad]
            self.beam_waist / d
            self.wave_number * np.sin(self.wave_length / (d * thetaG * np.pi * n))
            coff = (
                (np.sqrt(np.pi) / self.beam_waist)
                * (np.exp(j * (self.z_coor / self.rayleigh_range)))
                * (np.exp(j * (3 * np.pi / 2) * self.angular_momentum))
                * (np.exp(j * self.wave_number * self.z_coor))
            )
            bessel_iv = special.iv(
                (self.angular_momentum - 1) / 2,
                (self.rho ** 2) / (2 * (self.beam_width ** 2)),
            ) - special.iv(
                (self.angular_momentum + 1) / 2,
                (self.rho ** 2) / (2 * (self.beam_width ** 2)),
            )
            self.E = coff * np.multiply(
                bessel_iv,
                np.multiply(
                    self.rho,
                    np.multiply(
                        np.exp(j * self.angular_momentum * self.phi),
                        np.exp(-(self.rho ** 2) / (self.beam_width ** 2)),
                    ),
                ),
            )
        else:
            raise ValueError(
                "The Bessel-Gauss kind attribute introduced is unkown or unsupported."
            )

    def compute_intensity(self):
        intensity = np.multiply(self.E, np.conj(self.E))
        return intensity

    def plot_oam(self):
        with plt.style.context(["dark_background", "science", "high-vis"]):
            plt.figure(figsize=(12, 12))
            intensity = self.compute_intensity()
            plt.imshow(np.real(intensity), extent=[0, 1, 0, 1], cmap="inferno")
            plt.title(
                r"Bessel-Gauss ${kind_}(z)$ Beam to First Diffraction Order with $m={am}$".format(
                    am=self.angular_momentum, kind_=self.kind
                ),
                fontsize=24,
            )
            plt.xlim(0.3, 0.7)
            plt.ylim(0.3, 0.7)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.show()
