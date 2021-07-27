# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from pyfso.core import BaseMesh

plt.style.use("science")


class ThinFilm(BaseMesh):
    def __init__(
        self,
        n_samples: int = 1023,
        grid_length: int = 5.0e-3,
        r: float = 20.0,
        angle: float = np.pi / 2,
        transmittance: float = 0.0,
    ):
        """Thin film

        Definition and creation of a thin film with different transmittance value.

        Attributes
        ----------
        n_samples : int, deafult = 1023
            Financial data series attribute as encapsulated data structure.
            One-dimensional ndarray with axis labels (including time series).

        grid_length : int, deafult = 5.0e-3
            Time interval equivalent to span parameter for the long term EWMA to
            compute MACD line.

        r: float, deafult = 20.0
            Radius of the region where the transmittance is one

        angle: float, deafult = pi/2
            Angle of the thin film.

        transmittance: float, deafult = 0.0
            Transmittance of the thin flim. Range of values [0,1], zero if no light passes and one if all light passes

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
        self.r = r
        self.angle = angle
        self.transmittance = transmittance
        self.grid_spacing = self.grid_length / self.n_samples
        super().__init__()

    def create(self):
        j = 0.0 + 1.0j
        self.c_mask = np.ones((len(self.xx), len(self.yy)))
        for i in range(0, len(self.xx)):
            for j in range(0, len(self.xx)):
                if (
                    self.yy[i, j] >= 0
                    and ((self.xx[i, j] ** 2) + (self.yy[i, j] ** 2)) <= self.r ** 2
                    and self.phi[i, j] <= np.pi / 2
                    and self.angle <= self.phi[i, j]
                ):
                    self.c_mask[i, j] = 1.0
                else:
                    self.c_mask[i, j] = self.transmittance
        return self.c_mask

    def plot_film(self):
        with plt.style.context(["science", "high-vis"]):
            plt.figure(figsize=(12, 12))
            plt.imshow(self.c_mask, extent=[0, 1, 0, 1], cmap="gray")
            plt.title(
                r"Thin film with transmittance $={tr}$".format(tr=self.transmittance),
                fontsize=24,
            )
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.show()
