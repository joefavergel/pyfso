# -*- coding: utf-8 -*-

from abc import ABC, abstractclassmethod

import numpy as np

from pyfso.utils.coords import cart2pol


class AbstractBase(ABC):
    def __init__(self):
        print(f"Initializing {self.__class__.__name__}")

    @abstractclassmethod
    def create_grid(self):
        pass


class BaseFilm(AbstractBase):
    def __init__(
        self,
        n_samples: int = 1023,
        grid_length: int = 5.0e-3,
        r: float = 20.0,
        angle: float = -np.pi / 2,
        transmittance: float = 0.0,
    ):
        self.n_samples = n_samples
        self.grid_length = grid_length
        self.r = r
        self.angle = angle
        self.transmittance = transmittance
        self.grid_spacing = self.grid_length / self.n_samples
        self.create_grid()

    def create_grid(self):
        self.x = (
            np.arange(-self.n_samples / 2, self.n_samples / 2 + 1, 1)
            * self.grid_spacing
        )
        self.y = (
            np.arange(self.n_samples / 2, -self.n_samples / 2 - 1, -1)
            * self.grid_spacing
        )
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.phi, self.rho = cart2pol(self.xx, self.yy)
        self.phi = np.rot90(self.phi)
