from abc import ABC, abstractclassmethod

import numpy as np

from pyfso.utils.coords import cart2pol


class AbstractMesh(ABC):
    def __init__(self):
        print(f"Initializing {self.__class__.__name__}")

    @abstractclassmethod
    def create_grid(self):
        raise NotImplementedError("Can't use 'set' on an ADC!")


class BaseMesh(AbstractMesh):
    def __init__(
        self,
        n_samples: int = 1023,
        grid_length: int = 5.0e-3,
    ):
        self.n_samples = n_samples
        self.grid_length = grid_length
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


class AbstractBeam(ABC):
    def __init__(self):
        print(f"Initializing {self.__class__.__name__}")

    @abstractclassmethod
    def compute_optical_params(self):
        raise NotImplementedError("Can't use 'set' on an ADC!")


class BaseBeam(AbstractBeam, BaseMesh):
    def __init__(
        self,
        n_samples: int = 1023,
        grid_length: int = 5.0e-3,
        wave_length: float = 532e-9,
        angular_momentum: float = 5.0,
        beam_waist: float = (0.57e-3) / 2.0,
        z_coor: float = 0.0,
    ):
        self.n_samples = n_samples
        self.grid_length = grid_length
        self.wave_length = wave_length
        self.angular_momentum = angular_momentum
        self.beam_waist = beam_waist
        self.z_coor = z_coor
        self.grid_spacing = self.grid_length / self.n_samples
        BaseMesh.__init__(self)
        self.create_grid()
        self.compute_optical_params()

    def compute_optical_params(self):
        self.wave_number = 2 * np.pi / self.wave_length
        self.rayleigh_range = (np.pi * (self.beam_waist ** 2)) / self.wave_length
        self.beam_width = self.beam_waist * np.sqrt(
            1 + (self.z_coor / self.rayleigh_range) ** 2
        )
        self.kt = self.wave_number / 500
