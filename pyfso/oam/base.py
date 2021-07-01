from abc import ABCMeta


class BaseBeam(ABCMeta):
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
        self.dl = self.grid_length / self.n_samples

    def __create_grid(self):
        self.x = (
            np.arange(-self.n_samples / 2, self.n_samples / 2 + 1, 1)
            * self.grid_spacing
        )
        self.y = (
            np.arange(self.n_samples / 2, -self.n_samples / 2 - 1, -1)
            * self.grid_spacing
        )
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.phi, self.rho = cart2pol(self.X, self.Y)
        self.phi = np.rot90(self.phi)
