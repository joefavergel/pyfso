# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from pyfso.core.utils import cart2pol


class BesselGauss(object):
    def __init__(self, N, L, lamb, m, wo, z):
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
            sp.special.jv(self.m, self.kt*self.rho),
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