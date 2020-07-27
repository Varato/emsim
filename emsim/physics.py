import math
import numpy as np


# Constants
water_num_dens = 0.031    # number of molecules per A3
e = 14.39964              # electron charge in Volts * Angstrom
a0 = 0.529177             # Bohr radius in Angstrom
hbar_c = 1975.9086        # hbar*c in eV * Angstrom
hc = 12415                # h*c in eV * Angstrom
m0c2 = 510.9989461        # electron mass in keV


def electron_relativity_gamma(beam_energy_kev: float) -> float:
    gamma = 1 + beam_energy_kev / m0c2
    return gamma


def electron_relativity_mass(beam_energy_kev: float) -> float:
    """
    the relativity electron mass in mc2
    Parameters
    ----------
    beam_energy_kev

    Returns
    -------

    """
    gamma = electron_relativity_gamma(beam_energy_kev)
    return gamma * m0c2


# electron engergy
def electron_wave_length_angstrom(beam_energy_kev: float) -> float:
    """
    Computes the relativity electron wave length according to its energy in keV.

    Parameters
    ----------
    beam_energy_kev: float

    Returns
    -------
    float
        the wave length in Angstrom

    """
    wave_len = hc * 0.001 / math.sqrt(beam_energy_kev**2 + 2*m0c2*beam_energy_kev)
    return wave_len


def interaction_parameter(beam_energy_kev: float) -> float:
    """
    Computes the interaction parameter sigma.
    Here we use the definition fo sigma that has dimension 1/([E] * [L]), which requires the projected potential has
    dimension [E] * [L].

    Parameters
    ----------
    beam_energy_kev: float
        the electron energy in keV

    Returns
    -------
    float
        the interaction paremeter has dimension 1/hbar_c

    """
    dimensionless = (m0c2 + beam_energy_kev) / math.sqrt(beam_energy_kev * (2*m0c2 + beam_energy_kev))
    sigma = (e/hbar_c) * dimensionless
    return sigma


def interaction_parameter2(beam_energy_kev: float) -> float:
    lmd = electron_wave_length_angstrom(beam_energy_kev)
    mc2 = electron_relativity_mass(beam_energy_kev)
    sigma = 2 * math.pi * mc2 * 1000 * e * lmd / hc**2
    return sigma


# optics
def aberration(wave_length_angstrom: float, cs_mm: float, defocus_angstrom: float):
    def aberration_(k):
        chi = 0.5 * math.pi * cs_mm * 1e7 * pow(wave_length_angstrom, 3) * pow(k, 4) \
              - math.pi * defocus_angstrom * wave_length_angstrom * k ** 2
        return chi
    return aberration_


def mtf(wave_length_angstrom: float, cs_mm: float, defocus_angstrom: float):
    """Modulation Transfer Function"""
    def mtf_(k):
        aberr = aberration(wave_length_angstrom, cs_mm, defocus_angstrom)
        return np.exp(-1j * aberr(k))
    return mtf_


def ctf(wave_length_angstrom: float, cs_mm: float, defocus_angstrom: float):
    """
    Contrast Transfer Function: the imaginary part of MTF.
    """
    def ctf_(k):
        aberr = aberr = aberration(wave_length_angstrom, cs_mm, defocus_angstrom)
        return np.sin(aberr(k))
    return ctf_

