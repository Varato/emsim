import math
import numpy as np


# Constants
water_num_dens = 0.031    # number of molecules per A3
e = 4.8032042510e-10      # statC = 1 esu = 1 g^(1/2) cm^(3/2) s^(−1)
h = 6.62607015e-27        # cm2 g / s = erg s
hbar = h / 2.0 / math.pi
m0 = 9.1093835611e-28     # g
a0 = 5.29177210903e-9  # cm
c = 29979245800.0         # cm / s
m0c2 = m0 * c**2          # erg: 1erg = 1e-7 J = 1e-10 / e_coulumb keV
hc = h * c                # erg cm
hbar_c = hbar * c         # erg cm

e_coulumb = 1.602176634e-19
m0c2_keV = m0c2 * 1e-7 / e_coulumb / 1000.
hbar_c_keV_Ang = hbar_c * 0.01 / e_coulumb
hc_keV_Ang = hc * 0.01 / e_coulumb


def electron_relativity_gamma(beam_energy_kev: float) -> float:
    gamma = 1 + beam_energy_kev / m0c2_keV
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
    wave_len = hc_keV_Ang / math.sqrt(beam_energy_kev**2 + 2*m0c2_keV*beam_energy_kev)
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
    dimensionless = (m0c2_keV + beam_energy_kev) / math.sqrt(beam_energy_kev * (2*m0c2_keV + beam_energy_kev))
    sigma = (e/hbar_c) * dimensionless
    return sigma


# optics
def aberration(wave_length_angstrom: float, cs_mm: float, defocus_angstrom: float):
    cslambda = cs_mm * 1e7 * wave_length_angstrom
    D = defocus_angstrom / math.sqrt(cslambda)
    def aberration_(k):
        K = k * pow(cslambda * wave_length_angstrom**2, 0.25)
        chi = math.pi * (0.5*K**4 - D*K**2)
        return chi
    return aberration_


def mtf(wave_length_angstrom: float, cs_mm: float, defocus_angstrom: float):
    """
    Modulation Transfer Function
    """
    def mtf_(k):
        aberr = aberration(wave_length_angstrom, cs_mm, defocus_angstrom)
        return np.exp(-1j * aberr(k))
    return mtf_


def ctf(wave_length_angstrom: float, cs_mm: float, defocus_angstrom: float):
    """
    Contrast Transfer Function: the imaginary part of MTF.
    """
    def ctf_(k):
        aberr = aberration(wave_length_angstrom, cs_mm, defocus_angstrom)
        return np.sin(aberr(k))
    return ctf_

def optimal_focus(wave_length_angstrom: float, cs_mm: float, nd: int):
    """
    The optimal defocuses
    df = sqrt( (2nd - 0.5) * cs * lambda ), nd = 1, 2, 3...

    Parameters
    ----------
    wave_length_angstrom: float
        the electron's wave length in Angstroms
    cs_mm: float
        the spherical aberration in mm
    nd: int
        the index of optimal focuses.

    Returns
    -------
    float
        the optimal defocus value
    """
    cslambda = cs_mm * 1e7 * wave_length_angstrom
    return math.sqrt((2*nd - 0.5)* cslambda)

def scherzer_condition(wave_length_angstrom: float, cs_mm: float):
    """
    Compute Scherzer focus and aperture.

    Parameters
    ----------
    wave_length_angstrom: float
        the electron's wave length in Angstroms
    cs_mm: float
        the spherical aberration in mm

    Returns
    -------
    df: float
        the Scherzer focus value in Angstroms
    aper: float
        the Scherzer aperture value in rad
    """
    df = optimal_focus(wave_length_angstrom, cs_mm, 1)
    aper = pow(6*wave_length_angstrom/(cs_mm * 1e7), 0.25)
    return df, aper