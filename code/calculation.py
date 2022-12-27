import numpy as np


def get_atom_position_and_velocity(file_name, skip_lines):
    """
    This function reads the position and velocity of each atom in the lammps output file
    and return it as a numpy array
    :param file_name:
    :param skip_lines:
    :return: atom info
    """
    atom_info = np.loadtxt(fname=file_name,
                           dtype=np.float64,
                           comments='#',
                           delimiter=None,
                           converters=None,
                           skiprows=skip_lines,
                           usecols=None,
                           unpack=False,
                           ndmin=0,
                           encoding='bytes',
                           max_rows=None,
                           quotechar=None,
                           like=None)
    return atom_info


def get_electron_ionization_crosssection(energy_eV):
    """
    This function get the electron diffraction crosssection for simulation
    :param energy_eV:
    :return:
    """
    pass


def get_photon_ionization_crosssection(energy_eV):
    """
    This function return the photon ionization cross-section for photon with water
    :param energy_eV:
    :return:
    """
    pass


def update_particle_status(particles, photons, ions):
    """

    :param particles:
    :param photons:
    :param ions:
    :return:
    """
    # Step 1: calculate the potential