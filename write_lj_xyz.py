"""
Write a LAMMPS data file in xyz format
"""

from ase.io.lammpsrun import read_lammps_dump_text
from ase.io import write

def main():
    with open("/Users/robertwexler/Desktop/LJ/dump.lammpstrj", "r") as f:
        atoms = read_lammps_dump_text(f, index=-1)
        write("/Users/robertwexler/Desktop/LJ/atoms.xyz", atoms)

if __name__ == "__main__":
    main()