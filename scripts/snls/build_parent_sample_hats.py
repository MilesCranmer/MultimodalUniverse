"""Build HATS catalog for SNLS (Supernova Legacy Survey) SN-Ia lightcurves.

Raw input:
    /mnt/ceph/users/polymathic/external_data/astro/snls/JLA2014_SNLS_*.dat
Each .dat file is one SN in SNANA ASCII format. The shared SNANA logic lives
in :mod:`mmu.sn_ia_snana` — this script is a thin wrapper that just supplies
the catalog name, raw subdir and filename glob.
"""

from __future__ import annotations

import sys

from mmu.hats_configs import DATASETS
from mmu.sn_ia_snana import build_main


CATALOG_NAME = "snls"


def main(argv: list[str] | None = None) -> int:
    return build_main(
        catalog_name=CATALOG_NAME,
        raw_subdir=DATASETS[CATALOG_NAME].raw_path,
        filename_glob="JLA2014_SNLS_*.dat",
        object_id_prefix="",
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
