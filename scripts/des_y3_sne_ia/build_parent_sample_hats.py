"""Build HATS catalog for DES Y3 SN-Ia lightcurves.

Raw input:
    /mnt/ceph/users/polymathic/external_data/astro/des_y3_sne_ia/des_real_*.dat
Each .dat file is one SN in SNANA ASCII format. v1 prefixes object_id with
``DES_`` so we match that.
"""

from __future__ import annotations

import sys

from mmu.hats_configs import DATASETS
from mmu.sn_ia_snana import build_main


CATALOG_NAME = "des_y3_sne_ia"


def main(argv: list[str] | None = None) -> int:
    return build_main(
        catalog_name=CATALOG_NAME,
        raw_subdir=DATASETS[CATALOG_NAME].raw_path,
        filename_glob="des_real_*.dat",
        object_id_prefix="DES_",
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
