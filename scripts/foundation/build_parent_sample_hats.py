"""Build HATS catalog for Foundation DR1 SN-Ia lightcurves.

Raw input:
    /mnt/ceph/users/polymathic/external_data/astro/foundation/Foundation_DR1_*.txt
Each .txt file is one SN in SNANA ASCII format. Shared SNANA logic lives in
:mod:`mmu.sn_ia_snana` — this script is a thin wrapper.
"""

from __future__ import annotations

import sys

from mmu.hats_configs import DATASETS
from mmu.sn_ia_snana import build_main


CATALOG_NAME = "foundation"


def main(argv: list[str] | None = None) -> int:
    return build_main(
        catalog_name=CATALOG_NAME,
        raw_subdir=DATASETS[CATALOG_NAME].raw_path,
        filename_glob="Foundation_DR1_*.txt",
        object_id_prefix="",
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
