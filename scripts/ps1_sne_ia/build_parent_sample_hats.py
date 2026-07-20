"""Build HATS catalog for Pan-STARRS1 SN-Ia lightcurves.

Raw input:
    /mnt/ceph/users/polymathic/external_data/astro/ps1_sne_ia/PS1_PSc*.txt
Each .txt file is one SN in SNANA ASCII format. v1 prefixes object_id with
``PS1_`` so we match that.
"""

from __future__ import annotations

import sys

from mmu.hats_configs import DATASETS
from mmu.sn_ia_snana import build_main


CATALOG_NAME = "ps1_sne_ia"


def main(argv: list[str] | None = None) -> int:
    return build_main(
        catalog_name=CATALOG_NAME,
        raw_subdir=DATASETS[CATALOG_NAME].raw_path,
        filename_glob="PS1_PSc*.txt",
        object_id_prefix="PS1_",
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
