"""Build HATS catalog for Swift UV/optical SN-Ia lightcurves.

Raw input:
    /mnt/ceph/users/polymathic/external_data/astro/swift_sne_ia/*.dat
Each .dat file is one SN in SNANA ASCII format (mix of ASASSN-*, iPTF*,
LSQ*, ... name schemes).
"""

from __future__ import annotations

import sys

from mmu.hats_configs import DATASETS
from mmu.sn_ia_snana import build_main


CATALOG_NAME = "swift_sne_ia"


def main(argv: list[str] | None = None) -> int:
    return build_main(
        catalog_name=CATALOG_NAME,
        raw_subdir=DATASETS[CATALOG_NAME].raw_path,
        filename_glob="*.dat",
        object_id_prefix="",
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
