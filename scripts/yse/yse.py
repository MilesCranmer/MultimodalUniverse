# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datasets
from datasets import Features, Value, Sequence
from datasets.data_files import DataFilesPatternsDict
import itertools
import h5py
import numpy as np
import os

_CITATION = r"""% CITATION
@dataset{aleo_2022_7317476,
  author       = {Aleo, Patrick D. and
                  Malanchev, Konstantin and
                  Sharief, Sammy N. and
                  Jones, David O. and
                  Narayan, Gautham and
                  Ryan, Foley J. and
                  Villar, V. Ashley and
                  Angus, Charlotte R. and
                  Baldassare, Vivienne F. and
                  Bustamante-Rosell, Maria. J. and
                  Chatterjee, Deep and
                  Cold, Cecilie and
                  Coulter, David A. and
                  Davis, Kyle W. and
                  Dhawan, Suhail and
                  Drout, Maria R. and
                  Engel, Andrew and
                  French, K. Decker and
                  Gagliano, Alexander and
                  Gall, Christa and
                  Hjorth, Jens and
                  Huber, Mark E. and
                  Jacobson-Galan, Wynn V. and
                  Kilpatrick, Charles D. and
                  Langeroodi, Danial and
                  Macias, Phillip and
                  Mandel, Kaisey S. and
                  Margutti, Raffaella and
                  Matasic, Filip and
                  McGill, Peter and
                  Pierel, Justin D. R. and
                  Ransome, Conor L. and
                  Rojas-Bravo, Cesar and
                  Siebert, Matthew R. and
                  Smith, Ken W and
                  de Soto, Kaylee M. and
                  Stroh, Michael C. and
                  Tinyanont, Samaporn and
                  Taggart, Kirsty and
                  Ward, Sam M. and
                  Wojtak, Radosław and
                  Auchettl, Katie and
                  Blanchard, Peter K. and
                  de Boer, Thomas J. L. and
                  Boyd, Benjamin M. and
                  Carroll, Christopher M. and
                  Chambers, Kenneth C. and
                  DeMarchi, Lindsay and
                  Dimitriadis, Georgios and
                  Dodd, Sierra A. and
                  Earl, Nicholas and
                  Farias, Diego and
                  Gao, Hua and
                  Gomez, Sebastian and
                  Grayling, Matthew and
                  Grillo, Claudia and
                  Hayes, Erin E. and
                  Hung, Tiara and
                  Izzo, Luca and
                  Khetan, Nandita and
                  Kolborg, Anne Noer and
                  Law-Smith, Jamie A. P. and
                  LeBaron, Natalie and
                  Lin, Chien C. and
                  Luo, Yufeng and
                  Magnier, Eugene A. and
                  Matthews, David and
                  Mockler, Brenna and
                  O'Grady, Anna J. G. and
                  Pan, Yen-Chen and
                  Politsch, Collin A. and
                  Raimundo, Sandra I. and
                  Rest, Armin and
                  Ridden-Harper, Ryan and
                  Sarangi, Arkaprabha and
                  Schrøder, Sophie L. and
                  Smartt, Stephen J. and
                  Terreran, Giacomo and
                  Thorp, Stephen and
                  Vazquez, Jason and
                  Wainscoat, Richard and
                  Wang, Qinan and
                  Wasserman, Amanda R. and
                  Yadavalli, S. Karthik and
                  Yarza, Ricardo and
                  Zenati, Yossef},
  title        = {{The Young Supernova Experiment Data Release 1 (YSE 
                   DR1) Light Curves}},
  month        = nov,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.7317476},
  url          = {https://doi.org/10.5281/zenodo.7317476}
}
"""

_ACKNOWLEDGEMENTS = r"""% ACKNOWLEDGEMENTS
% From: https://yse.ucsc.edu/acknowledgements/ 

YSE is a collaboration between the DARK Cosmology Centre (University of Copenhagen), UC Santa Cruz, the University of Illinois, and PIs Vivienne Baldassare (Washington State University), Maria Drout (University of Toronto), Kaisey Mandel (Cambridge University), Raffaella Margutti (UC Berkeley) and V. Ashley Villar (Penn State).

The Young Supernova Experiment is supported by the National Science Foundation through grants AST-1518052, AST-1815935, AST-1852393, AST-1911206, AST-1909796, and AST-1944985; the David and Lucile Packard Foundation; the Gordon & Betty Moore Foundation; the Heising-Simons Foundation; NASA through grants NNG17PX03C, 80NSSC19K1386, and 80NSSC20K0953; the Danish National Research Foundation through grant DNRF132; VILLUM FONDEN Investigator grants 16599, 10123 and 25501; the Science and Technology Facilities Council through grants ST/P000312/1, ST/S006109/1 and ST/T000198/1; the Australian Research Council Centre of Excellence for All Sky Astrophysics in 3 Dimensions (ASTRO 3D) through project number CE170100013; the Hong Kong government through GRF grant HKU27305119; the Independent Research Fund Denmark via grant numbers DFF 4002-00275 and 8021-00130, and the European Union’s Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie through grant No. 891744.

The Pan-STARRS1 Surveys (PS1) and the PS1 public science archive have been made possible through contributions by the Institute for Astronomy, the University of Hawaii, the Pan-STARRS Project Office, the Max-Planck Society and its participating institutes, the Max Planck Institute for Astronomy, Heidelberg and the Max Planck Institute for Extraterrestrial Physics, Garching, The Johns Hopkins University, Durham University, the University of Edinburgh, the Queen’s University Belfast, the Harvard-Smithsonian Center for Astrophysics, the Las Cumbres Observatory Global Telescope Network Incorporated, the National Central University of Taiwan, the Space Telescope Science Institute, the National Aeronautics and Space Administration under Grant No. NNX08AR22G issued through the Planetary Science Division of the NASA Science Mission Directorate, the National Science Foundation Grant No. AST-1238877, the University of Maryland, Eotvos Lorand University (ELTE), the Los Alamos National Laboratory, and the Gordon and Betty Moore Foundation.

The YSE team is also thankful for observations obtained with the Samuel Oschin 48-inch Telescope at the Palomar Observatory as part of the Zwicky Transient Facility project. ZTF is supported by the National Science Foundation under Grant No. AST-1440341 and a collaboration including Caltech, IPAC, the Weizmann Institute for Science, the Oskar Klein Center at Stockholm University, the University of Maryland, the University of Washington, Deutsches Elektronen-Synchrotron and Humboldt University, Los Alamos National Laboratories, the TANGO Consortium of Taiwan, the University of Wisconsin at Milwaukee, and Lawrence Berkeley National Laboratories. Operations are conducted by COO, IPAC, and UW.

YSE computations are aided by the University of Chicago Research Computing Center, the Illinois Campus Cluster, and facilities at the National Center for Supercomputing Applications at UIUC.

Time-series dataset from the Young Supernova Experiment Data Release 1 (YSE DR1).
"""

_DESCRIPTION = "Time-series dataset from the Young Supernova Experiment Data Release 1 (YSE DR1)."

_HOMEPAGE = "https://zenodo.org/records/7317476"

_LICENSE = "CC BY 4.0"

_VERSION = "0.0.1"

_STR_FEATURES = [
    "object_id",
    "obj_type",
]

_FLOAT_FEATURES = [
    "ra", 
    "dec", 
    "redshift",
    "host_log_mass"
]


class YSEDR1(datasets.GeneratorBasedBuilder):
    """"""

    VERSION = _VERSION

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="yse_dr1",
            version=VERSION,
            data_files=DataFilesPatternsDict.from_patterns({"train": ["./yse_data/healpix=*/*.hdf5"]}), # This seems fairly inflexible. Probably a massive failure point.
            description="Light curves from YSE DR1",
        ),
    ]

    DEFAULT_CONFIG_NAME = "yse_dr1"

    @classmethod
    def _info(self):
        """Defines the features available in this dataset."""
        # Starting with all features common to light curve datasets
        features = {
            'lightcurve': Sequence(feature={
                'band': Value('string'),
                'flux': Value('float32'),
                'flux_err': Value('float32'),
                'time': Value('float32'),
            }),
        }

        # Adding all values from the catalog
        for f in _FLOAT_FEATURES:
            features[f] = Value("float32")
        for f in _STR_FEATURES:
            features[f] = Value("string")

        ACKNOWLEDGEMENTS = "\n".join([f"% {line}" for line in _ACKNOWLEDGEMENTS.split("\n")])

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=Features(features),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=ACKNOWLEDGEMENTS + "\n" + _CITATION,
        )

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        if not self.config.data_files:
            raise ValueError(
                f"At least one data file must be specified, but got data_files={self.config.data_files}"
            )
        splits = []
        for split_name, files in self.config.data_files.items():
            if isinstance(files, str):
                files = [files]
            splits.append(
                datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files})
            )
        return splits

    def _generate_examples(self, files, object_ids=None):
        """Yields examples as (key, example) tuples."""
        if object_ids is not None:
            files = [f for f in files if os.path.split(f)[-1][:-5] in object_ids]
            # Filter files by object_id
        for file in files:
            with h5py.File(file, "r") as data:
                # Parse data
                idxs = np.arange(0, data["flux"].shape[0])
                band_idxs = idxs.repeat(data["flux"].shape[-1]).reshape(
                    len(data["bands"][()].decode('utf-8').split(",")), -1
                )
                bands = data["bands"][()].decode('utf-8').split(",")
                example = {
                    "lightcurve": {
                        "band": np.asarray([bands[band_number] for band_number in band_idxs.flatten().astype("int32")]).astype("str"),
                        "time": np.asarray(data["time"]).flatten().astype("float32"),
                        "flux": np.asarray(data["flux"]).flatten().astype("float32"),
                        "flux_err": np.asarray(data["flux_err"]).flatten().astype("float32"),
                    }
                }
                    
                # Add remaining features
                for f in _FLOAT_FEATURES:
                    example[f] = np.asarray(data[f]).astype("float32")
                for f in _STR_FEATURES:
                    example[f] = data[f][()].decode('utf-8')

                yield str(data["object_id"][()]), example
