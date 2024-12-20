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
@ARTICLE{2019ApJ...874..106B,
       author = {{Brout}, D. and {Sako}, M. and {Scolnic}, D. and {Kessler}, R. and {D'Andrea}, C.~B. and {Davis}, T.~M. and {Hinton}, S.~R. and {Kim}, A.~G. and {Lasker}, J. and {Macaulay}, E. and {M{\"o}ller}, A. and {Nichol}, R.~C. and {Smith}, M. and {Sullivan}, M. and {Wolf}, R.~C. and {Allam}, S. and {Bassett}, B.~A. and {Brown}, P. and {Castander}, F.~J. and {Childress}, M. and {Foley}, R.~J. and {Galbany}, L. and {Herner}, K. and {Kasai}, E. and {March}, M. and {Morganson}, E. and {Nugent}, P. and {Pan}, Y. -C. and {Thomas}, R.~C. and {Tucker}, B.~E. and {Wester}, W. and {Abbott}, T.~M.~C. and {Annis}, J. and {Avila}, S. and {Bertin}, E. and {Brooks}, D. and {Burke}, D.~L. and {Carnero Rosell}, A. and {Carrasco Kind}, M. and {Carretero}, J. and {Crocce}, M. and {Cunha}, C.~E. and {da Costa}, L.~N. and {Davis}, C. and {De Vicente}, J. and {Desai}, S. and {Diehl}, H.~T. and {Doel}, P. and {Eifler}, T.~F. and {Flaugher}, B. and {Fosalba}, P. and {Frieman}, J. and {Garc{\'\i}a-Bellido}, J. and {Gaztanaga}, E. and {Gerdes}, D.~W. and {Goldstein}, D.~A. and {Gruen}, D. and {Gruendl}, R.~A. and {Gschwend}, J. and {Gutierrez}, G. and {Hartley}, W.~G. and {Hollowood}, D.~L. and {Honscheid}, K. and {James}, D.~J. and {Kuehn}, K. and {Kuropatkin}, N. and {Lahav}, O. and {Li}, T.~S. and {Lima}, M. and {Marshall}, J.~L. and {Martini}, P. and {Miquel}, R. and {Nord}, B. and {Plazas}, A.~A. and {Roodman}, A. and {Rykoff}, E.~S. and {Sanchez}, E. and {Scarpine}, V. and {Schindler}, R. and {Schubnell}, M. and {Serrano}, S. and {Sevilla-Noarbe}, I. and {Soares-Santos}, M. and {Sobreira}, F. and {Suchyta}, E. and {Swanson}, M.~E.~C. and {Tarle}, G. and {Thomas}, D. and {Tucker}, D.~L. and {Walker}, A.~R. and {Yanny}, B. and {Zhang}, Y. and {DES COLLABORATION}},
        title = "{First Cosmology Results Using Type Ia Supernovae from the Dark Energy Survey: Photometric Pipeline and Light-curve Data Release}",
      journal = {\apj},
     keywords = {cosmology: observations, supernovae: general, techniques: photometric, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2019,
        month = mar,
       volume = {874},
       number = {1},
          eid = {106},
        pages = {106},
          doi = {10.3847/1538-4357/ab06c1},
archivePrefix = {arXiv},
       eprint = {1811.02378},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019ApJ...874..106B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
"""

_ACKNOWLEDGEMENTS = r"""% ACKNOWLEDGEMENTS
From https://www.darkenergysurvey.org/the-des-project/data-access/ :

We request that all papers that use DES public data include the acknowledgement below. In addition, we would appreciate if authors of all such papers would cite the following papers where appropriate:

DR1
The Dark Energy Survey Data Release 1, DES Collaboration (2018)
The Dark Energy Survey Image Processing Pipeline, E. Morganson, et al. (2018)
The Dark Energy Camera, B. Flaugher, et al, AJ, 150, 150 (2015)

This project used public archival data from the Dark Energy Survey (DES). Funding for the DES Projects has been provided by the U.S. Department of Energy, the U.S. National Science Foundation, the Ministry of Science and Education of Spain, the Science and Technology FacilitiesCouncil of the United Kingdom, the Higher Education Funding Council for England, the National Center for Supercomputing Applications at the University of Illinois at Urbana-Champaign, the Kavli Institute of Cosmological Physics at the University of Chicago, the Center for Cosmology and Astro-Particle Physics at the Ohio State University, the Mitchell Institute for Fundamental Physics and Astronomy at Texas A\&M University, Financiadora de Estudos e Projetos, Funda{\c c}{\~a}o Carlos Chagas Filho de Amparo {\`a} Pesquisa do Estado do Rio de Janeiro, Conselho Nacional de Desenvolvimento Cient{\'i}fico e Tecnol{\'o}gico and the Minist{\'e}rio da Ci{\^e}ncia, Tecnologia e Inova{\c c}{\~a}o, the Deutsche Forschungsgemeinschaft, and the Collaborating Institutions in the Dark Energy Survey.

The Collaborating Institutions are Argonne National Laboratory, the University of California at Santa Cruz, the University of Cambridge, Centro de Investigaciones Energ{\'e}ticas, Medioambientales y Tecnol{\'o}gicas-Madrid, the University of Chicago, University College London, the DES-Brazil Consortium, the University of Edinburgh, the Eidgen{\"o}ssische Technische Hochschule (ETH) Z{\"u}rich,  Fermi National Accelerator Laboratory, the University of Illinois at Urbana-Champaign, the Institut de Ci{\`e}ncies de l'Espai (IEEC/CSIC), the Institut de F{\'i}sica d'Altes Energies, Lawrence Berkeley National Laboratory, the Ludwig-Maximilians Universit{\"a}t M{\"u}nchen and the associated Excellence Cluster Universe, the University of Michigan, the National Optical Astronomy Observatory, the University of Nottingham, The Ohio State University, the OzDES Membership Consortium, the University of Pennsylvania, the University of Portsmouth, SLAC National Accelerator Laboratory, Stanford University, the University of Sussex, and Texas A\&M University.
Based in part on observations at Cerro Tololo Inter-American Observatory, National Optical Astronomy Observatory, which is operated by the Association of Universities for Research in Astronomy (AURA) under a cooperative agreement with the National Science Foundation.
"""

# You can copy an official description
_DESCRIPTION = """
Time-series dataset from Dark Energy Survey Year 3 SN Ia (DES Y3 SNe Ia).
"""

_HOMEPAGE = "https://des.ncsa.illinois.edu/releases/sn"

_LICENSE = "CC BY-NC-ND 4.0"

_VERSION = "0.0.1"

_STR_FEATURES = [
    "object_id",
    "obj_type"
]

_FLOAT_FEATURES = [
    "ra", 
    "dec", 
    "redshift",
    "host_log_mass"
]


class DESY3SNIa(datasets.GeneratorBasedBuilder):
    """"""

    VERSION = _VERSION

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="des_y3_sne_ia",
            version=VERSION,
            data_files=DataFilesPatternsDict.from_patterns({"train": ["./healpix=*/*.hdf5"]}), # This seems fairly inflexible. Probably a massive failure point.
            description="Light curves from  DES Y3",
        ),
    ]

    DEFAULT_CONFIG_NAME = "des_y3_sne_ia"

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
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        splits = []
        for split_name, files in self.config.data_files.items():
            if isinstance(files, str):
                files = [files]
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files})) 
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
