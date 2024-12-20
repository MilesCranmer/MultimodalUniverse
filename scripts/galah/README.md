# GALAH Dataset Collection

This folder contains the scripts and queries used to build the GALAH spectroscopic parent sample, based on 
all optical spectra at the time of dr3.

## Sample selection

In the current version of the dataset, we retrieve all optical spectra from DR3. Only the following
cuts are applied:
```
    - snr_c3_iraf > 30           # A recommended minimum signal-to-noise ratio
    - flag_sp = 0    
    - flag_fe_h = 0
```

These cuts follow the ![GALAH DR3 best practices](https://www.galah-survey.org/dr3/using_the_data/).


## Data preparation
The data can be loaded for example as follows:
```python
from datasets import load_dataset
import matplotlib.pyplot as plt

dataset = load_dataset('./galah.py', trust_remote_code=True, split='train')
spectrum = dataset['train'][0]['spectrum']
spectrum.keys()

filter_ = 'B'
s_ind, e_ind = dataset['train'][0]['filter_indices'][f'{filter_}_start'], dataset['train'][0]['filter_indices'][f'{filter_}_end']
plt.plot(spectrum['lambda'][s_ind:e_ind], spectrum['flux'][s_ind:e_ind], color='royalblue')
plt.xlabel(r'wavelength ($\AA$)')
plt.ylabel(r'flux (erg/s/cm$^2$/$\AA$)')
plt.title('Spectrum for Selected Star: B Filter')
### Downloading data through Globus

### Spectra extraction

Once the GALAH data has been downloaded, you can create the parent sample by running the following script:
```bash
python build_parent_sample.py [path to GALAH data] [output directory]
```
e.g. `python build_parent_sample.py path_to_galah_data .../AstroPile/galah`

If there is no GALAH data downloaded in the location provided, it will be downloaded by the script - this can, however, take a considerable amount of time and storage.

### Documentation

- GALAH spectra documentation https://www.galah-survey.org/dr3/overview/


