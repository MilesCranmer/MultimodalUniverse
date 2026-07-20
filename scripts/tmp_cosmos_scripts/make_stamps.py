'''
Build from scratch the dataset by cutting out galaxy greyscale stamps. 
'''

from PIL import Image
from astropy.io import fits
from astropy import wcs
from astropy.nddata import Cutout2D
import pandas as pd
import os
import numpy as np
from skimage.transform import resize
from astropy.table import Table
from astropy.wcs import WCS
from astropy.visualization import MinMaxInterval
interval = MinMaxInterval()
from astropy.visualization import AsinhStretch,LogStretch
import pdb

# convert a greyscale numpy array to [0,255] jpg image
def array2img(arr,clipped_percent=0):
    arr = np.arcsinh(arr)
    # max_val = np.percentile(arr,100-clipped_percent/2)
    # min_val = np.percentile(arr,clipped_percent/2)
    # arr = np.clip(arr,min_val,max_val)
    # arr = (arr-min_val)/(max_val-min_val)*255
    max = np.max(arr)
    min = np.min(arr)
    arr = (arr-min)/(max-min)*300.5-0.5
    arr = np.clip(arr,0.,255.)
    return Image.fromarray(arr.astype(np.uint8))

def zero_pix_fraction(img):
    zeros = np.sum(np.max(img,axis=0)==0.)+np.sum(np.max(img,axis=1)==0.)
    print('nzeros:',zeros)
    size = img.shape[0]
    print('size',size)
    return zeros/size


filters = [
    'F115W', 'F150W',  'F277W', 'F444W', 
    'F770W', 
    'HST-F814W', 
    'CFHT-u', 
    'HSC-g', 'HSC-r', 'HSC-i', 'HSC-z', 'HSC-y', 
    'HSC-NB0816', 'HSC-NB0921', 'HSC-NB1010',
    'UVISTA-Y', 'UVISTA-J', 'UVISTA-H', 'UVISTA-Ks', 'UVISTA-NB118', 
    'SC-IA484', 'SC-IA527', 'SC-IA624', 'SC-IA679', 'SC-IA738', 'SC-IA767', 'SC-IB427', 
    ####'SC-IB464', 
    'SC-IB505', 'SC-IB574', 'SC-IB709', 'SC-IB827', 'SC-NB711', 'SC-NB816'
]

filters_FF = [
    'F770W', 
    'CFHT-u', 
    'HSC-g', 'HSC-r', 'HSC-i', 'HSC-z', 'HSC-y', 
    'HSC-NB0816', 'HSC-NB0921', 'HSC-NB1010',
    'UVISTA-Y', 'UVISTA-J', 'UVISTA-H', 'UVISTA-Ks', 'UVISTA-NB118', 
    'SC-IA484', 'SC-IA527', 'SC-IA624', 'SC-IA679', 'SC-IA738', 'SC-IA767', 'SC-IB427', 
    ####'SC-IB464', 
    'SC-IB505', 'SC-IB574', 'SC-IB709', 'SC-IB827', 'SC-NB711', 'SC-NB816'
]

filters_translate = {
    'F115W':       f'F115W',
    'F150W':       f'F150W',
    'F277W':       f'F277W',
    'F444W':       f'F444W',
    'F770W':       f'F770W',
    'HST-F814W':   f'f814w',
    'CFHT-u':      f'COSMOS.U2',
    'HSC-g':       f'HSC-G',
    'HSC-r':       f'HSC-R',
    'HSC-i':       f'HSC-I',
    'HSC-z':       f'HSC-Z',
    'HSC-y':       f'HSC-Y',
    'HSC-NB0816':  f'NB0816',
    'HSC-NB0921':  f'NB0921',
    'HSC-NB1010':  f'NB1010',
    'UVISTA-Y':    f'UVISTA_Y',
    'UVISTA-J':    f'UVISTA_J',
    'UVISTA-H':    f'UVISTA_H',
    'UVISTA-Ks':   f'UVISTA_Ks',
    'UVISTA-NB118':f'UVISTA_NB118',
    'SC-IA484':    f'SPC_L484',
    'SC-IA527':    f'SPC_L527',
    'SC-IA624':    f'SPC_L624',
    'SC-IA679':    f'SPC_L679',
    'SC-IA738':    f'SPC_L738',
    'SC-IA767':    f'SPC_L767',
    'SC-IB427':    f'SPC_L427',
    'SC-IB505':    f'SPC_L505',
    'SC-IB574':    f'SPC_L574',
    'SC-IB709':    f'SPC_L709',
    'SC-IB827':    f'SPC_L827',
    'SC-NB711':    f'SPC_L711',
    'SC-NB816':    f'SPC_L816'
    }

band_lambda_micron = {
                'CFHT-u'        : 0.386,
                'SC-IB427'      : 0.426,
                'HSC-g'         : 0.475,
                'SC-IA484'      : 0.485,
                'SC-IB505'      : 0.506,
                'SC-IA527'      : 0.526,
                'SC-IB574'      : 0.576,
                'HSC-r'         : 0.623,
                'SC-IA624'      : 0.623,
                'SC-IA679'      : 0.678,
                'SC-IB709'      : 0.707,
                'SC-NB711'      : 0.712,
                'SC-IA738'      : 0.736,
                'SC-IA767'      : 0.769,
                'HSC-i'         : 0.770,
                'SC-NB816'      : 0.815,
                'HSC-NB0816'    : 0.816,
                'HST-F814W'     : 0.820,
                'SC-IB827'      : 0.824,
                'HSC-z'         : 0.890,
                'HSC-NB0921'    : 0.921,
                'HSC-y'         : 1.000,
                'HSC-NB1010'    : 1.010,
                'UVISTA-Y'      : 1.102,
                'F115W'         : 1.150,      
                'UVISTA-NB118'  : 1.191,
                'UVISTA-J'      : 1.252,
                'F150W'         : 1.501,
                'UVISTA-H'      : 1.647,
                'UVISTA-Ks'     : 2.156,
                'F277W'         : 2.760, 
                'F444W'         : 4.408, 
                'F770W'         : 7.646
} 



def create_stamps_forzoobot_CEERS_old(img_dir, cat_name, output_dir,filter="f200w"):



    N_POINTINGS = 10
    POINTING1 = [1,2,3,6]
    POINTING2 = [4,5,7,8,9,10]

    # path for catalog
    #cat_dir = "/scratch/ydong/cat"
    #cat_name = "CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v052_bug.csv"

    cat = pd.read_csv(cat_name)

    col_names = ['RA_1','DEC_1']
    coords = cat[col_names].values

    fit_flag = cat['F200W_FLAG'].values
    star_flag = cat['star_flag'].values
    Re_F200W = cat['F200W_RE'].values
    axis_ratio = cat['F200W_Q'].values

    len = len(fit_flag)

    found = np.zeros(len)

    for n in range(1,11):
        if n in POINTING1:
            img_name = "hlsp_ceers_jwst_nircam_nircam%i_"+filter+"f200w_dr0.5_i2d.fits"%n
        if n in POINTING2:
            img_name = "ceers_nircam%i_"+filter+"f200w_v0.51_i2d.fits"%n

        with fits.open(os.path.join(img_dir,img_name)) as hdul:
            # hdul.info()
            hdr = hdul[1].header
            w = wcs.WCS(hdr)
            data = hdul[1].data

            ymax, xmax = data.shape
            pixels = w.wcs_world2pix(coords,0)
            pix_size = 0.031

            for i in range(len):
                if (found[i]==0) & (fit_flag[i]==0) & (star_flag[i]==0):
                    size = 212*np.maximum(0.04*Re_F200W[i]*np.sqrt(axis_ratio[i])/pix_size,0.1)
                    pix = pixels[i]
                    up = int(pix[0]+size)
                    down = up-size*2
                    right = int(pix[1]+size)
                    left = right-size*2
                    if all([up<xmax,down>-1,right<ymax,left>-1]):   
                        # cut = data[left:right,down:up]
                        cut = Cutout2D(data,pix,wcs=w,size=size*2).data

                        if zero_pix_fraction(cut)<0.1:  # exclude images with too many null pixels
                            print(i,n)
                            resized_cut = resize(cut,output_shape=(424,424))

                            image = array2img(resized_cut)

                            # save the images
                            image.save(output_dir+filter+'_%i.jpg'%i)

                            found[i] = 1



def create_stamps_forzoobot_CEERS(img_dir, cat_name, output_dir,filter="f200w"):



   

    # path for catalog
    #cat_dir = "/scratch/ydong/cat"
    #cat_name = "CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v052_bug.csv"

    cat = pd.read_csv(cat_name)

    col_names = ['RA_1','DEC_1']
    coords = cat[col_names].values

    fit_flag = cat['F200W_FLAG'].values
    star_flag = cat['star_flag'].values
    Re_F200W = cat['F200W_RE'].values
    axis_ratio = cat['F200W_Q'].values

    len_cat = len(fit_flag)

    found = np.zeros(len_cat)

    img_name = f"fullceers_ddta_{filter}_v0.51_30mas_sci.fits.gz"

    with fits.open(os.path.join(img_dir,img_name)) as hdul:
        # hdul.info()
        hdr = hdul[0].header
        w = wcs.WCS(hdr)
        data = hdul[0].data

        ymax, xmax = data.shape
        pixels = w.wcs_world2pix(coords,0)
        pix_size = 0.031

        for i in range(len_cat):
            if (found[i]==0) & (fit_flag[i]==0) & (star_flag[i]==0):
                size = 212*np.maximum(0.04*Re_F200W[i]*np.sqrt(axis_ratio[i])/pix_size,0.1)
                pix = pixels[i]
                up = int(pix[0]+size)
                down = up-size*2
                right = int(pix[1]+size)
                left = right-size*2
                if all([up<xmax,down>-1,right<ymax,left>-1]):   
                    # cut = data[left:right,down:up]
                    cut = Cutout2D(data,pix,wcs=w,size=size*2).data

                    if zero_pix_fraction(cut)<0.1:  # exclude images with too many null pixels
                        print(i)
                        resized_cut = resize(cut,output_shape=(424,424))

                        image = array2img(resized_cut)

                            # save the images
                        image.save(os.path.join(output_dir,filter+'_%i.jpg'%i))

                        found[i] = 1





def create_stamps_forzoobot_JADES(img_dir, cat_name, output_dir,filter="f200w"):

    with fits.open(cat_name) as hdul:
        # Assuming the table you're interested in is in the first extension
        # This might need adjustment if your data is in a different HDU
        data = hdul[1].data
    
        # Convert the FITS data to an Astropy Table
        table = Table(data)
    
        # Now convert the Astropy Table into a Pandas DataFrame
        cat = table.to_pandas()
    
    
    cat["F200_AB"] = -2.5*(np.log10(cat.F200W_CIRC0*1e-9))+8.90

    sel = cat.query('F200_AB<27')
    #nir_f200 = fits.open(data_path+"images/hlsp_ceers_jwst_nircam_nircam"+str(c)+"_"+filter+"_dr0.5_i2d.fits.gz")
    
    
    
    

    col_names = ['RA_1','DEC_1']
    coords = sel[col_names].values

    #fit_flag = cat['F200W_FLAG'].values
    #star_flag = cat['star_flag'].values
    Re_F200W = sel['F200W_RHALF'].values
    axis_ratio = sel['Q'].values

    nobj = len(axis_ratio)

    found = np.zeros(nobj)

    

    with fits.open(img_dir+"hlsp_jades_jwst_nircam_goods-s-deep_"+filter+"_v2.0_drz.fits") as img:
        
        w = WCS(img[1].header)
        data = img[1].data
        # Check if the data array is big-endian and convert it if necessary
        if data.dtype.byteorder == '>':
            data = data.byteswap().newbyteorder()
            

        ymax, xmax = data.shape
        pixels = w.wcs_world2pix(coords,0)
        pix_size = 0.031

        for i in range(nobj):
            if (found[i]==0):
                # Skip this object if its pixel values are NaN
                pix = pixels[i]
                if np.isnan(pix).any():
                    continue  # Skip the rest of the loop and proceed with the next iteration

                # Calculate size and check if it is NaN
                size_value = 0.04 * Re_F200W[i] * np.sqrt(axis_ratio[i]) / pix_size
                if np.isnan(size_value):
                    continue  # Skip if size calculation results in NaN

        #        Ensure size is at least a minimum value to avoid too small cuts
                size = 212 * np.maximum(size_value, 0.1)
                
                #size = 212*np.maximum(0.04*Re_F200W[i]*np.sqrt(axis_ratio[i])/pix_size,0.1)
                
                up = int(pix[0]+size)
                down = up-size*2
                right = int(pix[1]+size)
                left = right-size*2
                if all([up<xmax,down>-1,right<ymax,left>-1]):   
                    # cut = data[left:right,down:up]
                    cut = Cutout2D(data,pix,wcs=w,size=size*2).data

                    if zero_pix_fraction(cut)<0.1:  # exclude images with too many null pixels
                        #print(i,n)
                        resized_cut = resize(cut,output_shape=(424,424))

                        image = array2img(resized_cut)

                        # save the images
                        image.save(output_dir+filter+'_%i.jpg'%i)

                        found[i] = 1


def get_filename(path, filt, key):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            if filt in file:
                if key in file:
                    return str(file)


def load_imgs(tile):

    if tile in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10']:
        name_img_det        = f'/n23data2/hakins/COSMOS-Web/detection_images/detection_chi2pos_SWLW_{tile}.fits'
        sci_imas={
            'F115W':       f'/n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8/mosaic_nircam_f115w_COSMOS-Web_30mas_{tile}_v0_8_sci.fits',
            'F150W':       f'/n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8/mosaic_nircam_f150w_COSMOS-Web_30mas_{tile}_v0_8_sci.fits',
            'F277W':       f'/n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8/mosaic_nircam_f277w_COSMOS-Web_30mas_{tile}_v0_8_sci.fits',
            'F444W':       f'/n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8/mosaic_nircam_f444w_COSMOS-Web_30mas_{tile}_v0_8_sci.fits',
            'F770W':       f'/n17data/shuntov/COSMOS-Web/Images_MIRI/Full_v0.7/mosaic_miri_f770w_COSMOS-Web_60mas_{tile}_v0_7_sci.fits',
            'HST-F814W':   f'/n17data/shuntov/COSMOS-Web/Images_HST-ACS/Jan24Tiles/mosaic_cosmos_web_2024jan_30mas_tile_{tile}_hst_acs_wfc_f814w_drz_zp-28.09.fits',
            'CFHT-u':      f'/n17data/shuntov/CWEB-GroundData-Tiles/COSMOS.U2.clipped_zp-28.09_{tile}.fits',
            'HSC-g':       f'/n17data/shuntov/HSC_Tiles-JAN24/{tile}--cutout-HSC-G-9813-pdr3_dud_rev_sci_zp-28.09.fits',
            'HSC-r':       f'/n17data/shuntov/HSC_Tiles-JAN24/{tile}--cutout-HSC-R-9813-pdr3_dud_rev_sci_zp-28.09.fits',
            'HSC-i':       f'/n17data/shuntov/HSC_Tiles-JAN24/{tile}--cutout-HSC-I-9813-pdr3_dud_rev_sci_zp-28.09.fits',
            'HSC-z':       f'/n17data/shuntov/HSC_Tiles-JAN24/{tile}--cutout-HSC-Z-9813-pdr3_dud_rev_sci_zp-28.09.fits',
            'HSC-y':       f'/n17data/shuntov/HSC_Tiles-JAN24/{tile}--cutout-HSC-Y-9813-pdr3_dud_rev_sci_zp-28.09.fits',
            'HSC-NB0816':  f'/n17data/shuntov/HSC_Tiles-JAN24/{tile}--cutout-NB0816-9813-pdr3_dud_rev_sci_zp-28.09.fits',
            'HSC-NB0921':  f'/n17data/shuntov/HSC_Tiles-JAN24/{tile}--cutout-NB0921-9813-pdr3_dud_rev_sci_zp-28.09.fits',
            'HSC-NB1010':  f'/n17data/shuntov/HSC_Tiles-JAN24/{tile}--cutout-NB1010-9813-pdr3_dud_rev_sci_zp-28.09.fits',
            'UVISTA-Y':    f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_Y_12_01_24_allpaw_skysub_015_dr6_rc_v1_zp-28.09_{tile}.fits',
            'UVISTA-J':    f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_J_12_01_24_allpaw_skysub_015_dr6_rc_v1_zp-28.09_{tile}.fits',
            'UVISTA-H':    f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_H_12_01_24_allpaw_skysub_015_dr6_rc_v1_zp-28.09_{tile}.fits',
            'UVISTA-Ks':   f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_Ks_12_01_24_allpaw_skysub_015_dr6_rc_v1_zp-28.09_{tile}.fits',
            'UVISTA-NB118':f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_NB118_12_01_24_allpaw_skysub_015_dr6_rc_v1_zp-28.09_{tile}.fits',
            'SC-IA484':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L484_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IA527':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L527_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IA624':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L624_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IA679':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L679_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IA738':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L738_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IA767':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L767_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IB427':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L427_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IB505':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L505_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IB574':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L574_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IB709':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L709_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IB827':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L827_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-NB711':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L711_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-NB816':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L816_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'ALMA-1mm':    f'/n17data/shuntov/ALMA-CHAMPS/coadd.fits'
            }

    if tile in ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']:
        name_img_det        = f'/n23data2/hakins/COSMOS-Web/detection_images/detection_chi2pos_SWLW_{tile}.fits'
        sci_imas={
            'F115W':       f'/n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8/mosaic_nircam_f115w_COSMOS-Web_30mas_{tile}_v0_8_sci.fits',
            'F150W':       f'/n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8/mosaic_nircam_f150w_COSMOS-Web_30mas_{tile}_v0_8_sci.fits',
            'F277W':       f'/n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8/mosaic_nircam_f277w_COSMOS-Web_30mas_{tile}_v0_8_sci.fits',
            'F444W':       f'/n17data/shuntov/COSMOS-Web/Images_NIRCam/v0.8/mosaic_nircam_f444w_COSMOS-Web_30mas_{tile}_v0_8_sci.fits',
            'F770W':       f'/n17data/shuntov/COSMOS-Web/Images_MIRI/Full_v0.7/mosaic_miri_f770w_COSMOS-Web_60mas_{tile}_v0_7_sci.fits',
            'HST-F814W':   f'/n17data/shuntov/COSMOS-Web/Images_HST-ACS/AprilTiles/mosaic_cosmos_web_2023apr_30mas_tile_{tile}_hst_acs_wfc_f814w_drz_zp-28.09.fits',
            'CFHT-u':      f'/n17data/shuntov/CWEB-GroundData-Tiles/COSMOS.U2.clipped_zp-28.09_{tile}.fits',
            'HSC-g':       f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-G-9813-pdr3_dud_rev-230412-135737_sci_zp-28.09_{tile}.fits',
            'HSC-r':       f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-R-9813-pdr3_dud_rev-230413-121613_sci_zp-28.09_{tile}.fits',
            'HSC-i':       f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-I-9813-pdr3_dud_rev-230413-121625_sci_zp-28.09_{tile}.fits',
            'HSC-z':       f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-Z-9813-pdr3_dud_rev-230413-121629_sci_zp-28.09_{tile}.fits',
            'HSC-y':       f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-Y-9813-pdr3_dud_rev-230413-121631_sci_zp-28.09_{tile}.fits',
            'HSC-NB0816':  f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-NB0816-9813-pdr3_dud_rev-230413-121622_sci_zp-28.09_{tile}.fits',
            'HSC-NB0921':  f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-NB0921-9813-pdr3_dud_rev-230413-121626_sci_zp-28.09_{tile}.fits',
            'HSC-NB1010':  f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-NB1010-9813-pdr3_dud_rev-230413-121845_sci_zp-28.09_{tile}.fits',
            'UVISTA-Y':    f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_Y_12_01_24_allpaw_skysub_015_dr6_rc_v1_zp-28.09_{tile}.fits',
            'UVISTA-J':    f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_J_12_01_24_allpaw_skysub_015_dr6_rc_v1_zp-28.09_{tile}.fits',
            'UVISTA-H':    f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_H_12_01_24_allpaw_skysub_015_dr6_rc_v1_zp-28.09_{tile}.fits',
            'UVISTA-Ks':   f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_Ks_12_01_24_allpaw_skysub_015_dr6_rc_v1_zp-28.09_{tile}.fits',
            'UVISTA-NB118':f'/n17data/shuntov/CWEB-GroundData-Tiles/UVISTA_NB118_12_01_24_allpaw_skysub_015_dr6_rc_v1_zp-28.09_{tile}.fits',
            'SC-IA484':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L484_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IA527':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L527_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IA624':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L624_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IA679':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L679_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IA738':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L738_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IA767':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L767_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IB427':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L427_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IB505':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L505_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IB574':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L574_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IB709':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L709_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-IB827':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L827_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-NB711':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L711_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'SC-NB816':    f'/n17data/shuntov/CWEB-GroundData-Tiles/SPC_L816_20-09-29a_cosmos_zp-28.09_{tile}.fits',
            'ALMA-1mm':    f'/n17data/shuntov/ALMA-CHAMPS/coadd.fits'
            }
        if tile in ['A1', 'A2', 'A3', 'A8', 'A7', 'A6']:
            sci_imas['HSC-g'] =      f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-G-9813-pdr3_dud_rev-230412-135737_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-r'] =      f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-R-9813-pdr3_dud_rev-230413-121613_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-i'] =      f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-I-9813-pdr3_dud_rev-230413-121625_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-z'] =      f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-Z-9813-pdr3_dud_rev-230413-121629_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-y'] =      f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-Y-9813-pdr3_dud_rev-230413-121631_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-NB0816'] = f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-NB0816-9813-pdr3_dud_rev-230413-121622_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-NB0921'] = f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-NB0921-9813-pdr3_dud_rev-230413-121626_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-NB1010'] = f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-NB1010-9813-pdr3_dud_rev-230413-121845_sci_zp-28.09_{tile}.fits'

        elif tile in ['A4', 'A5', 'A9', 'A10']:
            sci_imas['HSC-g'] =      f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-G-9813-pdr3_dud_rev-230413-130357_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-r'] =      f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-R-9813-pdr3_dud_rev-230413-130346_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-i'] =      f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-I-9813-pdr3_dud_rev-230413-130351_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-z'] =      f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-Z-9813-pdr3_dud_rev-230413-130355_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-y'] =      f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-HSC-Y-9813-pdr3_dud_rev-230413-130357_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-NB0816'] = f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-NB0816-9813-pdr3_dud_rev-230413-130357_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-NB0921'] = f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-NB0921-9813-pdr3_dud_rev-230413-130520_sci_zp-28.09_{tile}.fits'
            sci_imas['HSC-NB1010'] = f'/n17data/shuntov/CWEB-GroundData-Tiles/cutout-NB1010-9813-pdr3_dud_rev-230413-130522_sci_zp-28.09_{tile}.fits'

    filters_translate['F115W'] = 'f115w'
    filters_translate['F150W'] = 'f150w'
    filters_translate['F277W'] = 'f277w'
    filters_translate['F444W'] = 'f444w'
    filters_translate['F770W'] = 'f770w'




    imgname_chi2_c20 = '/n08data/COSMOS2020/images/COSMOS2020_izYJHKs_chimean-v3.fits'

    if tile in ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10']:
        ver = 'v3.1.0'
        path_checkimg = f'/n17data/shuntov/COSMOS-Web/CheckImages/JAN24-{tile}_{ver}-ASC/'

    if tile in ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']:
        ver = 'v3.1.0'
        path_checkimg = f'/n17data/shuntov/COSMOS-Web/CheckImages/JAN24-{tile}_{ver}-ASC/'

    # name_img_part  = path_checkimg+get_filename(path_checkimg, '', '_partition.fits')
    name_img_part = f'/n17data/shuntov/COSMOS-Web/ASSOC-files/SE-hotcold_{tile}_grouped_assoc.fits'
    model_imas = {}
    resid_imas = {}

  


    return name_img_det, name_img_part, sci_imas, model_imas, resid_imas, path_checkimg, imgname_chi2_c20, filters_translate


def image_make_cutout(filename, ra1, dec1, arcsec_cut, nameout=None, get_wcs=None):
    import os
    from astropy.coordinates import SkyCoord
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS
    from astropy import units as u
    from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord

    if 'miri' in filename:
        try:
            image_data = fits.getdata(filename, ext=1)
            hdu = fits.open(filename)[1]
            # get image WCS
            w = WCS(hdu.header)
        except:
            image_data = fits.getdata(filename, ext=0)
            hdu = fits.open(filename)[0]
            w = WCS(filename)  
    else:
        image_data = fits.getdata(filename, ext=0)
        hdu = fits.open(filename)[0]
        w = WCS(filename)
    

    # get the pixels from the defined sky coordinates
    sc_1 = SkyCoord(ra1, dec1, unit='deg')
#     sc_2 = SkyCoord(ra2, dec2, unit='deg')
    sc_pix_1 = skycoord_to_pixel(sc_1, w)
#     sc_pix_2 = skycoord_to_pixel(sc_2, w)
#     size_pix_ra = np.int(np.abs(sc_pix_1[0]-sc_pix_2[0]))
#     size_pix_dec = np.int(np.abs(sc_pix_1[1]-sc_pix_2[1]))

    ny, nx = image_data.shape
    image_pixel_scale = wcs.utils.proj_plane_pixel_scales(w)[0]
    image_pixel_scale *= w.wcs.cunit[0].to('arcsec')

    size_pix_ra = arcsec_cut/image_pixel_scale
    size_pix_dec = arcsec_cut/image_pixel_scale

        
#     print(sc_pix_1)
    pos_pix_ra = int((sc_pix_1[0])) #+sc_pix_2[0])/2)
    pos_pix_dec = int((sc_pix_1[1])) #+sc_pix_2[1])/2)

    ## perform the cut
#     try:
    cutout = Cutout2D(image_data, (pos_pix_ra, pos_pix_dec), (size_pix_dec, size_pix_ra), wcs=w)
    
    # Put the cutout image in the FITS HDU
    datacutout = cutout.data
    
    # Update the FITS header with the cutout WCS
#     hdu = fits.PrimaryHDU(data=datacutout, header=cutout.wcs.to_header())
#     hdu.header.update(cutout.wcs.to_header())
    
    # Write the cutout to a new FITS file
#     cutout_filename = '/n08data/shuntov/Images/'+nameout
#     cutout_filename = nameout
#     hdu.writeto(cutout_filename, overwrite=True)
    if get_wcs == True:
        return datacutout, cutout.wcs
    else:
        return datacutout 



def create_stamps_forzoobot_COSMOS(img_dir, cat_name, output_dir,filter="F150W"):
    #name_SEpp_cat = COSMOS_path+"cats/COSMOSWeb_master_v1.6.0-sersic+BD-em_cgs_LePhare_nodupl_nomulti.fits"
    name_SEpp_cat = cat_name
    #COSMOS_path+"cats/COSMOSWeb_master_v2.0.1-sersic-cgs_LePhare-v2_FlaggedM.fits"
    cat_cosmos = Table.read(name_SEpp_cat, format='fits')

    #cat_cosmos = hdu[1].data
    names = [name for name in cat_cosmos.colnames if len(cat_cosmos[name].shape) <= 1]
    
    cat_cosmos_pd=cat_cosmos[names].to_pandas()
    print(cat_cosmos_pd.columns)
    sel = cat_cosmos_pd.query("MAG_MODEL_F150W<27 and MAG_MODEL_F150W>0") #and TILE !='JAN'")
    print(len(sel))
    #pdb.set_trace()
    source_ids = sel['ID_SE++']
    tiles = sel['TILE']
    ra  = sel['RA_MODEL']
    dec = sel['DEC_MODEL']
    Re = sel['RADIUS'].values
    axis_ratio = sel['AXRATIO'].values

    for idn,t,ra_cent,dec_cent,re,q in zip(source_ids,tiles,ra,dec,Re,axis_ratio):
            
            file_path = os.path.join(output_dir, f'{filter}_{idn}.jpg')
            if os.path.exists(file_path):
                continue
            name_img_det, name_img_part, sci_imas, model_imas, resid_imas, path_checkimg, imgname_chi2_c20, filters_translate = load_imgs(t.decode('utf-8'))
            #print(name_img_det)
            print(sci_imas[filter])
            #pdb.set_trace()
            try:
                arcsec_cut =3600*0.04*re * np.sqrt(q)
                size = 212 * np.maximum(arcsec_cut/0.03, 0.1)*0.03
                #size = 212*np.maximum(0.04*Re[i]*np.sqrt(axis_ratio[i])/pix_size,0.1)
                print(size/0.03)
                #print(arcsec_cut)
                if np.isnan(arcsec_cut):
                    print('Size')
                #pdb.set_trace()
                    continue  # Skip if size calculation results in NaN
                stamp, w = image_make_cutout(sci_imas[filter], ra_cent, dec_cent, size*2, nameout=None, get_wcs=True)
                #print(stamp.shape)
            except:
                print('Error creating stamp')
                continue
            full = 'nircam_'+str(t.decode())+'_'+str(idn)
            
            
            if np.isnan(stamp).any():
                print('Nan')
                #pdb.set_trace()
                continue  # Skip the rest of the loop and proceed with the next iteration

            
            print('zero pix',zero_pix_fraction(stamp))
            if zero_pix_fraction(stamp)<0.1:  # exclude images with too many null pixels
                transform = AsinhStretch() + interval
                norm = transform(stamp)  
                resized_cut = resize(stamp,output_shape=(424,424))
                try:
                    array2img(resized_cut).save(file_path)
                except:
                    os.mkdir(output_dir)
                    array2img(resized_cut).save(file_path)


                        
    




#create_stamps_forzoobot_JADES("/n03data/huertas/JADES/images/","/n03data/huertas/JADES/cats/JADES_DR2_PHOT_ZPHOT_PZETA_MASS_Re.fits","/n03data/huertas/JADES/zoobot/")
#create_stamps_forzoobot_COSMOS(None,"/n03data/huertas/COSMOS-Web/cats/COSMOSWeb_master_v3.1.0-sersic-cgs_err-calib_LePhare.fits","/n03data/huertas/COSMOS-Web/zoobot/stamps/f444w",filter='F444W')
#create_stamps_forzoobot_COSMOS(None,"/n03data/huertas/COSMOS-Web/cats/COSMOSWeb_master_v3.1.0-sersic-cgs_err-calib_LePhare.fits","/n03data/huertas/COSMOS-Web/zoobot/stamps/f277w",filter='F277W')

#reate_stamps_forzoobot_CEERS("/n03data/huertas/CEERS/images","/n03data/huertas/CEERS/cats/CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v052_bug.csv","/n03data/huertas/CEERS/zoobot/stamps/f150w",filter='f150w')
create_stamps_forzoobot_COSMOS(None,"/n03data/huertas/COSMOS-Web/cats/COSMOSWeb_master_v3.1.0-sersic-cgs_err-calib_LePhare.fits","/n03data/huertas/COSMOS-Web/zoobot/stamps/f150w",filter='F150W')