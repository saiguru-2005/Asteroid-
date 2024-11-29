import numpy as np


import cv2

from random import randint
from PIL import Image
import glob
import random
from skimage import exposure
import copy

from astropy.time import Time
from astropy.wcs import NoConvergence
from astropy.utils.data import clear_download_cache
from astropy.wcs import WCS

from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import sep
import os
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import json
import torch
import re

from numpy.linalg import norm
import sep

#device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(f'CHECK devices {device}')

from tqdm import tqdm

def boxes_to_full_img(box_pos, ref_pos):
    
    # parameters : box_pos (x_min, y_min, delta_x, delta_y) --> img unit
    #              ref_pos (x_min, y_min) ---> array unit
    
    # Output : center_x, center_y --->  output as image unit
    
    x_b_min, y_b_min, delta_x, delta_y = box_pos
    
    center_x_img = x_b_min + delta_x/2
    center_y_img = y_b_min + delta_y/2
    
    # Get center in full image
    center_x_img += ref_pos[1]
    center_y_img += ref_pos[0]
    
    return int(center_x_img), int(center_y_img)


#####################################################


def normalized_img(fit_img, threshold = 40000):
    fit_img = np.where(fit_img<0,0,fit_img)
    fit_img = np.where(fit_img>threshold,threshold,fit_img)
    fit_img /= threshold
    
    return fit_img
def get_pos_samples(template_img, labels, drop_margin = 0.05, ast_mapping_range = 0.3, num_rand = 10, crop_size=20):

    # images : list of one or more image path(s)
    # labels : list of [ra,dec] of the centers of asteroids

    # == Initialize lists == #
    pos_samples = []
    #break_threshold_count = 10
    
    #neg_samples = []

    template_img, pos_mapper = template_img # first image in an input array
    
    # == preprocessing template to gray scale == #
    #template_img = normalized_img(template_img, threshold = 40000)
    #print(np.min(template_img),np.max(template_img),len(labels))
    
    """# == Random range boundary == #
    max_x, max_y = np.shape(template_img) # size in array position unit

    # == Random asteroid position in an output image == #
    rand_x_min = int(np.round(max_x*drop_margin))
    rand_x_max = int(np.round(max_x*(1-drop_margin)))-crop_size



    rand_y_min = int(np.round(max_y*drop_margin))
    rand_y_max = int(np.round(max_y*(1-drop_margin)))-crop_size"""


    # == Convert radec to pixel == #
    radec = SkyCoord(labels, unit=(u.deg, u.deg), frame='fk5')
    y_list, x_list  = pos_mapper.world_to_pixel(radec) # Array unit
    
    
    ## ===========> Positive samples <=========== ##
    #print('SHOW POSITIVE TARGET')
    for sel_ast in range(len(x_list)) :
        for i in range(num_rand):
            ast_y, ast_x = y_list[sel_ast], x_list[sel_ast]


            # == Random asteroid position in an image == #
            # mapping_range = [int(crop_size*ast_mapping_range),int(crop_size*(1-ast_mapping_range))]
            # #print(f'mapping range : {mapping_range}')

            # map_x = randint(mapping_range[0],mapping_range[1])
            # map_y = randint(mapping_range[0],mapping_range[1])

            #print(ast_x, map_x)
            x_c = int(ast_x) #+ map_x - crop_size/2
            y_c = int(ast_y) # + map_y - crop_size/2

            pos_box = [x_c, y_c]

            # Test only : whether the position is in the pixel coordinates
            """fig,ax=plt.subplots()
            #ax.set_title(f'Masked position at {int(ast_y-y_c+crop_size/2),int(ast_x-x_c+crop_size/2)}  : {template_img[int(ast_y-y_c+crop_size/2),int(ast_x-x_c+crop_size/2)]}')
            ax.set_title(f'{pos_box} ') #with value : {template_img[int(ast_y-y_c+crop_size/2),int(ast_x-x_c+crop_size/2)]}
            plot_img = template_img[int(x_c-crop_size/2):int(x_c+crop_size/2),int(y_c-crop_size/2):int(y_c+crop_size/2)]
            plt.plot(int(ast_y-y_c+crop_size/2),int(ast_x-x_c+crop_size/2),'r*')

            ax.imshow(plot_img)"""
            #print(np.min(plot_img),np.max(plot_img))
            #plt.show()

            pos_samples.append(pos_box)


    # return as radec format
    ra,dec = split_ra_dec(flip_xy(pos_samples))
    pos_samples_world = pos_mapper.pixel_to_world(ra,dec)
    
    
    # output as radec of center
    return pos_samples_world

def set_pathname(input_path,day_key,dataset_root = '../../../../fits_images/'):
    month = '_'.join(day_key.split('_')[:-1])
    day = day_key.split('_')[-1]
    out_path = os.path.join(dataset_root,month,'img',day, os.path.basename(input_path))
    return out_path
    
def objects_to_boxes(objects):
    map_id = dict()
    
    #print(f'CHECK AST X AND Y : {x_t, y_t}')


    for i in range(len(objects['x'])):  
        
        x_min,y_min,delta_x,delta_y = BBox_fitElipse(x=objects['x'][i], y=objects['y'][i] ,a=4*objects['a'][i] ,b=4*objects['b'][i] 
                                                    ,theta=objects['theta'][i])
        

        #count_idx += 1
        #x_c = x_min + delta_x
        #y_c = y_min + delta_y
        map_id[i] = [int(x_min), int(y_min), int(delta_x), int(delta_y)] # x_center, y_center, delta_x(width), delta_y(height)

    return map_id

def BBox_fitElipse(x,y,a,b,theta):
    
    
    param_x = -b*np.tan(theta)/a
    param_x = np.arctan(param_x) #x_min
    param_y = b/(a*np.tan(theta))
    param_y = np.arctan(param_y) #y_min 
    

    x_1 =  x + a*np.cos(param_x-np.pi)*np.cos(theta) - b*np.sin(param_x-np.pi)*np.sin(theta)
    y_1 = y +  b*np.sin(param_y-np.pi)*np.cos(theta) + a*np.cos(param_y-np.pi)*np.sin(theta)

    
    x_2 = x + a*np.cos(param_x)*np.cos(theta) - b*np.sin(param_x)*np.sin(theta) 
    y_2 = y + b*np.sin(param_y)*np.cos(theta) + a*np.cos(param_y)*np.sin(theta) 

    return min(x_1,x_2),min(y_1,y_2),abs(x_2-x_1),abs(y_2-y_1)

def adjust_bbox(xmin,ymin,w,h,img_size=256):
    
    xmax = xmin + w
    ymax = ymin + h
    
    if xmin < 0:
        w = xmax
        xmin = 0
        
    if ymin < 0:
        h = ymax
        ymin = 0
    
    if xmax > img_size:
        w = xmax - img_size
        
    
    if ymax > img_size:
        h = ymax - img_size
        
            
    return xmin,ymin,w,h
            
            
def flip_xy(samples : list):

    # This elements xy in samples must be flipped to be transformed by astropy pixel to world
    flipped_samples = [ [y,x] for x,y in samples ]

    return flipped_samples

def split_ra_dec(samples : list):
    
    ra_list = []
    dec_list = []
    for element in samples:
        ra, dec = element
        ra_list.append(ra)
        dec_list.append(dec)
    
    return ra_list, dec_list

def get_neg_samples(template_img, labels, drop_margin = 0.05, crop_size = 30, search_multiplier=2, ast_mapping_range = 0.2, num_rand = 10,det_threshold=2, break_threshold_count=10):

    # images : list of one or more image path(s)
    # labels : list of [ra,dec]

    # == Initialize lists == #
    neg_samples = []

    template_img, pos_mapper = template_img # first image in an input array
    
    # == Random range boundary == #
    max_x, max_y = np.shape(template_img) # size in array position unit

    # == Random asteroid position in an output image == #
    rand_x_min = int(np.round(max_x*drop_margin))
    rand_x_max = int(np.round(max_x*(1-drop_margin)))-crop_size



    rand_y_min = int(np.round(max_y*drop_margin))
    rand_y_max = int(np.round(max_y*(1-drop_margin)))-crop_size


    # == Convert radec to pixel == #
    radec = SkyCoord(labels, unit=(u.deg, u.deg), frame='fk5')
    y_list, x_list  = pos_mapper.world_to_pixel(radec) # Array unit

    for i in range(num_rand):
        
        
        ## ===========> Negative samples <=========== ##
        while True:
            
            bbox = [randint(rand_x_min, rand_x_max),randint(rand_y_min, rand_y_max) ,crop_size ,crop_size] # random point
            
            # check whether there are asteroid in this box
            for point in list(zip(x_list, y_list)):
                
                #print(f'Check bbox : {bbox}')
                if check_ast_in_box(bbox,point):
                    found_ast = True
                    break

                found_ast = False
                    
            if not found_ast:
                # perform source extraction in a small patch
            
                #print(bbox)
                neg_img = template_img[int(bbox[0]):int(bbox[0]+crop_size*search_multiplier), int(bbox[1]):int(bbox[1]+crop_size*search_multiplier)]

                #print(np.shape(neg_img))
                rp2, rp98 = np.percentile(neg_img, (2, 98))
                scale_img = exposure.rescale_intensity(neg_img, in_range=(rp2, rp98))
                bkg_noise = sep.Background(scale_img)

                data_sub = scale_img-bkg_noise
                c_data = data_sub.copy(order='C')
                objects = sep.extract(c_data,det_threshold,bkg_noise.globalrms)
                #print(f'objects in a patch {len(objects)}')
                obj_boxes = objects_to_boxes(objects,size = crop_size)

                #return obj_boxes
                
                # if no box found
                if len(obj_boxes) <1 :
                    continue



                repeat = 0
                skip_append =  False
                while True:
                    rand_box = random.choice(obj_boxes) # image unit format : (x_min, y_min, width, height)
                    #print(rand_box)
                    #obj_boxes.remove(rand_box)
                    
                    
                    # Conditional selection of random process
                    x_center, y_center = rand_box[0]+rand_box[2]/2, rand_box[1]+rand_box[3]/2
                    allow_range = [crop_size, crop_size*(search_multiplier-1)]
                    
                    
                    
                    
                    
                    # break if the center is in the range
                    # min range
                    if x_center > allow_range[0] and y_center > allow_range[0]:
                        
                        # max range
                        if x_center < allow_range[1] and y_center < allow_range[1]:
                            break
                    
                    # if no src that pass the condition
                    if repeat > break_threshold_count :
                        skip_append = True
                        break
                    repeat += 1

                if skip_append :
                    continue
                #print(f'Random box : {rand_box}')





                #Test only : whether the position is in the pixel coordinates
                """fig,ax=plt.subplots()

                ax.imshow(neg_img[int(rand_box[0]):int(rand_box[0]+crop_size),int(rand_box[1]):int(rand_box[1]+crop_size)])
                plt.title(f'{rand_box[0]+bbox[1]-bbox[3]/2},{rand_box[1]+bbox[0]-bbox[2]}')
                plt.show()"""



                neg_box = [y_center+bbox[0], x_center+bbox[1]] # center position in array-based unit 


                neg_samples.append(neg_box)
            #print(neg_box)
            break        
    
    
    ra,dec = split_ra_dec(flip_xy(neg_samples))
    neg_samples_world = pos_mapper.pixel_to_world(ra,dec)

            


    return neg_samples_world


def get_neg_samples_norand(template_img, labels, drop_margin = 0.05, crop_size = 30, search_multiplier=2, ast_mapping_range = 0.2,det_threshold=2, break_threshold_count=10):

    # images : list of one or more image path(s)
    # labels : list of [ra,dec]

    # == Initialize lists == #
    neg_samples = []

    template_img, pos_mapper = template_img # first image in an input array
    
    # == Random range boundary == #
    max_x, max_y = np.shape(template_img) # size in array position unit

    # == Random asteroid position in an output image == #
    rand_x_min = int(np.round(max_x*drop_margin))
    rand_x_max = int(np.round(max_x*(1-drop_margin)))-crop_size



    rand_y_min = int(np.round(max_y*drop_margin))
    rand_y_max = int(np.round(max_y*(1-drop_margin)))-crop_size


    # == Convert radec to pixel == #
    radec = SkyCoord(labels, unit=(u.deg, u.deg), frame='fk5')
    y_list, x_list  = pos_mapper.world_to_pixel(radec) # Array unit

    for i in range(num_rand): 
        
        
        ## ===========> Negative samples <=========== ##
        while True:
            
            bbox = [randint(rand_x_min, rand_x_max),randint(rand_y_min, rand_y_max) ,crop_size ,crop_size] # random point
            
            # check whether there are asteroid in this box
            for point in list(zip(x_list, y_list)):
                
                #print(f'Check bbox : {bbox}')
                if check_ast_in_box(bbox,point):
                    found_ast = True
                    break

                found_ast = False
                    
            if not found_ast:
                # perform source extraction in a small patch
            
                #print(bbox)
                neg_img = template_img[int(bbox[0]):int(bbox[0]+crop_size*search_multiplier), int(bbox[1]):int(bbox[1]+crop_size*search_multiplier)]

                #print(np.shape(neg_img))
                rp2, rp98 = np.percentile(neg_img, (2, 98))
                scale_img = exposure.rescale_intensity(neg_img, in_range=(rp2, rp98))
                bkg_noise = sep.Background(scale_img)

                data_sub = scale_img-bkg_noise
                c_data = data_sub.copy(order='C')
                objects = sep.extract(c_data,det_threshold,bkg_noise.globalrms)
                #print(f'objects in a patch {len(objects)}')
                obj_boxes = objects_to_boxes(objects,size = crop_size)

                #return obj_boxes
                
                # if no box found
                if len(obj_boxes) <1 :
                    continue



                repeat = 0
                skip_append =  False
                while True:
                    rand_box = random.choice(obj_boxes) # image unit format : (x_min, y_min, width, height)
                    #print(rand_box)
                    #obj_boxes.remove(rand_box)
                    
                    
                    # Conditional selection of random process
                    x_center, y_center = rand_box[0]+rand_box[2]/2, rand_box[1]+rand_box[3]/2
                    allow_range = [crop_size, crop_size*(search_multiplier-1)]
                    
                    
                    
                    
                    
                    # break if the center is in the range
                    # min range
                    if x_center > allow_range[0] and y_center > allow_range[0]:
                        
                        # max range
                        if x_center < allow_range[1] and y_center < allow_range[1]:
                            break
                    
                    # if no src that pass the condition
                    if repeat > break_threshold_count :
                        skip_append = True
                        break
                    repeat += 1

                if skip_append :
                    continue
                #print(f'Random box : {rand_box}')





                #Test only : whether the position is in the pixel coordinates
                """fig,ax=plt.subplots()

                ax.imshow(neg_img[int(rand_box[0]):int(rand_box[0]+crop_size),int(rand_box[1]):int(rand_box[1]+crop_size)])
                plt.title(f'{rand_box[0]+bbox[1]-bbox[3]/2},{rand_box[1]+bbox[0]-bbox[2]}')
                plt.show()"""



                neg_box = [y_center+bbox[0], x_center+bbox[1]] # center position in array-based unit 


                neg_samples.append(neg_box)
            #print(neg_box)
            break        
    
    
    ra,dec = split_ra_dec(flip_xy(neg_samples))
    neg_samples_world = pos_mapper.pixel_to_world(ra,dec)

            


    return neg_samples_world

def load_raw_fits_img(img_path):
    data = fits.getdata(img_path)
    header = fits.getheader(img_path, ext=1)
    return data, WCS(header)

def load_norm_fits_img(img_path):
    data = fits.getdata(img_path)
    header = fits.getheader(img_path, ext=1)
    
    data = normalized_img(data)
    return data, WCS(header)

# ================ Scaler ======================== #
def load_stdscaler_fits_img(img_path):
    data = fits.getdata(img_path)
    header = fits.getheader(img_path, ext=1)
    
    data = StandardScaler_img(data)
    return data, WCS(header)

def StandardScaler_img(fit_img):
    scaler = StandardScaler()
    scaler.fit(fit_img)
    fit_img = scaler.transform(fit_img)

    return fit_img

def load_minmaxscaler_fits_img(img_path):
    data = fits.getdata(img_path)
    header = fits.getheader(img_path, ext=1)
    
    data = MinMaxscaler_img(data)
    return data, WCS(header)

def MinMaxscaler_img(fit_img):
    scaler = MinMaxScaler()
    scaler.fit(fit_img)
    fit_img = scaler.transform(fit_img)

    return fit_img



# axes plots function
def check_ast_in_box(bbox,point, padding=0):
    x_min, y_min, width, height = bbox
    ast_in_x, ast_in_y = False, False
    if x_min-padding <= point[0] <= x_min+width+padding:
        ast_in_x = True
    if y_min-padding <= point[1] <= y_min+height+padding:
        ast_in_y = True
    
    if ast_in_x and ast_in_y:
        return True

def plot_raw_image(data, ax=None, **kwargs):
    ax = ax or plt.gca()
    ax.axis('off')
    # Do some cool data transformations...
    return ax.imshow(data, **kwargs)

def plot_noise_sub_image(data, ax=None, **kwargs):
    ax = ax or plt.gca()
    ax.axis('off')
    
    rp2, rp98 = np.percentile(data, (2, 98))
    scale_img = exposure.rescale_intensity(data, in_range=(rp2, rp98))
    bkg_noise = sep.Background(scale_img)

    data_sub = scale_img-bkg_noise
    # Do some cool data transformations...
    return ax.imshow(data_sub, **kwargs)


    
def plot_diff_image(template,target, ax=None, scaling= None, **kwargs):
    ax = ax or plt.gca()
    data = target - template
    if scaling=='norm_near_zero' :
        # norm into [-1,1]
        data /= np.max(data)

    ax.axis('off')    
    # Do some cool data transformations...
    return ax.imshow(data, **kwargs)
    
    
#convolve(src_img,kernel)
def plot_convolved_image(template,target, kernel, ax=None, scaling= None, **kwargs):
    ax = ax or plt.gca()
    data = target - template
    data = convolve(data, kernel)

    ax.axis('off')    
    # Do some cool data transformations...
    return ax.imshow(data, **kwargs)

def load_pos_samples(input_dict, pos_images, box_size):
    pos_src_box = []


    for day_key, day_data in input_dict.items():

        print(f'{day_key} contains {len(day_data)} trks')
        #continue
        for trk_data in tqdm(day_data):

            imgs_in_a_trk, labels = trk_data
            #first_img = load_raw_fits_img(imgs_in_a_trk[0])

            #print(np.shape(first_img))

            pos_sample = pos_images[imgs_in_a_trk[0]]
            #print(pos_sample)
            # Pre loading images
            template, wcs1 = load_norm_fits_img(imgs_in_a_trk[0])
            img2, wcs2 = load_norm_fits_img(imgs_in_a_trk[1])
            img3, wcs3 = load_norm_fits_img(imgs_in_a_trk[2])

            #print(f'CHECK VALUE RANGE : {np.max(template),np.min(template)}')
            box_size = 20


            # ======= Convert radec to pixels ======= #
            y_list_template, x_list_template  = wcs1.world_to_pixel(pos_sample)
            y_list_img2, x_list_img2  = wcs2.world_to_pixel(pos_sample)
            y_list_img3, x_list_img3  = wcs3.world_to_pixel(pos_sample)

            #print(y_list_template, x_list_template)
            fig.suptitle('Positive images')

            for idx in range(len(x_list_template)):



                y_template, x_template = int(y_list_template[idx]), int(x_list_template[idx])
                box_template = template[int(x_template-box_size/2):int(x_template+box_size/2),int(y_template-box_size/2):int(y_template+box_size/2)]
                #print(f'{int(x_template),int(y_template)}')

                #plot_raw_image(box_template,ax=ax)
                #plot_noise_sub_image(box_template,ax=ax)
                #print(np.shape(box_template))


                # ================================================================= #

                y_img2, x_img2 = int(y_list_img2[idx]), int(x_list_img2[idx])
                box_img2 = img2[int(x_img2-box_size/2):int(x_img2+box_size/2),int(y_img2-box_size/2):int(y_img2+box_size/2)]


                #plot_raw_image(box_img2,ax=ax)
                #plot_diff_image(template=box_template,target=box_img2,ax=ax)
                #plot_noise_sub_image(box_img2,ax=ax)
                #print(np.shape(box_img2))


                # ================================================================= #

                y_img3, x_img3 = int(y_list_img3[idx]), int(x_list_img3[idx])
                box_img3 = img3[int(x_img3-box_size/2):int(x_img3+box_size/2),int(y_img3-box_size/2):int(y_img3+box_size/2)]
                #print(np.shape(box_img3))
                #plot_raw_image(box_img3,ax=ax)
                #plot_noise_sub_image(box_img3,ax=ax)
                #plot_diff_image(template=box_template,target=box_img3,ax=ax)


                # ================================================================= #
                pos_src_box.append([box_template,box_img2,box_img3])
    return pos_src_box

def list_to_skycoord(input_dict):
    dict_out = dict()
    for idx, candidates in input_dict.items():
        candidates_arr = np.array(candidates)
        dict_out[idx] = SkyCoord(candidates_arr[:,0],candidates_arr[:,1], unit=(u.deg, u.deg), frame='fk5')
        
    return dict_out


def norm_0_to_1(image_arr):
    out_img = np.empty(np.shape(image_arr))
    for dim in range(np.shape(image_arr)[0]):
        norm_img = image_arr[dim]
        norm_img -= np.min(norm_img) 
        norm_img /= np.max(norm_img) 
        out_img[dim] = norm_img
    return out_img


def Standard_scaler(image_arr):
    out_img = np.empty(np.shape(image_arr))
    for dim in range(np.shape(image_arr)[0]):
        
        # out = (arr-mean)/std
        
        stdscl_img = image_arr[dim]
        stdscl_img -= np.mean(stdscl_img) 
        stdscl_img /= np.std(stdscl_img) 
        out_img[dim] = stdscl_img
    return out_img

def get_p2p(trk_imgs):
    patch_size = np.shape(trk_imgs)[1]
    ptp_img = np.empty([1, patch_size, patch_size])
    
    
    # get first 3 channel as original must be chosen
    ptp_img[0] = np.ptp(trk_imgs[:3], axis=0)
    
    #print(np.shape(ptp_img))
    return ptp_img       

def get_similarity(trk_imgs):
    patch_size = np.shape(trk_imgs)[1]
    cosine_sim_imgs = np.empty([2, patch_size, patch_size])
    template = trk_imgs[0]
    img2 = trk_imgs[1]
    img3 = trk_imgs[2]
 
    
    cosine_sim_imgs[0] = np.dot(template,img2)/(norm(template)*norm(img2))
    cosine_sim_imgs[1] = np.dot(template,img3)/(norm(template)*norm(img3))
    
    return cosine_sim_imgs

def get_diff_imgs(trk_imgs):
    patch_size = np.shape(trk_imgs)[1]
    diff_imgs = np.empty([2, patch_size, patch_size])
    template = trk_imgs[0]
    
    diff_imgs[0] = trk_imgs[1]-trk_imgs[0]
    diff_imgs[1] = trk_imgs[2]-trk_imgs[0]
    
    return diff_imgs 


def get_subtraction_stamp(trk_imgs):
    patch_size = np.shape(trk_imgs)[1]
    diff_imgs = np.empty([2, patch_size, patch_size])
    template = trk_imgs[0]
    
    diff_imgs[0] = trk_imgs[1]-trk_imgs[0]
    diff_imgs[1] = trk_imgs[2]-trk_imgs[0]
    
    return diff_imgs 

def get_bramich_diff_imgs(trk_imgs):
    patch_size = np.shape(trk_imgs)[1]
    diff_imgs = np.empty([2, patch_size, patch_size])
    #template = trk_imgs[0]
    
    diff_imgs[0] = optimal_system(trk_imgs[1], trk_imgs[0])[0]#trk_imgs[1]-trk_imgs[0]
    diff_imgs[1] = optimal_system(trk_imgs[2], trk_imgs[0])[0]#trk_imgs[2]-trk_imgs[0]
    
    return diff_imgs 

def sep_standard_scaler(input_tensor):
    
    print(input_tensor.size)
    data = input_tensor.clone().numpy()
    print(data.shape)
    
    # image 1
    bkg = sep.Background(data)
    mean = bkg.globalback
    std = bkg.globalrms

    
    input_tensor = (input_tensor-mean)/std
    input_tensor = torch.clamp(a, min=-5, max=5)
    return input_tensor

# Last update 28/02/2023 : get parmas from template
def tensor_standard_scaler(input_tensor):
    
    mean = input_tensor[0].mean().item()
    std = input_tensor[0].std().item()
    
    input_tensor = (input_tensor-mean)/std
    
    
    return input_tensor
def tensor_L2Norm(input_tensor):
    
    input_norm = torch.norm(input_tensor[0]).item()
    # std = input_tensor[0].std().item()
    
    input_tensor = (input_tensor)/input_norm
    
    
    return input_tensor

def tensor_L2Norm_SupCon(input_tensor):
    
    
    # adjust minimum
    min_val = np.min([torch.min(input_tensor[0]).item(),torch.min(input_tensor[1]).item()])
    
    input_tensor = input_tensor-min_val
        
        
    input_norm = np.max([torch.norm(input_tensor[0]).item(),torch.norm(input_tensor[1]).item()])
    # std = input_tensor[0].std().item()
    
    input_tensor = (input_tensor)/input_norm
    
    
    return input_tensor

def tensor_norm_scaler(input_tensor):
    
    
    min_val = input_tensor.min().item()
    
    input_tensor = input_tensor-min_val
    
    max_val = input_tensor.max().item()
    
    input_tensor = input_tensor/max_val
    
    return input_tensor

def is_box_in_range(boxes, x_range, y_range):
    
    y_c, x_c, __, _ = boxes
    
    if not x_range[0] < x_c <  x_range[1]:
        
        return False
    
    if not y_range[0] < y_c <  y_range[1]:
        
        return False
    
    return True

def replace_root_path(path):
    new_root= '../../../../fits_images/'
    base_name = os.path.basename(path)
    #print('CHECK FUNCTION RESULTS')
    #print(path)
    patrn = r'[0-9]+-[0-9]+-[0-9]+'
    match = re.search(patrn, path).group()
    
    year, month, day = match.split('-')
    
    
    
    
    year_month = '_'.join([year, month])
    
    outpath = os.path.join(new_root,year_month,'img',day,base_name)
    
    if os.path.exists(outpath):
        return outpath
    else:
        # day+1
        day = int(day)+1
        day = f'{day:02}'
        outpath = os.path.join(new_root,year_month,'img',day,base_name)
        return outpath