# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

log:
    2023.06.09 Received "Type1_Track_v7.ipynb" from Yinglin.
    2023.06.13 Add tkinter for video file reading and plots/Excel saving in the selected folder.
    2023.06.16 Modified the "red", "green" to match the corresponding raw data channels.
               Also change the channel number from two to three to match the raw data I got.
    2023.06.20 Add subplot showing derivative
    2023.06.21 Save table for data selection
    2023.06.22 Set criteria.ver1 and allow polts being saved in different folders
    2023.06.23 Cleaning the code and test with no saponin data
    2023.06.28 Combine "HMM fitting_v2 with Track_v7"
    2023.10.31 Modify the function "contrast_img" so the mutable input image won't be changed.
"""

# Imports all necessary modules

import math
import warnings
warnings.filterwarnings('ignore')
import napari # napari
import matplotlib.pyplot as plt  # matplotlib
import numpy as np
import pandas as pd
import trackpy as tp  # trackpy
import tkinter as tk  # tkinter
from tkinter import filedialog
from tkinter.filedialog import askdirectory
import os
import pims  # pims
import random
import cv2  # opencv-python
from PyQt5 import QtWidgets  # PyQt5
from scipy import signal
from scipy.optimize import curve_fit
from hmmlearn import hmm
import time
# import multiprocessing


#%% Input values
# import tkinter as tk

# # Top level window
# frame = tk.Tk()
# frame.title("TextBox Input")
# frame.geometry('400x200')
# # Function for getting Input
# # from textbox and printing it
# # at label widget

# def printInput():
# 	inp = inputtxt.get(1.0, "end-1c")
# 	lbl.config(text = "Provided Input: "+inp)
    

# # TextBox Creation
# inputtxt = tk.Text(frame,
# 				height = 5,
# 				width = 20)

# inputtxt.pack()

# # Button Creation
# printButton = tk.Button(frame,
# 						text = "Print",
# 						command = printInput)
# printButton.pack()

# # Label Creation
# lbl = tk.Label(frame, text = "aa")
# lbl.pack()
# frame.mainloop()
# #%%
''' 1. Experimental parameters'''

RedCh = 1 # Red channel
FreCh = 2 # FRET channel
GreCh = 2 # Green channel
AddSap = 10 # The frame number when saponin was added
sec_per_f = 5 # how many sec per frame


''' 2. Tracking parameters '''
Diameter = 7     # in pixels. Represents the minimum diameter of the the feature in microns.
Min_mass = 400     # in total brightness. This is the minimum integrated brightness of the feature.
Separation = 3      # in pixels. The minimum separation between two features in microns.
Percentile = 97     # in percent. Features must have a peak brightness higher than pixels in this percentile to eliminate spurious points.
Max_distance = 5    # in pixels. Maximum distance in microns that features can move between frames to be considered the same feature
Memory = 5         # in frames. Maximum number of frames in which a feature can disappear, then reappear nearby, and be considered the same particle
Threshold = 20      # in frames. Minimum number of points for a track to be considered.

''' 3. Subtract the background '''
Background_radius = 1   #in pixels

''' 4. Stitch together tracks '''
Frame_diff = 4      # in frames
Location_diff = 4   # in pixels

''' 5. Data analysis '''
saveplot = 'y'  # y for yes and n for no
Lo_clus_r = 10      # limit of cluster/green_inten_mean

''' 6. excel filename '''
File_name = "20220405_CPSF6_10uM_01_1_R3D_merge_test3"

# In[] Read video (tiff format)
# Read_video  = r"C:\Users\YLU263\OneDrive - Emory University\TrackPy\Raw data\sqv01_R3D.tif"  # Lab Video

Read_video = filedialog.askopenfilename(filetypes=(("Image files", ("*.jpg","*.jpeg","*.tif","*.tiff","*.png","*.bmp")), ("All files", "*.*")), title='Open Video File')
reader = pims.open(Read_video)


totCh = max([GreCh, FreCh, RedCh]) # number of channels
red = np.array(reader[(RedCh-1)::totCh])    # mCherry in the first channel. Red channel repeats every 3 frames.
fret = np.array(reader[(FreCh-1)::totCh])   # FRET in the second channel. Repeats every 3 frames.
green = np.array(reader[(GreCh-1)::totCh])  # YFP in the third channel.
both = np.array(reader)

print('The file has been imported sucessfully')
print(f'There are {str(red.shape[0])}  frames with {totCh} color channels')


#%% TrackPy locates the spots
def contrast_img(img,min_,max_):
    img1 = img.copy()
    img1[img1>max_]=max_
    img1[img1<(min_)]=min_
    img1 -= min_
    return img1

rand1 = random.randrange(1,both.shape[0],totCh)
rand2 = random.randrange(1,both.shape[0],totCh)

img = contrast_img(green[0], np.min(green[0]), np.mean(green[0])*10)
img1 = contrast_img(reader[rand1], np.min(reader[rand1]), np.mean(reader[rand1])*5)
img2 = contrast_img(reader[rand2], np.min(reader[rand2]), np.mean(reader[rand2])*5)
img

# Do not edit any of the below code
# Diameter = 2*math.floor(Diameter/Dimension*green[0].shape[0]/2)+1   # Converts diameter to nearest odd pixel value
# Separation = Separation/Dimension*green[0].shape[0]                 # Converts separation to pixel
# Max_distance = Max_distance/Dimension*green[0].shape[0]             # Converts max_distance to pixel

particles0 = tp.locate(green[0], Diameter, minmass=Min_mass, separation=Separation, percentile=Percentile)
particles1 = tp.locate(green[int(rand1/totCh)], Diameter, minmass=Min_mass, separation=Separation, percentile=Percentile)
particles2 = tp.locate(green[int(rand2/totCh)], Diameter, minmass=Min_mass, separation=Separation, percentile=Percentile)

img3=tp.preprocessing.bandpass(reader[rand1], lshort=3, llong=11, threshold=1, truncate=4)
img4 = contrast_img(img3, np.min(reader[rand1]), np.mean(reader[rand1])*2)

with plt.rc_context({'figure.facecolor':'white'}):
    plt.figure(figsize=(9,9), dpi=300)
    print(tp.annotate(particles0, img, plot_style={'markersize': 9}))
    plt.figure(figsize=(9,9), dpi=300)
    print(tp.annotate(particles2, img2, plot_style={'markersize': 9}))
    # find frequency of pixels in range 0-255
plt.hist(img.ravel(), 100, label='img')
# plt.ylim(0, len(reader[3].ravel())/2)
plt.yscale('log', base = 10)
plt.legend()
plt.show()

# %% Performs location on many images in batch

=stop_here=
'''Not sure why this cannot run in cell... Select the line and press "F9" to run instead'''

particles = tp.batch(green, Diameter, minmass=Min_mass, separation=Separation, percentile=Percentile)

#%% Tracking
tracks = tp.link_df(particles, Max_distance, memory=Memory)
tracks = tp.filter_stubs(tracks,threshold =Threshold)

selected_columns = tracks[["x","y","signal","size","frame","particle"]]
tracks = selected_columns.copy()
number = list(range(0,tracks.shape[0]))
number = np.array(number)
tracks['number'] = number.tolist()
tracks = tracks.set_index('number')
tracks = tracks.sort_values(['frame','particle'])
tracks = tracks.reset_index()
tracks = tracks.drop(columns=['number'])
tracks
print(tracks)

n = pd.unique(tracks['particle']).shape[0]     
f = tracks[tracks['frame']==0].shape[0]
q = round(f/n *100,2)
print('There are ' + str(n) + ' identified tracks, and ' + str(f) + ' (' + str(q) + '%) of those begin in the first frame.')

plt.figure(figsize=(10,10), dpi=300)
tp.plot_traj(tracks,superimpose=img)

#%% Mask the images and subtract the background

# =============================================================================
# This mask is generated from the previously obtained tracks and will be applied 
# over the green channel and red channel both to ensure similarity between how 
# the particles are identified in the images. 
# In addition, background subtraction will be implemented to verify that the 
# intensity values are not affected by the background. The user can edit the 
# first line below to determine how large the background around each particle is. 
# =============================================================================
# Do not edit any of the below code
# Calculates intensity of green and red channels at each green particle location
i=0
green_intensity_mean = []
red_intensity_mean = []
green_intensity_max = []
red_intensity_max = []
green_intensity_total = []
red_intensity_total = []
pixel_count = []
green_background_mean = []
red_background_mean = []
tracks2 = tracks

for a in range(tracks.shape[0]):
    if i == tracks2['frame'][a]:
        pass
    else:
        i+=1
    mask = np.zeros(green[0].shape)
    center = (round(tracks['x'][a]),round(tracks['y'][a]))
    radius = tracks['size'][a]/2
    cv2.circle(mask,center,math.ceil(radius),255,-1)
    where = np.where(mask==255)
    green_signal = green[i][where[0],where[1]]
    red_signal = red[i][where[0],where[1]]
    green_signal_mean = np.mean(green_signal)
    red_signal_mean = np.mean(red_signal)
    green_signal_max = green_signal.max()
    red_signal_max = red_signal.max()
    green_signal_total = green_signal.sum()
    red_signal_total = red_signal.sum()
    outside = cv2.circle(mask,center,math.ceil(radius)+Background_radius,255,-1)
    inside = cv2.circle(outside,center,math.ceil(radius),0,-1)
    where2 = np.where(inside==255)
    green_back = green[i][where2[0],where2[1]]
    red_back = red[i][where2[0],where2[1]]
    green_back_mean = np.mean(green_back)
    red_back_mean = np.mean(red_back)
    green_inten_mean = green_signal_mean-green_back_mean
    red_inten_mean = red_signal_mean-red_back_mean
    green_inten_max = green_signal_max-green_back_mean
    red_inten_max = red_signal_max-red_back_mean
    green_inten_total = green_signal_total-(green_back_mean*green_signal.size)
    red_inten_total = red_signal_total-(red_back_mean*red_signal.size)
    green_intensity_mean.append(green_inten_mean)
    red_intensity_mean.append(red_inten_mean)
    green_intensity_max.append(green_inten_max)
    red_intensity_max.append(red_inten_max)
    green_intensity_total.append(green_inten_total)
    red_intensity_total.append(red_inten_total)
    green_background_mean.append(green_back_mean)
    red_background_mean.append(red_back_mean)
    pixel_count.append(green_signal.size)

tracks2['Mean Green Intensity'] = green_intensity_mean
tracks2['Mean Red Intensity'] = red_intensity_mean
tracks2['Max Green Intensity'] = green_intensity_max
tracks2['Max Red Intensity'] = red_intensity_max
tracks2['Total Green Intensity'] = green_intensity_total
tracks2['Total Red Intensity'] = red_intensity_total
tracks2['Green Background Mean'] = green_background_mean
tracks2['Red Background Mean'] = red_background_mean
tracks2['Number of Pixels'] = pixel_count
# tracks2['Size']=tracks2['size']*Dimension/green[0].shape[0]  # Converts size to microns
tracks2 = tracks2.drop(columns=['signal','size'])
tracks2 = tracks2.rename(columns={'x':'X Position', 'y':'Y Position','frame':'Frame','particle':'Particle','size':'Size [Pixels]'})
tracks2


#%% Stitch together tracks

# Do not edit any of the below code
# Stitches together particles that are within certain number of frames and pixels from one another
# Location_diff = Location_diff/Dimension*green[0].shape[0]  # Converts x- and y- locations to pixels

for times in range(10):
    first_frames = tracks2.drop_duplicates(subset=['Particle'],keep='first')
    last_frames = tracks2.drop_duplicates(subset=['Particle'],keep='last')
    first_frames = first_frames.reset_index()
    last_frames = last_frames.reset_index()

    stitch1 = last_frames.iloc[: 0]
    stitch2 = first_frames.iloc[: 0]
    for a in range(first_frames.shape[0]):
        for b in range(last_frames.shape[0]):
            if first_frames['Frame'][a]-last_frames['Frame'][b] <= Frame_diff and first_frames['Frame'][a]-last_frames['Frame'][b] >= 0:
                if abs(first_frames['X Position'][a]-last_frames['X Position'][b])<=Location_diff:
                    if abs(first_frames['Y Position'][a]-last_frames['Y Position'][b])<=Location_diff:
                        stitch1 = stitch1.append(last_frames.iloc[b])
                        stitch2 = stitch2.append(first_frames.iloc[a])
    stitch1 = stitch1.reset_index().drop(columns='index')
    stitch2 = stitch2.reset_index().drop(columns='index')

    stitch_1 = []
    stitch_2 = []
    for c in range(stitch1.shape[0]):
        stitch_1.append(int(stitch1['Particle'][c]))
        stitch_2.append(int(stitch2['Particle'][c]))

    for d in range(len(stitch_2)):
        tracks2['stitch tracks'] = tracks2['Particle'].replace(stitch_2[d],stitch_1[d])
        tracks2 = tracks2.drop(columns=['Particle'])
        tracks2 = tracks2.rename(columns={'stitch tracks':'Particle'})

n = pd.unique(tracks['particle']).shape[0] 
n2 = pd.unique(tracks2['Particle']).shape[0] 
print(str(n-n2)+' tracks were stitched together. There are now '+str(n2)+' total tracks.')
tracks2

#%% Classification and Plotting
if saveplot == 'y':
    # Save plots in a folder
    video_name = Read_video.split('/')[-1].split('.')[0]
    folder = askdirectory()
    new_folder = folder+"/" +video_name+str(2)
    # Creat folders for classification
    try: 
        os.mkdir(os.path.join(new_folder)) 
    except:
        pass
    
    try: 
        os.mkdir(os.path.join(new_folder, 'bad')) 
    except:
        pass
        
    try: 
        os.mkdir(os.path.join(new_folder, 'lysis')) 
    except:
        pass
       
    try: 
        os.mkdir(os.path.join(new_folder, 'uncoating')) 
    except:
        pass
    
    try: 
        os.mkdir(os.path.join(new_folder, 'Nonlytic')) 
    except:
        pass    

# Creates functions to round down and up to nearest hundred to define limits in the graph
def rounddown(x):
     return int(math.floor(x/100.0))*100
def roundup(x):
     return int(math.ceil(x/100.0))*100
# Creates function to count the number of data points in a tracks belows a set value
def BelowCount(data, bar):
    data = data.replace(np.nan, 0)
    a=[data[i] for i in range(len(data)) if data[i] < bar]
    return len(a)

def makeplot(x, y_G, y_R, y_HMM, y_Res):
    fig, (ax1, ax2) = plt.subplots(2, height_ratios=[2, 1])
    fig.tight_layout() ## Adjust the space between plots
    ax1.scatter(x*sec_per_f, y_G, color='g', label='Max Green Intensity')
    ax1.scatter(x*sec_per_f, y_R, color='r', label='Max Red Intensity')
    # ax1.plot(x*sec_per_f, R_smooth, color='k', label='Smoothed Max Red Intensity',linestyle ='--')
    ax1.plot(x*sec_per_f, y_HMM,  color='b', label= 'HMM fit')
    ax1.set_xlim(0,green.shape[0]*sec_per_f)
    # ax1.set_ylim(-5,110)
    ax1.legend()
    ax1.set_xlabel('Time (sec)')
    ax1.set_ylabel('Normalized Intensity (%)')
    ax1.set_title(PN)
    ax2.plot(range(green.shape[0]*sec_per_f), [i* 0 for i in range(green.shape[0]*sec_per_f)], color='grey', linestyle='--')
    ax2.plot(x*sec_per_f, y_Res, label='Residuals of HMM')
    # ax2.plot(temp_X, diff_R, label='Derivative of Max Red Intensity')
    # ax2.plot(temp_X, diff_Rs, label='Derivative of Smoothed Max Red Intensity', color='k', linestyle='--')
    ax2.legend()
    ax2.set_xlabel('Time (sec)')
    ax2.set_xlim(0,green.shape[0]*sec_per_f)
    ax2.set_ylim(-20,20)
    fig.set_size_inches(6., 6.)
    plot = plt.gcf()
    return plot

LoD = Lo_clus_r* green_inten_mean + green_back_mean
LoD = green_inten_mean
effective = []
lysis=[]
uncoating=[]
l_Env_t = []
u_Env_t = []
u_Cap_t = []
for particle in range(tracks2['Particle'].max()):
    if tracks2['Particle'][tracks2['Particle']==particle].shape[0]>Threshold:

        # Data cleaning
        temp = tracks2.loc[tracks2['Particle']==particle].sort_values(by=['Frame'])
        temp_X = temp['Frame'].reset_index(drop = True)
        temp_G = temp['Max Green Intensity'].reset_index(drop = True)
        Norm_G = temp_G/temp_G[0]*100     # HMM fitting fails when the y value is too small (not sure why)
        temp_R = temp['Max Red Intensity'].reset_index(drop = True)
        Norm_R = temp_R/temp_R[0]*100
        diff_R = temp_R.diff()
        diff_NR = Norm_R.diff()
        R_smooth = pd.DataFrame(signal.savgol_filter(temp_R, window_length=5, polyorder=3, mode="nearest"), columns = ['Smoothed Max Red']).reset_index(drop = True)
        diff_Rs = R_smooth.diff().squeeze()
        PN = 'Particle '+str(particle)
        
        NormLoD = LoD /max(temp_G)*100
        # try:
        #####========================HMM fitting==========================#####
        rrr= np.array(Norm_R) #Choose the dataset for HMM fitting
        bic=list()
        scores = list()
        models = list()
        maxres = list()
        for i in range(1,4):    
            scores2 = list()
            models2 = list()
            try:
                for idx in range(20):
                    temp_mean =np.array([[80], [20], [-10]])
                    model = hmm.GaussianHMM(n_components=i, covariance_type='diag', random_state=idx, n_iter=10)#, means_prior=temp_mean, means_weight=1) 
                    model.fit(rrr.reshape(-1, 1))
                    models2.append(model)
                    scores2.append(model.score(rrr.reshape(-1, 1)))
                model = models2[np.argmax(scores2)]
                bic.append(model.bic(rrr.reshape(-1, 1)))
                models.append(model)
                scores.append(model.score(rrr.reshape(-1, 1)))
                hmm_result = model.means_[model.predict(rrr.reshape(-1, 1))].reshape(len(rrr))
                # Residuals
                res1 = rrr - hmm_result
                maxres.append(abs(max(res1)))
            except:
                pass
            
            

            
        # get the best model (Use BIC instead of score to prevent overfit)           
        if len(bic) >2:
            if np.argmin(bic) == 1 and (bic[2]-bic[1])/bic[1]<0.05 and maxres[1]>40:
                evalu_b = models[2]
            else:
                evalu_b = models[np.argmin(bic)]
        else:
            evalu_b = models[np.argmin(bic)]
        # Plot the best model & count the transition time  
        hmm_result = evalu_b.means_[evalu_b.predict(rrr.reshape(-1, 1))].reshape(len(rrr))
        tran_num=0
        for j in range(len(hmm_result)):
            if j>=1 and hmm_result[j]!= hmm_result[j-1]:
                tran_num +=1
        # Residuals
        res = rrr - hmm_result
        #####=====================End of HMM fitting======================#####

        #####=======================Classification========================#####

        # Bad data
        if (BelowCount(temp_G, LoD)/len(temp_G) >0.99) or (max(hmm_result)/100*max(temp_R))<LoD or temp_X[0]>green.shape[0]/3:   
            ## delete the tarjectory that is too short, too many missing data, starts too late, or too weak.
        
            if saveplot == 'y':
                os.chdir(new_folder+'/bad')
                makeplot(temp_X, Norm_G, Norm_R, hmm_result, res)  
                print(PN, "Bad: Reason 1")
    
                plt.savefig(PN, bbox_inches="tight")
                plt.close() # figures won't be displayed. This will save time.
            else:
                pass
        # Bad data
        elif (tran_num>4 or tran_num<1) and (min(hmm_result[-5:])/100*max(temp_R))<LoD:
            
            if saveplot == 'y':
                os.chdir(new_folder+'/bad')
                makeplot(temp_X, Norm_G, Norm_R, hmm_result, res)  
                print(PN, "Bad: Reason 2")
                plt.savefig(PN, bbox_inches="tight")
                plt.close() # figures won't be displayed. This will save time.
            else:
                pass

        # Effective
        else:
            effective.append(int(particle))             
            #Lysis
            if tran_num == 1 and hmm_result[0]>NormLoD and hmm_result[-1]==min(hmm_result) and (min(hmm_result)/max(hmm_result))<0.3:
                
                temp = np.sort(evalu_b.means_.reshape(evalu_b.n_components))
                for i in range(len(hmm_result)):
                    if hmm_result[i] == temp[-1]:
                        Env = temp_X[i]
                
                print('lysis', PN)
                lysis.append(int(particle))
                l_Env_t.append(Env)
                if saveplot == 'y':
                    
                    os.chdir(new_folder+'/lysis')                        
                    makeplot(temp_X, Norm_G, Norm_R, hmm_result, res)  
                    plt.savefig(PN, bbox_inches="tight")
                    plt.close() # figures won't be displayed. This will save time.
                else:
                    pass
            
            # uncoating
            elif evalu_b.n_components ==3 and 1< tran_num < 5 and max(hmm_result[0:2])==max(hmm_result) and hmm_result[-1]==min(hmm_result):
                # find kinetic info
                temp = np.sort(evalu_b.means_.reshape(evalu_b.n_components))
                for i in range(len(hmm_result)):
                    if hmm_result[i] == temp[-1]:
                        Env = temp_X[i]
                    elif hmm_result[i] == temp[-2]:
                        Cap = temp_X[i]
                Cap = Cap-Env           
                # capsid opening time is defined as "the time between permibilization and integrity loss
                
                if Cap>0:
                    print('uncoating', PN)
                    uncoating.append(int(particle))
                    u_Env_t.append(Env)
                    u_Cap_t.append(Cap)
                
                    if saveplot == 'y':
                        
                        os.chdir(new_folder+'/uncoating')                        
                        makeplot(temp_X, Norm_G, Norm_R, hmm_result, res)  
                        plt.savefig(PN, bbox_inches="tight")
                        plt.close() # figures won't be displayed. This will save time.
                    else:
                        pass
                elif Cap <=0:
                    print('lysis', PN)
                    lysis.append(int(particle))
                    
                    if saveplot == 'y':
                        
                        os.chdir(new_folder+'/lysis')                        
                        makeplot(temp_X, Norm_G, Norm_R, hmm_result, res)  
                        plt.savefig(PN, bbox_inches="tight")
                        plt.close() # figures won't be displayed. This will save time.
                    else:
                        pass

            # Nonlytic
            else:
                if saveplot == 'y':
                    with plt.rc_context({'figure.facecolor':'white'}):
                        
                        os.chdir(new_folder+'/nonlytic')
                        makeplot(temp_X, Norm_G, Norm_R, hmm_result, res)       
                        plt.savefig(PN, bbox_inches="tight")
                        plt.close() # figures won't be displayed. This will save time.
                else:
                    pass
        # except:
        #     print('error: particle', particle)
        #     pass

l_Env_t1=[l_Env_t[i]*sec_per_f for i in range(len(l_Env_t))]    
u_Env_t1=[u_Env_t[i]*sec_per_f for i in range(len(u_Env_t))]
u_Cap_t1=[u_Cap_t[i]*sec_per_f for i in range(len(u_Cap_t))]    
    
    
plt.hist(l_Env_t1, label='lysis')
plt.legend()
plt.show()
plt.hist(u_Env_t1, label='Env', alpha = 0.8)     # plot histogram of permibilization time
plt.hist(u_Cap_t1, label='Cap', alpha = 0.6)     # plot histogram of (integrity loss time - permibilization time)
plt.legend()
plt.show()
     

print('================================',
      '\nFile: ', video_name, '\n--------------------------------'
      '\nPercentage of Lysis-only:' ,round(len(lysis)/len(effective)*100, 3), '%', 
      '\nPercentage of uncoating:' ,round(len(uncoating)/len(effective)*100, 3), '%', 
      '\n--------------------------------', '\nNumber of lysis-only:', len(lysis), 
      '\nNumber of uncoating:', len(uncoating), '\nNumber of effective:', len(effective), 
      '\n================================')      
if saveplot == 'y':
    ratio_info = ['================================', 
                  '\nFile: ', video_name, '\n--------------------------------', 
                  '\nPercentage of Lysis-only:' ,str(round(len(lysis)/len(effective)*100, 3)), '%', 
                  '\nPercentage of uncoating:' ,str(round(len(uncoating)/len(effective)*100, 3)), '%', 
                  '\n--------------------------------', '\nNumber of lysis-only:', str(len(lysis)), 
                  '\nNumber of uncoating:', str(len(uncoating)), '\nNumber of effective:', str(len(effective)), 
                  '\n================================',
                  '\nDiameter = ', str(Diameter), '\nMin_mass = ', str(Min_mass), '\nSeparation = ', str(Separation),
                  '\nPercentile = ', str(Percentile), '\nMax_distance = ', str(Max_distance),
                  '\nMemory = ', str(Memory), '\nThreshold = ', str(Threshold)]
    os.chdir(new_folder)
    with open('ratio_info.txt', 'w') as f:
        for line in ratio_info:
            f.write(line)
            # f.write('\n')
else:
    pass

     
#%% Kinetic info statistic
def expo(x, A, k, y0):
    return A * np.exp(-x/k) + y0 # exponatial fitting

counts, bins, bars = plt.hist(l_Env_t,bins=25)
plt.close()

# to find index of first element just 
# greater than K 
# using next() + enumerate
res = next(x for x, val in enumerate(bins) if val >counts.argmax())

surv=len(lysis)-np.append(0, np.cumsum(counts))
popt, pcov = curve_fit(expo, bins[:-(res)], surv[(res):], bounds = ([0,0,0], [max(surv)*3, 300, 15]))


plt.scatter(bins*sec_per_f,  surv, c='r', alpha = 0.7, label = 'Lysis only')
plt.plot(bins[(res):]*sec_per_f, expo(bins[:-(res)], *popt), label = 'Exp fitting')
plt.xlabel("Time (sec)")
plt.ylabel("Counts")
plt.legend()
plt.show()
print(f' lifetime = {round(popt[1]*sec_per_f,3)} sec (lysis only)')


counts1, bins1, bars1 = plt.hist(u_Env_t,bins=60)
counts2, bins2, bars2 = plt.hist(u_Cap_t,bins=35)
plt.close()

res1 = next(x for x, val in enumerate(bins1) if val >counts1.argmax())
res2 = next(x for x, val in enumerate(bins2) if val >counts2.argmax())

surv1=len(uncoating)-np.append(0, np.cumsum(counts1))
surv2=len(uncoating)-np.append(0, np.cumsum(counts2))
popt1, pcov1 = curve_fit(expo, bins1[:(len(bins1)-(res1))], surv1[(res1):], bounds = ([0,0,0], [max(surv1)*3, 300, 15]))
popt2, pcov2 = curve_fit(expo, bins2[:(len(bins2)-(res2))], surv2[(res2):], bounds = ([0,0,0], [max(surv2)*3, 300, 15]))
# use [:(len(bins2)-(res2))] to replace [:-(res2)] to avoid the bug appearing when res2 = 0

plt.scatter(bins1*sec_per_f,  surv1, c='r', alpha = 0.7, label = 'Lysis')
# plt.scatter(bins2*sec_per_f,  surv2, c='purple', alpha = 0.7, label = 'Uncoating')

plt.plot(bins1[(res1):]*sec_per_f, expo(bins1[:-(res1)], *popt1), label = 'Exp fitting')
# plt.plot(bins2[(res2):]*sec_per_f, expo(bins2[:-(res2)], *popt2), label = 'Exp fitting')

plt.xlabel("Time (sec)")
plt.ylabel("Counts")
plt.legend()
plt.show()
print(f' lifetime = {round(popt1[1]*sec_per_f,3)} sec (lysis) and  {round(popt2[1]*sec_per_f,3)} sec (uncoating)')

#%% Napari viewer and 
data = tracks2.loc[:, ['Particle','Frame','Y Position','X Position']]
data = data.to_numpy()

properties = tracks2.loc[:, ['Max Red Intensity', 'Max Green Intensity']]


viewer = napari.Viewer(title='Tracks')

viewer.add_image(np.array(green),blending='additive',colormap='green',name='Green')
viewer.add_image(np.array(red), blending='additive', colormap='red', name='Red')
viewer.add_tracks(data, properties=properties, name='tracks', tail_width=0.1, opacity=1, blending='opaque')

#%% plot intensity profiles for specific particles
particle_1 = 3
if tracks2['Particle'][tracks2['Particle']==particle_1].shape[0]>Threshold:
    
    # Data cleaning
    temp = tracks2.loc[tracks2['Particle']==particle_1].sort_values(by=['Frame'])
    temp_X = temp['Frame'].reset_index(drop = True)
    temp_G = temp['Max Green Intensity'].reset_index(drop = True)
    Norm_G = temp_G/max(temp_G)*100
    temp_R = temp['Max Red Intensity'].reset_index(drop = True)
    Norm_R = temp_R/max(temp_R)*100
    diff_R = temp_R.diff()
    diff_NR = Norm_R.diff()
    R_smooth = pd.DataFrame(signal.savgol_filter(temp_R, window_length=5, polyorder=3, mode="nearest"), columns = ['Smoothed Max Red']).reset_index(drop = True)
    diff_Rs = R_smooth.diff().squeeze()
    PN = 'Particle '+str(particle_1)
    
 
# np.random.seed(40)
rrr= np.array(R_smooth/max(R_smooth['Smoothed Max Red'].reset_index(drop = True))*100) #adjust data 
rrr= np.array(Norm_R) #adjust data 

# rrr=rrr.reshape(-1, 1)
bic=list()
scores = list()
models = list()
maxres = list()
try:
    for i in range(1,4):    
        scores2 = list()
        models2 = list()
        for idx in range(20):
            temp_mean =np.array([[80], [20], [-10]])
            model = hmm.GaussianHMM(n_components=i, covariance_type='diag', random_state=idx, n_iter=10)#, means_prior=temp_mean, means_weight=1) 
            model.fit(rrr.reshape(-1, 1))
            models2.append(model)
            scores2.append(model.score(rrr.reshape(-1, 1)))
        model = models2[np.argmax(scores2)]
        bic.append(model.bic(rrr.reshape(-1, 1)))
        models.append(model)
        scores.append(model.score(rrr.reshape(-1, 1)))
        hmm_result = model.means_[model.predict(rrr.reshape(-1, 1))].reshape(len(rrr))
        # Residuals
        res1 = rrr - hmm_result
        maxres.append(abs(max(res1)))
except:
    pass
    # hmm_result = model.means_[model.predict(rrr.reshape(-1, 1))].reshape(len(rrr))
        
    

# print("Chi square =", sum(res**2/hmm_result))
print(f'{model.n_components} states\t\t'
      f'Converged: {model.monitor_.converged}\t\t'
      f'Score: {scores[-1]}\t\t'
      f'BIC: {bic[-1]}\t\t'
      f'Chi square: {sum(res1**2/hmm_result)}')
    
# get the best model (Use BIC instead of score to prevent overfit)
if len(bic) >2:
    if np.argmin(bic) == 1 and (bic[2]-bic[1])/bic[1]<0.05 and maxres[1]>40:
        evalu_b = models[2]
        print('y')
    else:
        evalu_b = models[np.argmin(bic)]
        print('y->n')
else:
    evalu_b = models[np.argmin(bic)]
    print('n')
    
print(f'The best model had a BIC value of {evalu_b.bic} and '
      f'{evalu_b.n_components} components') 
print("Number of hidden states", evalu_b.n_components)  #
print("Mean value")
print(evalu_b.means_)
print("Start probability")
print(evalu_b.startprob_)
print("State Transition Matrix")
print(evalu_b.transmat_)

# Plot the best model   
hmm_result = evalu_b.means_[evalu_b.predict(rrr.reshape(-1, 1))].reshape(len(hmm_result))
tran_num=0
for j in range(len(hmm_result)):
    if j>=1 and hmm_result[j]!= hmm_result[j-1]:
        tran_num +=1
# Residuals
res = rrr - hmm_result


# find kinetic info

temp = np.sort(evalu_b.means_.reshape(evalu_b.n_components))
for i in range(len(hmm_result)):
    if hmm_result[i] == temp[-1]:
        Env = temp_X[i]
    elif hmm_result[i] == temp[-2]:
        Cap = temp_X[i]
        
Cap = Cap-Env

    
# print("Chi square =", sum(res**2/hmm_value))
# makeplot(temp_X, Norm_R, Norm_R, hmm_result, res)
# =============================================================================
# # This part is for testing and maintainance
# =============================================================================
fig, (ax1, ax2) = plt.subplots(2, height_ratios=[2, 1])
fig.tight_layout() ## Adjust the space between plots
ax1.plot(temp_X, Norm_G, color='g', label='Max Green Intensity')
ax1.scatter(temp_X, rrr, color='r', label='Max Red Intensity')
# ax1.bar(temp_X, rrr, color='r', label='Max Red Intensity')

# ax1.plot(temp_X, R_smooth, color='k', label='Smoothed Max Red Intensity',linestyle ='--')
ax1.plot(temp_X, hmm_result,  color='b', label= 'HMM fit')
ax1.set_xlim(0,green.shape[0])
ax1.set_xlim(0,100)

# ax1.set_ylim(0,4200)

ax1.legend()
ax1.set_ylabel('Intensity')
ax1.set_title(PN)
ax2.plot(range(green.shape[0]), [i* 0 for i in range(green.shape[0])], color='grey', linestyle='--')
ax2.plot(temp_X, res, label='Residuals')
# ax2.plot(temp_X, diff_R, label='Derivative of Max Red Intensity')
# ax2.plot(temp_X, diff_Rs, label='Derivative of Smoothed Max Red Intensity', color='k', linestyle='--')
ax2.set_title('Residuals of HMM fitting')
ax2.set_xlim(0,green.shape[0])
# ax2.legend(loc='upper right')
fig.set_size_inches(6., 6.)
# =============================================================================
# =============================================================================
plt.show()

# print("Chi square =", sum(res**2/hmm_value))
# plt.plot(hmm_result)
# plt.title("States: "+str(f'{evalu_b.n_components}'))



#%% Save excel file
# Excel_path = r"C:/Users/YLU263" 
# File_name = "20220405_CPSF6_10uM_01_1_R3D_merge_test3"

writer = pd.ExcelWriter(folder+'/'+video_name+'.xlsx',engine='xlsxwriter')
tracks3 = tracks2.drop(columns=['Max Green Intensity', 'Max Red Intensity', 'Total Green Intensity', 'Total Red Intensity',
                                'Green Background Mean', 'Red Background Mean', 'Number of Pixels'])
for a in range(tracks3['Particle'].max()):
    subset = tracks3[tracks3['Particle']==a]
    if subset.shape[0] > 30:     # Only exports particles that are present for 30 frames
      subset.to_excel(writer,sheet_name='Particle'+str(a), index=False) 
      
writer.save()   


data_trace = pd.DataFrame()
PN_list=[]
for particle in range(tracks2['Particle'].max()):
    if tracks2['Particle'][tracks2['Particle']==particle].shape[0]>Threshold:
        with plt.rc_context({'figure.facecolor':'white'}):
            
            # Data cleaning
            temp = tracks2.loc[tracks2['Particle']==particle].sort_values(by=['Frame'])
            temp_X = temp['Frame'].reset_index(drop = True)
            temp_G = temp['Max Green Intensity'].reset_index(drop = True)
            temp_R = temp['Max Red Intensity'].reset_index(drop = True)
            diff_R = temp_R.diff().reset_index(drop = True)
            R_smooth = pd.DataFrame(signal.savgol_filter(temp_R, window_length=5, polyorder=3, mode="nearest"), columns = ['Smoothed Max Red'])
            diff_Rs = R_smooth.diff()

            # Keep data in a table for output and analysis
            PN = 'Particle '+str(particle)
            temp2 = pd.concat([temp_X, temp_G, temp_R, R_smooth, diff_R, diff_Rs], axis=1)
            temp2.columns= [PN+' Frame', PN+' Max Green Int', PN+' Max Red Int', PN+ ' Smoothed Red', PN+ ' Red Deriv', PN+ ' Smoothed Red Deriv']
            temp2=temp2.set_index([PN+' Frame'], drop=True)
            data_trace = pd.concat([data_trace, temp2], axis=1)
            PN_list.append(int(particle))



print("done")