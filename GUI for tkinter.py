# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:43:43 2024

@author: YLU263
"""

import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askdirectory
import os
os.environ["OMP_NUM_THREADS"] = "1" 
import matplotlib.pyplot as plt  # matplotlib
import trackpy as tp  # trackpy
import numpy as np
import pandas as pd
import pims  # pims
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import cv2  # opencv-python
from PyQt5 import QtWidgets  # PyQt5
from scipy import signal
from scipy.optimize import curve_fit
from hmmlearn import hmm
import time
import math





# Global dictionary to store the inputs
inputs = {}

def open_file():
    global filename, video_name
        
    filename = filedialog.askopenfilename(filetypes=(("Image files", ("*.jpg","*.jpeg","*.tif","*.tiff","*.png","*.bmp")), ("All files", "*.*")), title='Open Video File')
    entries[0].delete(0, tk.END)
    entries[0].insert(0, filename)
    video_name = filename.split('/')[-1].split('.')[0]
    entries[17].delete(0, tk.END)
    entries[17].insert(0, video_name)
    

def toggle_entry_state(index, checkbox_var, entry):
    if checkbox_var.get():
        entry.config(state=tk.NORMAL)
    else:
        entry.delete(0, tk.END)
        entry.config(state=tk.DISABLED)
        entry.insert(0, "default value")

def submit():
    global inputs, GreCh, FreCh, RedCh, AddSap, sec_per_f, Diameter, Min_mass, \
        Separation, Percentile, Max_distance, Memory, Threshold, Background_radius, \
            Frame_diff, Location_diff, Lo_clus_r, Tr_targ, Fit_targ, saveplot, File_name
    
    variable_names = [
        "File", "Red channel", "FRET channel", "Green channel", "Adding saponin at frame",
        "Second per frame", "Diameter", "Minimum intensity", "Separation", "Percentile",
        "Max distance per frames", "Memory", "Minimum length", "Background radius",
        "Frame_diff", "Location_diff", "Limit of detection factor", "Track target", 
        "Fit target","Saveplot", "New filename"
    ]
    
    for i, entry in enumerate(entries[:-1]):
        inputs[variable_names[i]] = entry.get()
        
    inputs["Track target"] = track_var.get()    
    inputs["Fit target"] = Fit_var.get()    
    inputs["Saveplot"] = saveplot_var.get()
    inputs["New filename"] = entry_extra.get()
    
    # if checkbox_var.get():
    #     inputs["New filename"] = entries[18].get()
    # else:
    #     inputs["New filename"] = video_name
    
    print("Submitted values:")
    for key, value in inputs.items():
        print(f"{key}: {value}")
    
    [GreCh, FreCh, RedCh, AddSap, sec_per_f, Diameter, Min_mass, Separation, 
     Percentile, Max_distance, Memory, Threshold, Background_radius, Frame_diff, 
     Location_diff, Lo_clus_r, Tr_targ, Fit_targ, saveplot, File_name] = [
         int(inputs["Green channel"]), int(inputs["FRET channel"]), int(inputs["Red channel"]), 
         int(inputs["Adding saponin at frame"]), int(inputs["Second per frame"]), 
         int(inputs["Diameter"]), int(inputs["Minimum intensity"]), int(inputs["Separation"]), 
         int(inputs["Percentile"]), int(inputs["Max distance per frames"]), 
         int(inputs["Memory"]), int(inputs["Minimum length"]), int(inputs["Background radius"]), 
         int(inputs["Frame_diff"]), int(inputs["Location_diff"]), int(inputs["Limit of detection factor"]), 
         inputs["Track target"], inputs["Fit target"],inputs["Saveplot"], inputs["New filename"]
        ]


    
def finish():
    root.destroy()  # Close the window and terminate the program
    root.quit()

def contrast_img(img,min_,max_):
    img1 = img.copy()
    img1[img1>max_]=max_
    img1[img1<(min_)]=min_
    img1 -= int(min_)
    return img1

def loadimage():
    global red, fret, green, img, particles0, Tr_Ch
    # Create a sample plot using matplotlib
    Read_video = inputs["File"]
    reader = pims.open(Read_video)
    totCh = max([GreCh, FreCh, RedCh]) # number of channels
    red = np.array(reader[(RedCh-1)::totCh])    # mCherry in the first channel. Red channel repeats every 3 frames.
    fret = np.array(reader[(FreCh-1)::totCh])   # FRET in the second channel. Repeats every 3 frames.
    green = np.array(reader[(GreCh-1)::totCh])  # YFP in the third channel.
    both = np.array(reader)
    if Tr_targ =="red":
        Tr_Ch = red
    elif Tr_targ =="fret":
        Tr_Ch = fret
    elif Tr_targ =="green":
        Tr_Ch = green

    FrameN = int(frame_entry.get())
    if FrameN>=np.size(Tr_Ch,0):
        FrameN = np.size(Tr_Ch,0)-1
        frame_entry.delete(0, tk.END)
        frame_entry.insert(0, FrameN)
        
    try:
        MinInt = float(Min_entry.get())
        MaxInt = float(Max_entry.get())
    except:
        MinInt = np.min(Tr_Ch[FrameN])
        Min_entry.delete(0, tk.END)
        Min_entry.insert(0, MinInt)
        MaxInt = np.mean(Tr_Ch[FrameN])*2
        Max_entry.delete(0, tk.END)
        Max_entry.insert(0, MaxInt)
    particles0 = tp.locate(Tr_Ch[FrameN], Diameter, minmass=Min_mass, separation=Separation, percentile=Percentile)
    img = contrast_img(Tr_Ch[FrameN], MinInt, MaxInt)

def fig_show(fig):  # Display the plot in the Tkinter canvas
    canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
    canvas.draw_idle()
    canvas.get_tk_widget().grid(row=0, column=4, rowspan=20, columnspan=14, padx=1, pady=1)
    plt.close(fig)
    # navigation toolbar
    toolbarFrame = tk.Frame(master=root)
    toolbarFrame.grid(row=0,column=4, columnspan=5)
    toolbar = NavigationToolbar2Tk(canvas, toolbarFrame, pack_toolbar=False)
    toolbar.grid()

    
def RawImg():
    with plt.rc_context({'figure.facecolor':'white'}):
          fig, ax = plt.subplots(figsize=(6.5,6.5), dpi=100)
          tp.annotate(particles0, img, plot_style={'markersize': 0}, ax=ax)
    # Display the plot in the Tkinter canvas
    fig_show(fig)

def ParImg():
    with plt.rc_context({'figure.facecolor':'white'}):
          fig, ax = plt.subplots(figsize=(6.5,6.5), dpi=100)
          tp.annotate(particles0, img, plot_style={'markersize': 7}, ax=ax)
    # Display the plot in the Tkinter canvas
    fig_show(fig)
    
def track():
    global particles, tracks
    particles = tp.batch(Tr_Ch, Diameter, minmass=Min_mass, separation=Separation, percentile=Percentile,  processes=1)
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
    message = 'There are ' + str(n) + ' identified tracks, and ' + str(f) + ' (' + str(q) + '%) of those begin in the first frame.'
    disp(message)

    with plt.rc_context({'figure.facecolor':'white'}):
          fig, ax = plt.subplots(figsize=(6.5,6.5), dpi=100)
          tp.plot_traj(tracks,superimpose=img, ax=ax)

    # Display the plot in the Tkinter canvas
    fig_show(fig)
    # Enable the process function
    process_button.config(state=tk.NORMAL)


def process():
    global tracks2, lysis, effective, uncoating, inte_loss
    # Mask the images and subtract the background
    # =============================================================================
    # This mask is generated from the previously obtained tracks and will be applied 
    # over the green channel and red channel both to ensure similarity between how 
    # the particles are identified in the images. 
    # In addition, background subtraction will be implemented to verify that the 
    # intensity values are not affected by the background. The user can edit the 
    # first line below to determine how large the background around each particle is. 
    # =============================================================================
    # Do not edit any of the below code
    # Calculates intensity of red and red channels at each red particle location
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
        mask = np.zeros(Tr_Ch[0].shape) # Based on the assigned channel "Tr_Ch"
        center = (round(tracks['x'][a]),round(tracks['y'][a]))
        radius = tracks['size'][a]/2
        cv2.circle(mask,center,math.ceil(radius),255,-1)
        where = np.where(mask==255)
        green_signal =  green[i][where[0],where[1]]
        red_signal =    red[i][where[0],where[1]]
        green_signal_mean = np.mean(green_signal)
        red_signal_mean =   np.mean(red_signal)
        green_signal_max =  green_signal.max()
        red_signal_max =    red_signal.max()
        green_signal_total =    green_signal.sum()
        red_signal_total =      red_signal.sum()
        outside = cv2.circle(mask,center,math.ceil(radius)+Background_radius,255,-1)
        inside = cv2.circle(outside,center,math.ceil(radius),0,-1)
        where2 = np.where(inside==255)
        green_back =    green[i][where2[0],where2[1]]
        red_back =      red[i][where2[0],where2[1]]
        green_back_mean =   np.mean(green_back)
        red_back_mean =     np.mean(red_back)
        green_inten_mean =  green_signal_mean-green_back_mean
        red_inten_mean =    red_signal_mean-red_back_mean
        green_inten_max =   green_signal_max-green_back_mean
        red_inten_max =     red_signal_max-red_back_mean
        green_inten_total = green_signal_total-(green_back_mean*green_signal.size)
        red_inten_total =   red_signal_total-(red_back_mean*red_signal.size)
        green_intensity_mean.append(green_inten_mean)
        red_intensity_mean.append(red_inten_mean)
        green_intensity_max.append(green_inten_max)
        red_intensity_max.append(red_inten_max)
        green_intensity_total.append(green_inten_total)
        red_intensity_total.append(red_inten_total)
        green_background_mean.append(green_back_mean)
        red_background_mean.append(red_back_mean)
        pixel_count.append(green_signal.size)

    tracks2['Mean Green Intensity'] =   green_intensity_mean
    tracks2['Mean Red Intensity'] =     red_intensity_mean
    tracks2['Max Green Intensity'] =    green_intensity_max
    tracks2['Max Red Intensity'] =      red_intensity_max
    tracks2['Total Green Intensity'] =  green_intensity_total
    tracks2['Total Red Intensity'] =    red_intensity_total
    tracks2['Green Background Mean'] =  green_background_mean
    tracks2['Red Background Mean'] =    red_background_mean
    tracks2['Number of Pixels'] = pixel_count
    # tracks2['Size']=tracks2['size']*Dimension/green[0].shape[0]  # Converts size to microns
    tracks2 = tracks2.drop(columns=['signal','size'])
    tracks2 = tracks2.rename(columns={'x':'X Position', 'y':'Y Position','frame':'Frame','particle':'Particle','size':'Size [Pixels]'})

    # Stitch together tracks
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
    message = str(n-n2)+' tracks were stitched together. There are now '+str(n2)+' total tracks.'
    disp(message)


    if saveplot == 'y':
        # Save plots in a folder
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


    if Tr_targ == 'red':        
        LoD = red_inten_mean
    else:
        LoD = red_inten_mean
        
    effective = []
    lysis=[]
    uncoating=[]
    l_Env_t = []
    u_Env_t = []
    u_Cap_t = []
    inte_loss = pd.DataFrame(data ={"Frame":range(tracks2['Frame'].max())})
    inte_loss.set_index("Frame", inplace=True)

    for particle in range(tracks2['Particle'].max()):
        if tracks2['Particle'][tracks2['Particle']==particle].shape[0]>Threshold:

            # Data cleaning
            temp = tracks2.loc[tracks2['Particle']==particle].sort_values(by=['Frame'])
            temp_X = temp['Frame'].reset_index(drop = True)
            temp_G = temp['Mean Red Intensity'].reset_index(drop = True)
            Norm_G = temp_G/temp_G[0]*100     # HMM fitting fails when the y value is too small (not sure why)
            temp_R = temp['Mean Green Intensity'].reset_index(drop = True)
            Norm_R = temp_R/temp_R[0]*100
            diff_R = temp_R.diff()
            diff_NR = Norm_R.diff()
            R_smooth = pd.DataFrame(signal.savgol_filter(temp_R, window_length=5, polyorder=3, mode="nearest"), columns = ['Smoothed Mean red']).reset_index(drop = True)
            diff_Rs = R_smooth.diff().squeeze()
            PN = 'Particle '+str(particle)
            
            NormLoD = LoD /max(temp_G)*100
            # try:
            #####========================HMM fitting==========================#####
            # rrr= np.array(Norm_R) #Choose the dataset for HMM fitting
            # bic=list()
            # scores = list()
            # models = list()
            # maxres = list()
            # for i in range(1,4):    
            #     scores2 = list()
            #     models2 = list()
            #     try:
            #         for idx in range(20):
            #             temp_mean =np.array([[80], [20], [-10]])
            #             model = hmm.GaussianHMM(n_components=i, covariance_type='diag', random_state=idx, n_iter=10)#, means_prior=temp_mean, means_weight=1) 
            #             model.fit(rrr.reshape(-1, 1))
            #             models2.append(model)
            #             scores2.append(model.score(rrr.reshape(-1, 1)))
            #         model = models2[np.argmax(scores2)]
            #         bic.append(model.bic(rrr.reshape(-1, 1)))
            #         models.append(model)
            #         scores.append(model.score(rrr.reshape(-1, 1)))
            #         hmm_result = model.means_[model.predict(rrr.reshape(-1, 1))].reshape(len(rrr))
            #         # Residuals
            #         res1 = rrr - hmm_result
            #         maxres.append(abs(max(res1)))
            #     except:
            #         pass

                
            # # get the best model (Use BIC instead of score to prevent overfit)           
            # if len(bic) >2:
            #     if np.argmin(bic) == 1 and (bic[2]-bic[1])/bic[1]<0.05 and maxres[1]>40:
            #         evalu_b = models[2]
            #     else:
            #         evalu_b = models[np.argmin(bic)]
            # else:
            #     evalu_b = models[np.argmin(bic)]
            # # Plot the best model & count the transition time  
            # hmm_result = evalu_b.means_[evalu_b.predict(rrr.reshape(-1, 1))].reshape(len(rrr))
            # tran_num=0
            # for j in range(len(hmm_result)):
            #     if j>=1 and hmm_result[j]!= hmm_result[j-1]:
            #         tran_num +=1
            # # Residuals
            # res = rrr - hmm_result
            #####=====================End of HMM fitting======================#####
            if Fit_targ == 'red':
                hmm_result, res, tran_num, evalu_b = HMM_fit(Norm_R)
            elif Fit_targ == 'green':
                hmm_result, res, tran_num, evalu_b = HMM_fit(Norm_G)
            #####=======================Classification========================#####

            # Bad data
            if (BelowCount(temp_G, LoD)/len(temp_G) >0.99) or (max(hmm_result)/100*max(temp_R))<LoD or temp_X[0]>red.shape[0]/3:   
                ## delete the tarjectory that is too short, too many missing data, starts too late, or too weak.
            
                if saveplot == 'y':
                    os.chdir(new_folder+'/bad')
                    makeplot(temp_X, Norm_G, Norm_R, hmm_result, res)  
                    
                    message = PN+ " Bad: Reason 1"
                    disp(message)
        
                    plt.savefig(PN, bbox_inches="tight")
                    plt.close() # figures won't be displayed. This will save time.
                else:
                    pass
            # Bad data
            elif (tran_num>4 or tran_num<1) and (min(hmm_result[-5:])/100*max(temp_R))<LoD:
                
                if saveplot == 'y':
                    os.chdir(new_folder+'/bad')
                    makeplot(temp_X, Norm_G, Norm_R, hmm_result, res)  
                    
                    message = PN+ " Bad: Reason 2"
                    disp(message)
                    
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
                    
                    message = 'lysis '+ PN
                    disp(message)
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
                            t_unco = temp_X[i]
                    Cap = t_unco-Env           
                    # capsid opening time is defined as "the time between permibilization and integrity loss

                    
                    
                    if Cap>0:
                        message = 'uncoating '+ PN
                        disp(message)
                        uncoating.append(int(particle))
                        u_Env_t.append(Env)
                        u_Cap_t.append(Cap)
                        
                        corrected_temp_G = pd.concat([temp_X[Cap:], temp_G[Cap:]], axis=1)
                        corrected_temp_G.set_index("Frame", inplace=True)
                        corrected_temp_G.rename({"Mean Red Intensity": 'Mean Red Intensity_'+str(particle)}, axis=1, inplace= True)
                        inte_loss=pd.concat([inte_loss, corrected_temp_G], axis=1)
                        del corrected_temp_G
                        
                        if saveplot == 'y':
                            
                            os.chdir(new_folder+'/uncoating')                        
                            makeplot(temp_X, Norm_G, Norm_R, hmm_result, res)  
                            plt.savefig(PN, bbox_inches="tight")
                            plt.close() # figures won't be displayed. This will save time.
                        else:
                            pass
                    elif Cap <=0:
                        message = 'lysis '+ PN
                        disp(message)

                        lysis.append(int(particle))
                        l_Env_t.append(Env)
                        
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
         
    report = '================================'+'\nFile: '+ video_name+ '\n--------------------------------'+\
        '\nPercentage of Lysis-only: ' +str(round(len(lysis)/len(effective)*100, 3))+ '%'+\
            '\nPercentage of uncoating: ' +str(round(len(uncoating)/len(effective)*100, 3))+ '%'+\
                '\n--------------------------------'+ '\nNumber of lysis-only: '+ str(len(lysis))+\
                    '\nNumber of uncoating: '+ str(len(uncoating))+ '\nNumber of effective: '+ str(len(effective))+\
                        '\n================================'
    print(report)      
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
    Export_button.config(state=tk.NORMAL)
    Plot_button.config(state=tk.NORMAL)
    tk.messagebox.showinfo(title="Result", message=report)

def particle_trace():
    
    canvas.delete("all")
    try:
        particle_1 = int(PN_entry.get())
        if tracks2['Particle'][tracks2['Particle']==particle_1].shape[0]>Threshold:
            
            # Data cleaning
            temp = tracks2.loc[tracks2['Particle']==particle_1].sort_values(by=['Frame'])
            temp_X = temp['Frame'].reset_index(drop = True)
            temp_G = temp['Mean Green Intensity'].reset_index(drop = True)
            Norm_G = temp_G/max(temp_G)*100
            temp_R = temp['Mean Red Intensity'].reset_index(drop = True)
            Norm_R = temp_R/max(temp_R)*100
            diff_R = temp_R.diff()
            diff_NR = Norm_R.diff()
            R_smooth = pd.DataFrame(signal.savgol_filter(temp_R, window_length=5, polyorder=3, mode="nearest"), columns = ['Smoothed Mean red']).reset_index(drop = True)
            diff_Rs = R_smooth.diff().squeeze()
            PN = 'Particle '+str(particle_1)
                 
            if Fit_targ == 'red':
                hmm_result, res, tran_num, evalu_b = HMM_fit(Norm_R)
            elif Fit_targ == 'green':
                hmm_result, res, tran_num, evalu_b = HMM_fit(Norm_G)
            fig = makeplot(temp_X, Norm_G, Norm_R, hmm_result, res)
            fig_show(fig)
            disp(PN)
    except:
        message = "Assign particle number"
        disp(message)

    
def save_csv():
    inte_loss_1 = pd.DataFrame(data ={"Frame":range(tracks2['Frame'].max())})*sec_per_f
    inte_loss_1.set_index("Frame", inplace=True)
    inte_loss_2 = inte_loss_1.copy()
    for i in inte_loss.columns:
        inte_loss_1=pd.concat([inte_loss_1, inte_loss[i].dropna().reset_index(drop=True)], axis=1) # let starting point to be the 
        inte_loss_2[i] = inte_loss_1[i]/inte_loss_1[i][0]

    try: 
        folder
    except NameError:
        folder = askdirectory()

    writer = pd.ExcelWriter(folder+'/'+video_name+'.xlsx',engine='xlsxwriter')
    inte_loss_1.to_excel(writer,sheet_name='Integrity loss', index=True) 
    inte_loss_2.to_excel(writer,sheet_name='Normalized', index=True) 

    tracks3 = tracks2.drop(columns=['Max Green Intensity', 'Max Red Intensity', 'Total Green Intensity', 'Total Red Intensity',
                                    'Green Background Mean', 'Red Background Mean', 'Number of Pixels'])
    for a in range(tracks3['Particle'].max()):
        subset = tracks3[tracks3['Particle']==a]
        if subset.shape[0] > 30:     # Only exports particles that are present for 30 frames
          subset.to_excel(writer,sheet_name='Particle'+str(a), index=False) 
          
    writer.save()   
    writer.close()
    

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

# HMM fitting
def HMM_fit(trace):
    rrr= np.array(trace) #Choose the dataset for HMM fitting
    bic=list()
    scores = list()
    models = list()
    maxres = list()
    for i in range(1,4):    
        scores2 = list()
        models2 = list()
        try:
            for idx in range(20):
                model = hmm.GaussianHMM(n_components=i, covariance_type='diag', random_state=idx, n_iter=10)
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
    # Find the best model & count the transition time  
    hmm_result = evalu_b.means_[evalu_b.predict(rrr.reshape(-1, 1))].reshape(len(rrr))
    tran_num=0
    for j in range(len(hmm_result)):
        if j>=1 and hmm_result[j]!= hmm_result[j-1]:
            tran_num +=1
    # Residuals
    res = rrr - hmm_result
    return hmm_result, res, tran_num, evalu_b

def makeplot(x, y_G, y_R, y_HMM, y_Res):
    fig, (ax1, ax2) = plt.subplots(2, height_ratios=[2, 1], dpi = 100)
    fig.tight_layout() ## Adjust the space between plots
    ax1.scatter(x*sec_per_f, y_R, color='r', label='Mean Red Intensity')
    ax1.scatter(x*sec_per_f, y_G, color='g', label='Mean Green Intensity')
    # ax1.plot(x*sec_per_f, R_smooth, color='k', label='Smoothed Max red Intensity',linestyle ='--')
    ax1.plot(x*sec_per_f, y_HMM,  color='b', label= 'HMM fit')
    ax1.set_xlim(0,red.shape[0]*sec_per_f)
    # ax1.set_ylim(-5,110)
    ax1.legend()
    ax1.set_xlabel('Time (sec)')
    ax1.set_ylabel('Normalized Intensity (%)')
    # ax1.set_title(PN)
    ax2.plot(range(red.shape[0]*sec_per_f), [i* 0 for i in range(red.shape[0]*sec_per_f)], color='grey', linestyle='--')
    ax2.plot(x*sec_per_f, y_Res, label='Residuals of HMM')
    # ax2.plot(temp_X, diff_R, label='Derivative of Max red Intensity')
    # ax2.plot(temp_X, diff_Rs, label='Derivative of Smoothed Max red Intensity', color='k', linestyle='--')
    ax2.legend()
    ax2.set_xlabel('Time (sec)')
    ax2.set_xlim(0,red.shape[0]*sec_per_f)
    ax2.set_ylim(-20,20)
    fig.set_size_inches(6., 6.)
    fig.tight_layout()
    plot = plt.gcf()
    return plot

def disp(message):
    print(message)
    message_entry.delete(0, tk.END)
    message_entry.insert(0, message)


root = tk.Tk()
root.title("Tracking Particles GUI")

# Create a list to hold the entry widgets
entries = []

labels = [
    "File:", "Red channel:", "FRET channel:", "Green channel:", "Adding saponin at frame:", 
    "Second per frame (sec):", "Diameter (nm):", "Minimum intensity:", "Separation:", "Percentile (%):", 
    "Max distance per frames (\u03bcm):", "Memory (frame):", "Minimum length (frame):", "Background radius (pixel):", 
    "Frame_diff (frame):", "Location_diff (pixel):", "Limit of detection factor:", "Saveplot:", "New filename:", 
]
try:
    default_values = [
        filename, RedCh, FreCh, GreCh, AddSap, sec_per_f, Diameter, Min_mass, 
        Separation, Percentile, Max_distance, Memory, Threshold, Background_radius, 
            Frame_diff, Location_diff, Lo_clus_r, saveplot, video_name
    ]

except:
    default_values = [
        "", "3", "2", "1", "10", "5", "7", "400", "3", "97", "5", "5", "20", "1", "4", "4", "10", "y", ""
    ]


# Create the first entry for the file input with a button to open the file dialog
file_label = tk.Label(root, text=labels[0])
file_label.grid(row=0, column=0, padx=10, pady=5)
file_entry = tk.Entry(root, width=50)
file_entry.insert(0, default_values[0])
file_entry.grid(row=0, column=1, columnspan=2, padx=10, pady=5)
file_button = tk.Button(root, text="Browse", command=open_file)
file_button.grid(row=0, column=3, padx=10, pady=5)
entries.append(file_entry)

# Create entries for the remaining variables
for i in range(1, 17):
    label = tk.Label(root, text=labels[i])
    label.grid(row=i, column=0, padx=5, pady=5)
    entry = tk.Entry(root, width=10)
    entry.insert(0, default_values[i])
    entry.grid(row=i, column=1, padx=5, pady=5)
    entries.append(entry)

# Create radio buttons for "Tracking"
track_var = tk.StringVar()
try:
    track_var.set(Tr_targ)  # Set existed value
except:
    track_var.set("red")  # Set default value    
track_red = tk.Radiobutton(root, text="Track", variable=track_var, value="red")
track_red.grid(row=1, column=2, padx=10, pady=5)
track_fret = tk.Radiobutton(root, text="Track", variable=track_var, value="fret")
track_fret.grid(row=2, column=2, padx=10, pady=5)
track_green = tk.Radiobutton(root, text="Track", variable=track_var, value="green")
track_green.grid(row=3, column=2, padx=10, pady=5)

# Create radio buttons for "Fit"
Fit_var = tk.StringVar()
try:
    Fit_var.set(Fit_targ)  # Set existed value
except:
    Fit_var.set("red")  # Set default value    
Fit_red = tk.Radiobutton(root, text="HMM", variable=Fit_var, value="red")
Fit_red.grid(row=1, column=3, padx=10, pady=5)
Fit_fret = tk.Radiobutton(root, text="HMM", variable=Fit_var, value="fret")
Fit_fret.grid(row=2, column=3, padx=10, pady=5)
Fit_green = tk.Radiobutton(root, text="HMM", variable=Fit_var, value="green")
Fit_green.grid(row=3, column=3, padx=10, pady=5)
    
# Create radio buttons for "Save plot"
saveplot_label = tk.Label(root, text="Save plot:")
saveplot_label.grid(row=17, column=0, padx=10, pady=5)
saveplot_var = tk.StringVar()
try:
    saveplot_var.set(saveplot)  # Set existed value
except:
    saveplot_var.set("y")  # Set default value    
saveplot_yes = tk.Radiobutton(root, text="Yes", variable=saveplot_var, value="y")
saveplot_yes.grid(row=17, column=1, padx=10, pady=5)
saveplot_no = tk.Radiobutton(root, text="No", variable=saveplot_var, value="n")
saveplot_no.grid(row=17, column=2, padx=10, pady=5)


# Create the extra variable for new filename
label_extra = tk.Label(root, text=labels[18])
label_extra.grid(row=19, column=0, padx=10, pady=5)
entry_extra = tk.Entry(root, width=40)
entry_extra.insert(0, default_values[18])
entry_extra.grid(row=19, column=1, columnspan=2, padx=10, pady=5)
entries.append(entry_extra)

# Create a message box
message_entry = tk.Entry(root, width=90)
message_entry.grid(row=20, column=0, columnspan=4, padx=5, pady=5)

# Create a submit button
submit_button = tk.Button(root, text="Update Parameters", command=submit)
submit_button.grid(row=21, columnspan=3, pady=10)

# Create a canvas on the right side
canvas = tk.Canvas(root, width=650, height=650, bg="white")
canvas.grid(row=0, column=4, rowspan=20, columnspan=14, padx=1, pady=1)

# Create a Min/Max entries for figure
Min_label = tk.Label(root, text="Min: ")
Min_label.grid(row=20, column=4, pady=5)
Min_entry = tk.Entry(root, width=10)
Min_entry.grid(row=20, column=5, padx=2, pady=5)
Max_label = tk.Label(root, text="Max: ")
Max_label.grid(row=20, column=6, pady=5)
Max_entry = tk.Entry(root, width=10)
Max_entry.grid(row=20, column=7, padx=2, pady=5)

# Create an entry of frame number to present
frame_label = tk.Label(root, text="Frame: ")
frame_label.grid(row=20, column=8, pady=5)
frame_entry = tk.Entry(root, width=5)
frame_entry.insert(0, 0)
frame_entry.grid(row=20, column=9, padx=2, pady=5)

# Create a button to show raw image
raw_button = tk.Button(root, text="Raw image", command=lambda: [submit(), loadimage(), RawImg()])
raw_button.grid(row=21, column=4, pady=10)

# Create a button to show particles
particle_button = tk.Button(root, text="Find Particles", command=lambda: [submit(), loadimage(), ParImg()])
particle_button.grid(row=21, column=5, pady=10)

# Create a button to show tracking result
particle_button = tk.Button(root, text="Track", command=lambda: [submit(), loadimage(), track()])
particle_button.grid(row=21, column=6, pady=10)

# Create a button to perform analysis (Background subtraction, trace stitching, classification)
process_button = tk.Button(root, text="Process", command=process)
process_button.grid(row=21, column=7, pady=10)
try:
    tracks
except:
    process_button.config(state=tk.DISABLED) # Initially disable the process_button if tracks does not exist

# Create an entry of frame number to present
PN_label = tk.Label(root, text="Particle: ")
PN_label.grid(row=20, column=10, pady=5)
PN_entry = tk.Entry(root, width=5)
PN_entry.insert(0, 0)
PN_entry.grid(row=20, column=11, padx=2, pady=5)

# Create a button to perform analysis (Background subtraction, trace stitching, classification)
Plot_button = tk.Button(root, text="Plot", command=particle_trace)
Plot_button.grid(row=21, column=10, pady=10)


# Create a button to export the results
Export_button = tk.Button(root, text="Export", command=save_csv)
Export_button.grid(row=21, column=11, pady=10)

try:
    tracks2
except:
    Plot_button.config(state=tk.DISABLED) # Initially disable the Plot_button if tracks2 does not exist
    Export_button.config(state=tk.DISABLED) # Initially disable the Export_button if tracks2 does not exist

# Creat a close button
Finish_button = tk.Button(root, text="Finish", command=finish)
Finish_button.grid(row=21, column=0, pady=10)

root.mainloop()

# # The program continues here after the window is closed
# print("Program finished.")
# print("Saved inputs:", inputs)