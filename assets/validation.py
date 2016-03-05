import math
import numpy as np
from assets.gen_motion import *
from assets.keras.keras.utils import np_utils

def validate_model(model_,reps,true_vals,run=[1,1,1,1,1]):
    adata = []
    
    true_coh, true_spd, true_dots,true_con,true_noise = true_vals
    # Check 1: Performance -> 0 as coherence goes to 0
    
    # we will use the same angles, a dif check will interpolate
    angles = [0,math.pi/4,math.pi/2,math.pi*3/4,math.pi,math.pi*5/4,math.pi*3/2,math.pi*7/4]

    if run[0]==1:
        coh_data = []
        for coh in np.linspace(0,1,9):
            scores = []
            for r in range(reps):
                # build a quick dataset
                X_train_low,Y_train_low,_,_,_,_ = gen_dataset((16,64,64),10,None,['translate'],[true_spd],angles,[coh],[true_dots],[0],dot_radius=2,contrast=true_con,Snoise=true_noise,retina=0)
                Y_train_low[:,2] = Y_train_low[:,2]/math.pi*4
                nb_classes = len(angles)
                Y_train_low_ = np_utils.to_categorical(Y_train_low[:,2]-1, nb_classes)
                score_low_train = model_.evaluate(X_train_low, Y_train_low_, batch_size=20, show_accuracy=True, verbose=0)
                scores.append(score_low_train[1])
            coh_data.append((coh,scores))

        cohs = []
        mus = []
        for c in coh_data:
            coh,dat = c
            cohs.append(coh)
            mus.append(np.mean(dat))

        plt.figure()
        plt.plot(cohs,mus)
        plt.axis([0,1,0,1])
        plt.axhline(0.125,linestyle='dashed',color='red')
        plt.xlabel('Coherence (%)')
        plt.ylabel('Classification Performance (% Correct)')
        plt.show()

        adata.append((cohs,mus))

    # Check 2: Performance -> as speed goes to 0
    
    if run[1]:
        spd_data = []
        for spd in np.linspace(0,10,15):
            scores = []
            for r in range(reps):
                # build a quick dataset
                X_train_low,Y_train_low,_,_,_,_ = gen_dataset((16,64,64),10,None,['translate'],[spd],angles,[true_coh],[true_dots],[0],dot_radius=2,contrast=true_con,Snoise=true_noise,retina=0)
                Y_train_low[:,2] = Y_train_low[:,2]/math.pi*4
                nb_classes = len(angles)
                Y_train_low_ = np_utils.to_categorical(Y_train_low[:,2]-1, nb_classes)
                score_low_train = model_.evaluate(X_train_low, Y_train_low_, batch_size=20, show_accuracy=True, verbose=0)
                scores.append(score_low_train[1])
            spd_data.append((spd,scores))

        spds = []
        mus = []
        for s in spd_data:
            spd,dat = s
            spds.append(spd)
            mus.append(np.mean(dat))

        plt.figure()
        plt.plot(spds,mus)
        plt.axis([0,5,0,1])
        plt.axhline(0.125,linestyle='dashed',color='red')
        plt.xlabel('Speed (px/frame)')
        plt.ylabel('Classification Performance (% Correct)')
        plt.show()

        adata.append((spds,mus))
    # Check 3: Performance -> 0 as number of dots goes to 0
    
    if run[2]:
        dot_data = []
        for dot in np.linspace(0,80,15):
            scores = []
            for r in range(reps):
                # build a quick dataset
                X_train_low,Y_train_low,_,_,_,_ = gen_dataset((16,64,64),10,None,['translate'],[true_spd],angles,[true_coh],[dot],[0],dot_radius=2,contrast=true_con,Snoise=true_noise,retina=0)
                Y_train_low[:,2] = Y_train_low[:,2]/math.pi*4
                nb_classes = len(angles)
                Y_train_low_ = np_utils.to_categorical(Y_train_low[:,2]-1, nb_classes)
                score_low_train = model_.evaluate(X_train_low, Y_train_low_, batch_size=20, show_accuracy=True, verbose=0)
                scores.append(score_low_train[1])
            dot_data.append((dot,scores))

        dots = []
        mus = []
        for d in dot_data:
            dot,dat = d
            dots.append(dot)
            mus.append(np.mean(dat))

        plt.figure()
        plt.plot(dots,mus)
        plt.axis([0,100,0,1])
        plt.axhline(0.125,linestyle='dashed',color='red')
        plt.xlabel('Number of Dots (#)')
        plt.ylabel('Classification Performance (% Correct)')
        plt.show()

        adata.append((dots,mus))

    #################################### CONTRAST ######################################
    
    if run[3]:
        con_data = []
        for con in np.linspace(0,1,10):
            scores = []
            for r in range(reps):
                # build a quick dataset
                X_train_low,Y_train_low,_,_,_,_ = gen_dataset((16,64,64),10,None,['translate'],[true_spd],angles,[true_coh],[true_dots],[0],dot_radius=2,contrast=con,Snoise=true_noise,retina=0)
                Y_train_low[:,2] = Y_train_low[:,2]/math.pi*4
                nb_classes = len(angles)
                Y_train_low_ = np_utils.to_categorical(Y_train_low[:,2]-1, nb_classes)
                score_low_train = model_.evaluate(X_train_low, Y_train_low_, batch_size=20, show_accuracy=True, verbose=0)
                scores.append(score_low_train[1])
            con_data.append((con,scores))

        cons = []
        mus = []
        for d in con_data:
            con,dat = d
            cons.append(con)
            mus.append(np.mean(dat))

        plt.figure()
        plt.plot(cons,mus)
        plt.axis([0,1,0,1])
        plt.axhline(0.125,linestyle='dashed',color='red')
        plt.xlabel('Contrast (%)')
        plt.ylabel('Classification Performance (% Correct)')
        plt.show()

        adata.append((cons,mus))
    
        #################################### NOISE ######################################
    
    if run[4]:
        noi_data = []
        for noi in np.linspace(0,150,15):
            scores = []
            for r in range(reps):
                # build a quick dataset
                X_train_low,Y_train_low,_,_,_,_ = gen_dataset((16,64,64),10,None,['translate'],[true_spd],angles,[true_coh],[true_dots],[0],dot_radius=2,contrast=true_con,Snoise=noi,retina=0)
                Y_train_low[:,2] = Y_train_low[:,2]/math.pi*4
                nb_classes = len(angles)
                Y_train_low_ = np_utils.to_categorical(Y_train_low[:,2]-1, nb_classes)
                score_low_train = model_.evaluate(X_train_low, Y_train_low_, batch_size=20, show_accuracy=True, verbose=0)
                scores.append(score_low_train[1])
            noi_data.append((noi,scores))

        nois = []
        mus = []
        for d in noi_data:
            noi,dat = d
            nois.append(noi)
            mus.append(np.mean(dat))

        plt.figure()
        plt.plot(nois,mus)
        plt.axis([0,150,0,1])
        plt.axhline(0.125,linestyle='dashed',color='red')
        plt.xlabel('Additive Gaussian Noise (Std)')
        plt.ylabel('Classification Performance (% Correct)')
        plt.show()

        adata.append((nois,mus))