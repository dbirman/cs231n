import numpy as np
from assets.motion import *
from assets.opticflowmotion import *
from assets.rotation import *
import random

#author: steeve laquitaine, dan birman
#purpose: wrapper to run different types of motions

def gen_dataset(size, N, types, velocity, theta, coherence, dots, test_prop=0.1):
########################
## Generate a Dataset ##
########################
#
# Call this function as:
# gen_dataset()
#
# All parameters can be passed as lists, in which 
# case N examples of that parameter, factorial with 
# every other parameter are generated.
#
# Parameters:
#
# size = (time, cols, rows)
# N = num repetitions of factorial parameters
# dot_radius (optional) 
# frames_per_second (optional)
#
# types: ['translate','rotation','opticflow'] 
# velocity: [1+] pixels per tick
# theta: [0-2*Math.pi]
# coherence: [0-1] % moving dots
# dots: # of dots
    ti,x,y = size
    total = len(types)*len(velocity)*len(theta)*len(coherence)*len(dots)*N
    all_data = np.zeros((total,1,ti,x,y))
    # Y will track all the dep vars in columns 1:5
    all_y = np.zeros((total,5))
    i = 0   
    for t in types:
        for v in velocity:
            for a in theta:
                for c in coherence:
                    for d in dots:
                        mot,ct = gen_motion(t,x,y,ti,d,v,a,c)
                        for n in np.arange(N):
                            mot.gen()
                            all_data[i,0,:,:,:] = mot.data 
                            all_y[i,:] = [ct,v,a,c,d]
                            i+=1
    # split out 10% of the data
    tr_fold = int(np.round(total * (1-test_prop)))
    mylist = range(total)
    train_ind = [ mylist[i] for i in sorted(random.sample(xrange(len(mylist)), tr_fold)) ]
    test_ind = list(set(mylist) - set(train_ind))
    X_train = all_data[train_ind,:,:,:,:]
    Y_train = all_y[train_ind,:]
    X_test = all_data[test_ind,:,:,:,:]
    Y_test = all_y[test_ind,:]
        
    return (X_train, Y_train, X_test, Y_test)           

def gen_motion(type,x,y,t,n,velocity,theta,coherence):
    dot_radius = 1
    frames_per_second = 1
    
    ctype = 0
    #translation
    if type in ['translate']:
        mymot = Translate(x,y,t,n,velocity,theta,coherence)
        ctype = 1

    #optic flow      
    elif type in ['opticflow']:
        mymot = opticflowmotion(x,y,t,n,velocity)
        ctype = 2

    #rotation dots        
    elif type in ['rotation']:
        mymot = Rotational(x,y,t,n,velocity,dot_radius,frames_per_second) 
        ctype = 3
        
    return (mymot,ctype)        