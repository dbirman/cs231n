import numpy as np
from assets.motion import *
from assets.expandContract import *
from assets.rotation import *
from assets.opticflow import *

#author: steeve laquitaine, dan birman
#purpose: wrapper to run different types of motions

def gen_dataset(size, N, types, velocity, theta, coherence, dots, direction, test_prop=0.1):
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
# size = (time, cols, rows, # dots)
# N = num repetitions of factorial parameters
# dot_radius (optional) 
# frames_per_second (optional)
#
# types: ['translate','rotation','expandContract','opticflow'] 
# velocity: [1+] pixels per tick
# theta: [0-2*Math.pi]
# coherence: [0-1] % moving dots
# dots: # of dots
# direction: if optic flow [1 or -1] else = 0
    ti,x,y = size
    total = len(types)*len(velocity)*len(theta)*len(coherence)*len(dots)*len(direction)*N
    all_data = np.zeros((total,1,ti,x,y))
    # Y will track all the dep vars in columns 1:5
    all_y = np.zeros((total,6))
    i = 0   
    for t in types:
        for v in velocity:
            for a in theta:
                for c in coherence:
                    for d in dots:
                        #check direction of optic flow
                        if t in ['opticflow']:
                            for di in direction: 
                                mot,ct = gen_motion(t,x,y,ti,d,v,a,c,di)
                                for n in np.arange(N):
                                    mot.gen()
                                    all_data[i,0,:,:,:] = mot.data 
                                    all_y[i,:] = [ct,v,a,c,d,di]
                                    i+=1                                 
                        else:    
                            mot,ct = gen_motion(t,x,y,ti,d,v,a,c,0)
                            for n in np.arange(N):
                                mot.gen()
                                all_data[i,0,:,:,:] = mot.data 
                                all_y[i,:] = [ct,v,a,c,d,0]
                                i+=1
                            
    # split out 10% of the data
    tr_fold = np.round(total * (1-test_prop))
    
    X_train = all_data[:tr_fold,:,:,:,:]
    Y_train = all_y[:tr_fold,:]
    X_test = all_data[tr_fold:,:,:,:,:]
    Y_test = all_y[tr_fold,:]
        
    return (X_train, Y_train, X_test, Y_test)           

def gen_motion(type,x,y,t,n,velocity,theta,coherence,direction):
    dot_radius = 1
    frames_per_second = 1
    
    ctype = 0
    #translation
    if type in ['translate']:
        mymot = Translate(x,y,t,n,velocity,theta,coherence)
        ctype = 1

    #expand contract
    elif type in ['expandContract']:
        mymot = expandContract(x,y,t,n,velocity)
        ctype = 2

    #rotation dots        
    elif type in ['rotation']:
        mymot = Rotational(x,y,t,n,velocity,dot_radius,frames_per_second) 
        ctype = 3
        
    #optic flow
    elif type in ['opticflow']:
        mymot = opticflow(x,y,t,n,velocity,coherence,direction) 
        ctype = 4        
        
    return (mymot,ctype)        