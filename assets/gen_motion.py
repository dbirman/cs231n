import numpy as np
from assets.motion import *
from assets.expandContract import *
from assets.rotation import *
from assets.opticflow import *
from assets.object_motion import *
import random

#author: steeve laquitaine, dan birman
#purpose: wrapper to run different types of motions

def gen_dataset(size, N, obj, types, velocity, theta, coherence, dots, direction, vt_prop=0.1):
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
# Object Parameters
# obj is a tuple of: (o_t, o_th, o_s, o_v) or None
# obj_type = 'square', 'circle', 'donut'
# obj_theta: angle of object motion 0-2*math.pi
# obj_size: (x,y,r)
# obj_vel: speed in pixels/frame
#
# Background Parameters
# types: ['translate','rotation','expandContract','opticflow'] 
# velocity: [1+] pixels per tick
# theta: [0-2*Math.pi]
# coherence: [0-1] % moving dots
# dots: # of dots
# direction: if optic flow [1 or -1] else = 0
    # Generate training data
    X_train,Y_train = gen_dataset_(size, N, obj, types, velocity, theta, coherence, dots, direction)
    sub = np.rint(N*vt_prop)
    X_val,Y_val = gen_dataset_(size, sub, obj, types, velocity, theta, coherence, dots, direction)
    X_test,Y_test = gen_dataset_(size, sub, obj, types, velocity, theta, coherence, dots, direction)
        
    return (X_train, Y_train,X_val,Y_val, X_test, Y_test)  

def gen_dataset_(size, N, obj, types, velocity, theta, coherence, dots, direction):
    if not obj==None:
        obj_type, obj_size, obj_theta, obj_vel = obj
    ti,x,y = size
    total = len(types)*len(velocity)*len(theta)*len(coherence)*len(dots)*len(direction)*N
    all_data = np.zeros((total,1,ti,x,y))
    # Y will track all the dep vars in columns 1:5
    all_y = np.zeros((total,6))
    i = 0   
    if not obj==None:
        obj_mask = None
        otype = 0
        ox,oy,o_r = obj_size
        # Build object, since this will always be consistent
        if obj_type=='square':
            obj_mask = mask_square(ox,oy)
            otype = 1
        elif obj_type=='circle':
            obj_mask = mask_circle(ox,oy,o_r)
            otype = 2
        elif obj_type=='donut':
            obj_mask = mask_donut(ox,oy,o_r)
            otype = 3
        obj = Object_Motion(obj_mask,obj_vel,obj_theta)                          
    for t in types:
        for v in velocity:
            for a in theta:
                for c in coherence:
                    for d in dots:
                        #check direction of optic flow
                        if t in ['opticflow','rotation','expandContract']:
                            for di in direction: 
                                mot,ct = gen_motion(t,x,y,ti,d,v,a,c,di)
                                for n in np.arange(N):
                                    mot.gen()
                                    all_data[i,0,:,:,:] = mot.data 
                                    all_y[i,:] = [ct,v,a,c,d,di]
                                    if not obj==None:
                                        all_data[i,0,:,:,:] = obj.gen(all_data[i,0,:,:,:])
                                    i+=1                                 
                        else:    
                            mot,ct = gen_motion(t,x,y,ti,d,v,a,c,0)
                            for n in np.arange(N):
                                mot.gen()
                                all_data[i,0,:,:,:] = mot.data 
                                all_y[i,:] = [ct,v,a,c,d,0]
                                if not obj==None:
                                    all_data[i,0,:,:,:] = obj.gen(all_data[i,0,:,:,:])
                                i+=1
         
    if not obj==None:
        ot = otype * np.ones((total,1))
        all_y = np.concatenate((all_y,ot),axis=1)
    return all_data, all_y
    

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
        mymot = expandContract(x,y,t,n,velocity,direction)
        ctype = 2

    #rotation dots        
    elif type in ['rotation']:
        mymot = Rotational(x,y,t,n,velocity,dot_radius,frames_per_second,direction) 
        ctype = 3
        
    #optic flow
    elif type in ['opticflow']:
        mymot = opticflow(x,y,t,n,velocity,coherence,direction) 
        ctype = 4        
        
    return (mymot,ctype)        