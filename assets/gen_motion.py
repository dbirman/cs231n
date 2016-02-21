import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from assets.motion import *
from assets.opticflowmotion import *
from assets.rotation import *

#author: steeve laquitaine
#purpose: wrapper to run different types of motions

def gen_motion(type,x,y,t,n,velocity,theta,coherence,dot_radius,frames_per_second):
    
    #translation
    if type in ['Translate']:
        mymot = Translate(x,y,t,n,velocity,theta,coherence)                

    #optic flow      
    elif type in ['opticflowmotion']:
        mymot = opticflowmotion(x,y,t,n,velocity)

    #rotation dots        
    elif type in ['rotation']:
        mymot = Rotational(x,y,t,n,velocity,dot_radius,frames_per_second)                         
    return mymot        