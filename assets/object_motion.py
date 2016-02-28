import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

#################
# OBJECT MOTION #
#################

# Object motion takes in a matrix that defines an object "mask" (0->255 values) 
# of size (mx,my) it then builds a video of (t,x,y) size consisting of background
# motion with coherence (backg_coh). The background motion moves in direction
# (backg_theta) with speed (backg_vel). # of background dots is (backg_n).

# The object itself starts at a random position in the field and moves with angle,
# velocity, and speeds, obj_*.

    
def mask_square(x,y):
    return (np.ones((x,y))*255).astype('uint8')
    
def mask_circle(x,y,rad):
    mask = np.zeros((x,y))
    if (not x%2==1) and (not y%2==1):
        print "Warning: Circles work better with odd numbers"
    mx = np.rint(x/2)
    my = np.rint(y/2)
    for xi in np.arange(x):
        for yi in np.arange(y):
            dist = np.sqrt((xi-mx)**2+(yi-my)**2)
            mask[xi,yi]=np.rint(np.exp(-(dist/rad)**4)*255)
    return mask

def mask_donut(x,y,rad):
    mask = np.zeros((x,y))
    if (not x%2==1) and (not y%2==1):
        print "Warning: Circles work better with odd numbers"
    mx = np.rint(x/2)
    my = np.rint(y/2)
    for xi in np.arange(x):
        for yi in np.arange(y):
            dist = np.sqrt((xi-mx)**2+(yi-my)**2)
            if (dist <= mx and dist > mx/2):
                mask[xi,yi]=np.rint(np.exp(-(abs(dist-rad)/rad)**4)*255)
    return mask

class Object_Motion():
                

    # Initialize
    def __init__(self,obj_mask,obj_vel,obj_theta):
        self.data = None
        # Object
        self.ox,self.oy = obj_mask.shape
        self.obj_mask = obj_mask.astype('uint8')
        self.obj_vel = obj_vel
        self.obj_theta = obj_theta
        
    def gen(self,data):
        self.t,self.x,self.y = data.shape
        if (not self.ox < self.x) or (not self.oy < self.y):
            print "Failure: mask size must be smaller than video size!!!"
            return
        # Add the object in front of the Background
        # Initialize object pos
        obj_x = np.random.randint(self.x)
        obj_y = np.random.randint(self.y)
        for t in np.arange(self.t):
            # merge the mask using np.max into the xs array
            for x in np.arange(self.ox):
                for y in np.arange(self.oy):
                    t_x = (np.rint(obj_x) + x) % self.x
                    t_y = (np.rint(obj_y) + y) % self.y
                    data[t,t_x,t_y] = np.max([data[t,t_x,t_y],self.obj_mask[x,y]])
            obj_x = obj_x + self.obj_vel * np.cos(self.obj_theta)
            obj_y = obj_y + self.obj_vel * np.sin(self.obj_theta)
            if obj_x > self.x:
                obj_x -= self.x
            if obj_x < 0:
                obj_x += self.x
            if obj_y > self.y:
                obj_y -= self.y
            if obj_y < 0:
                obj_y += self.y
        self.data = data
        return data

    # Display figure showing this motion
    def plot(self):
        if self.data==None:
            return
        fig = plt.figure()
        ax = plt.gca()
        im = plt.imshow(self.data[0,:,:], cmap='Greys_r', vmin=0, vmax=255)	
        def up(t):
            im.set_array(self.data[t,:,:])
            return im,
        ani = anim.FuncAnimation(fig, up, np.arange(self.t), interval=50, blit=True, repeat_delay=1000)
        plt.show()
        return ani

