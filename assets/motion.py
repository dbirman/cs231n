import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import math

#############
# TRANSLATE #
#############

#author: Dan Birman, modifed by Dylan cable and Steeve Laquitaine

# Translational motion in 360 degs
class Translate():

    # Initialize
    def __init__(self,x,y,t,n,r,velocity,theta,coherence=1.0,contrast=1,Snoise =0.0):
        # Data size
        self.x = x
        self.y = y
        self.t = t
        self.data = np.zeros((t,x,y))
        # Dot num
        self.n = n
        self.dot_radius = r
        # Speed 
        self.velocity = velocity
        # Angle in Radians np.pi (0 is down)
        self.theta = theta
        # Coherence
        self.coherence = coherence
        # Contrast
        self.contrast = float(contrast)
        # Stimulus gaussian noise std
        self.Snoise = Snoise

    # Generate a new motion stimulus
    def gen(self):
        # Initialize
        self.data = np.ones((self.t,self.x,self.y),dtype='uint8')
        # gray background
        bgColor = np.rint(255/2)   
        # set background to bgColor
        self.data = self.data * bgColor
        # get max contrast value
        maxC = (255-bgColor)*self.contrast+bgColor
        #dot radius
        dr = self.dot_radius
        c = np.random.rand(self.n,1)<self.coherence
        notc = (c==False)[:,0]
        xs = np.random.randint(0,self.x,(self.n,1))
        ys = np.random.randint(0,self.y,(self.n,1))
        # generate the dot matrix
        dot_mat = np.zeros((6*dr+1,6*dr+1)) # goes out 3*dr in both directions from center point
        cx = 3*dr
        cy = 3*dr
        for x in range(6*dr+1):
            for y in range(6*dr+1):
                dist = math.sqrt(math.pow(x-cx,2) + math.pow(y-cy,2))
                dot_mat[x,y] = maxC*math.exp(-math.pow(dist/dr,2))
                
        # Repeatedly place and then move
        for t in np.arange(self.t):
            for i in np.arange(len(xs)):
                #dot of radius dot_radius at (xs[i],ys[i])
                # get prop of dot_mat to use
                x_ = 0
                x__ = 6*dr+1
                y_ = 0
                y__ = 6*dr+1
                if xs[i] < 3*dr:
                    x_ = 3*dr-xs[i]
                if xs[i] > (self.x-1)-3*dr:
                    x__ = cx+self.x-xs[i]
                if ys[i] < 3*dr:
                    y_ = 3*dr-ys[i]
                if ys[i] > (self.y-1)-3*dr:
                    y__ = cx+self.y-ys[i]
                self.data[t,max(0,xs[i]-3*dr):min(self.x,xs[i]+3*dr+1),max(0,ys[i]-3*dr):min(self.y,ys[i]+3*dr+1)] += dot_mat[x_:x__,y_:y__]
                
            # Move all coherent dots
            xs[c] = (xs[c] + self.velocity * np.cos(self.theta))
            ys[c] = (ys[c] + self.velocity * np.sin(self.theta))
            # Move all incoherent dots
            inc_thetas = np.random.rand(np.sum(notc),1)*np.pi*2
            xinc = self.velocity * np.cos(inc_thetas)
            yinc = self.velocity * np.sin(inc_thetas)
            xs[notc] = (np.add(xs[notc], xinc))
            ys[notc] = (np.add(ys[notc], yinc))
            # Fix dots that go offscreen
            xs[xs>=self.x] = xs[xs>=self.x]-self.x
            xs[xs<0] = xs[xs<0]+self.x
            ys[ys>=self.y] = ys[ys>=self.y]-self.y
            ys[ys<0] = ys[ys<0]+self.x
            # Revert to int so positions can be parsed correctly
            xs = xs.astype(int)
            ys = ys.astype(int)
        
        # reduce to maxC
        self.data = np.clip(maxC,0,self.data)
        
        # Add Gaussian noise to stimulus pixels
        if self.Snoise>0:
            self.data += np.random.normal(0,self.Snoise,(self.t,self.x,self.y))
        self.data = np.rint(self.data)
        return self.data.astype('uint8')

    # File saving (for later loading into convnets)
    def fout(self):
        print "todo: implement save to file"

    # Display figure showing this motion
    def plot(self):
        fig = plt.figure()
        ax = plt.gca()
        im = plt.imshow(self.data[0,:,:], cmap='Greys_r', vmin=0, vmax=255,interpolation='none')
        def up(t):
            im.set_array(self.data[t,:,:])
            return im,
        ani = anim.FuncAnimation(fig, up, np.arange(self.t), interval=50, blit=True, repeat_delay=1000)
        plt.show()
        return ani

