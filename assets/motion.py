import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import math

#############
# TRANSLATE #
#############

# Translational motion in 360 degs
class Translate():

    # Initialize
    def __init__(self,x,y,t,n,r,velocity,theta,coherence=1.0,contrast=1,Snoise=0):
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
        self.contrast = contrast
        # Stimulus gaussian noise std
        self.Snoise = Snoise

    # Generate a new motion stimulus
    def gen(self):
        # Initialize
        # self.data = np.zeros((self.t,self.x,self.y),dtype='uint8')
        # Add Gaussian noise to stimulus pixels
        bgColor = np.rint(255/2)
        self.data = np.random.normal(bgColor,self.Snoise,(self.t,self.x,self.y)).astype('uint8')
        # self.data = self.data.astype(uint8)
        dot_radius = self.dot_radius
        c = np.random.rand(self.n,1)<self.coherence
        notc = (c==False)[:,0]
        xs = np.random.randint(0,self.x,(self.n,1))
        ys = np.random.randint(0,self.y,(self.n,1))
        # Repeatedly place and then move
        for t in np.arange(self.t):
            for i in np.arange(len(xs)):
                #dot of radius dot_radius at (xs[i],ys[i])
                for x in range(int(round(xs[i]-1-2*dot_radius)),int(round(xs[i]+2*dot_radius+1))):
                    for y in range(int(round(ys[i]-1-2*dot_radius)),int(round(ys[i]+2*dot_radius+1))):
                        #i = maxY - y
                        #j = x + maxX
                        #i = min(max(i,0),imageHeight-1) #constrain within range
                        #j = min(max(j,0),imageWidth-1) #constrain within range
                        #y_corr = maxY - i
                        #x_corr = j - maxX
                        if(x < self.x and x >= 0 and y < self.y and y >= 0):
                            x_corr = x
                            y_corr = y
                            dist = math.sqrt(math.pow(x_corr-xs[i],2) + math.pow(y_corr-ys[i],2))
                            self.data[t,x,y] = min(255, self.data[t,x,y] + 255*math.exp(-math.pow(dist/dot_radius,2))); #min(1,self.data[t][i][j]+math.exp(-math.pow(dist/dot_radius,2)));
                            #if(dist < 1e-8):
                                #print self.data[t,x,y]
                            #self.data[t,x,y] = min(self.data[t,x,y],50)
                #self.data[t,xs[i],ys[i]] = 255
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
        return self.data

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

