
#opticflowreal.py
#
#
# author: steeve laquitaine
#   date: 2016/01/25
#purpose: simulate expanding optic flow forward motion 
#
#  usage: 
#
# in ipython notebook
#
# 		from assets.opticflowmotion import *
# 		%load_ext autoreload  
#		%autoreload 2
# 		%matplotlib nbagg
#
#		c = opticflowreal(64,64,32,400,1,0.7)
#		c.gen()
#		c.plot()
#
#reference: inspired from Dan Birman motion.py module

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim


##############
# OPTIC FLOW #
##############

# Optic flow forward (expanding) backward (contracting)
class opticflowreal():

	# Initialize
	def __init__(self,x,y,t,n,velocity,coherence=1,direction=1):
		# Data size
		self.x = x
		self.y = y
		self.t = t
		self.data = np.zeros((x,y,t))

		# Dot num
		self.n = n

		# Speed 
		self.velocity = velocity
		# coherence
		self.coherence = coherence
		# direction        
		self.direction = direction                 
        
	def gen(self):
    
		#dots,coherence,myscreen)
		#get the coherent and incoherent dots
		#if (dots.coherency ~= coherence)
		# Initialize all points as black
		self.data = np.zeros((self.x,self.y,self.t),dtype='uint8')
		c = np.random.rand(self.n,1)<self.coherence
		notc = (c==False)
		notcn = np.sum(notc)
		#INITDOTS
		#focal length to projection plane
		#projection plane is defined to be 
		#1 unit wide and high, so with 
		#this focal length, we are looking at
		#a view of the world with a 90 deg fov
		dotsf = 0.5
		#translation and rotation matrices
		#get the step size
		dotsT = [0,0,self.direction*self.velocity]        
		dotsR = [0,0,0]
		#maximum depth of points
		dotsmaxZ = 10
		dotsminZ = dotsf
		dotsmaxX = 10
		dotsmaxY = 10       
		#initial position of dots     
		#screen-centered        
		xc = np.rint(self.x/2)
		yc = np.rint(self.y/2)
		np.random.seed(0)
		#Coordinates in 3D volume
		Xs = np.rint(np.transpose(np.random.randint(xc-dotsmaxX,xc+dotsmaxX,(1,self.n))))
		Ys = np.rint(np.transpose(np.random.randint(yc-dotsmaxY,yc+dotsmaxY,(1,self.n))))        
		Zs = (dotsmaxZ-dotsminZ)*np.random.rand(1,self.n)+dotsminZ        
		Zs = np.rint(np.transpose(Zs))                  
		#get projection on to plane
		Xsc = xc-Xs
		Ysc = yc-Ys        
		dotsxproj = dotsf*Xsc/Zs
		dotsyproj = dotsf*Ysc/Zs        
		#put back into screen coordinates in x,y plane
		xs = self.x*dotsxproj+xc
		ys = self.y*dotsyproj+yc
		#after scaling        
		#UPDATE DOTS
		# Repeatedly place and then move
		for t in np.arange(self.t):
                        
			#generate a random transformation matrix for each incoherent point
			randT = np.random.rand(3,notcn)-0.5;             
			#and normalize the transformation to have the same length
			#(i.e. speed) as the real transformation matrix
			randT = np.sqrt(np.sum(np.power(dotsT,2)))*randT/(np.outer( np.transpose([1,1,1]),np.sqrt( np.sum(np.power(randT,2)))))
            
			#move coherent dots in 3-space to observer           
			Xs[c] = Xs[c]-dotsT[0]          
			Ys[c] = Ys[c]-dotsT[1]
			Zs[c] = Zs[c]-dotsT[2]
        
# 			#move incoherent points according to the random trasnformation
			Xs[notc] = Xs[notc]-randT[0,:]
			Ys[notc] = Ys[notc]-randT[1,:]
			Zs[notc] = Zs[notc]-randT[2,:]
                              
			#get points that have moved off the screen            
			offscreen = Zs<dotsminZ        
			#and put them at the furthest distance
			Zs[offscreen] = dotsmaxZ
			#get all points that have moved out of view
			offscreen = Zs>dotsmaxZ
			#and move them to the front plane
			Zs[offscreen] = dotsminZ   

			#put points moved off the X edge back
			offscreen = Xs<xc-dotsmaxX
			Xs[offscreen] = xc+dotsmaxX                                   
			offscreen = Xs>xc+dotsmaxX
			Xs[offscreen] = xc-dotsmaxX                                   
			#put points moved off the Y edge back
			offscreen = Ys<yc-dotsmaxY
			Ys[offscreen] = yc+dotsmaxY                                   
			offscreen = Ys>yc+dotsmaxY
			Ys[offscreen] = yc-dotsmaxY
            
			#project on to screen             
			Xsc = xc - Xs        
			Ysc = yc - Ys          
			dotsxproj = dotsf*Xsc/Zs
			dotsyproj = dotsf*Ysc/Zs                         
			#put back into screen coordinates in x,y plane
			#after scaling
			xs = self.x*dotsxproj+xc
			ys = self.y*dotsyproj+yc
			#set xs within screen boundaries
			inBound1 = np.logical_and(xs<self.x, ys<self.y)    
			inBound2 = np.logical_and(xs>0, ys>0)                
			inBound = np.logical_and(inBound1,inBound2)                
			xsIn = xs[inBound]     
			ysIn = ys[inBound]                    
			cIn  = c[inBound]    
			notcIn = notc[inBound]     
			notcnIn = np.sum(notc)        
			# Revert to int so positions can be parsed correctly                                                       
			xs = xs.astype(int)
			ys = ys.astype(int) 
			xsIn = xsIn.astype(int)
			ysIn = ysIn.astype(int)             
# 			set white point                          
			for i in np.arange(len(xsIn)): 
				self.data[xsIn[i],ysIn[i],t] = 255
                                                                                                   
            
	# File saving (for later loading into convnets)
	def fout(self):
		print "todo: implement save to file"

	# Display figure showing this motion
	def plot(self):
		fig = plt.figure()
		ax = plt.gca()
		im = plt.imshow(self.data[:,:,0], cmap='Greys_r', vmin=0, vmax=255)	
		def up(t):
			#set in motion
			im.set_array(self.data[:,:,t])		
			return im,
		ani = anim.FuncAnimation(fig, up, np.arange(self.t), interval=50, blit=True, repeat_delay=1000)
		plt.show()
		return ani