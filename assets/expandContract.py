
#expandContract.py
#
#
# author: steeve laquitaine
#   date: 2016/01/25
#purpose: simulate expansion
#
#  usage: 
#
# in ipython notebook
#
# 		from assets.expandContract import *
# 		%load_ext autoreload  
#		%autoreload 2
# 		%matplotlib nbagg
#
#		c = expandContract(64,64,32,400,1,1)
#		c.gen()
#		c.plot()
#
#inputs:
#         x,y,t : image width, height, time (e.g., 32,32,16)
#     direction : [1 or -1] for expand or contract
#      velocity : any non negative number (e.g., 1)
#             n : number of dots (e.g., 50)

#reference: inspired from Dan Birman motion.py module

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

##############
# OPTIC FLOW #
##############

# Optic flow forward (expanding) backward (contracting)
class expandContract():

	# Initialize
	def __init__(self,x,y,t,n,velocity,direction):
		# Data size
		self.x = x
		self.y = y
		self.t = t
		self.data = np.zeros((t,x,y))

		# Dot num
		self.n = n
		
		# Speed 
		self.velocity = velocity

		# direction
		self.direction = direction

		# Generate a new motion stimulus
	def gen(self):
		# Initialize
		self.data = np.zeros((self.t,self.x,self.y))
		
		#initial dots position
		xs = np.random.randint(0,self.x,(self.n,1))
		ys = np.random.randint(0,self.y,(self.n,1))

		#copy dots into holder "display" vectors
		dxs = np.copy(xs)
		dys = np.copy(ys)

		#dots motion directions are expanding
		#w.r.t to center
		xc = self.x/2
		yc = self.y/2
		self.theta = np.arctan2(xc-xs,ys-yc)

		# Repeatedly place and then move		
		for t in np.arange(self.t):			
			for i in np.arange(len(xs)):
				#update white dots position
				#each t
				self.data[t,dxs[i],dys[i]] = 255		
										
			## Move all dots							
			xs = xs-self.direction*self.velocity*np.sin(self.theta)
			ys = ys+self.direction*self.velocity*np.cos(self.theta)      
        
			if self.direction==1:
				# Fix dots that go offscreen
				xs[ys>=self.y-1] = xc
				ys[ys>=self.y-1] = yc
				ys[xs>=self.x-1] = yc
				xs[xs>=self.x-1] = xc
				xs[ys<0] = xc
				ys[ys<0] = yc
				ys[xs<0] = yc
				xs[xs<0] = xc            

			elif self.direction==-1:     
				# Fix dots offscreen
				a = np.where(ys>=self.y-1)[0]                
				xs[a] = np.random.randint(0,self.x-1,(len(a),1))                
				ys[a] = np.random.randint(0,self.y-1,(len(a),1))                
				b = np.where(xs>=self.x-1)[0]                                
				ys[b] = np.random.randint(0,self.y-1,(len(b),1))                
				xs[b] = np.random.randint(0,self.x-1,(len(b),1))  
				aa = np.where(ys<0)[0]                                
				xs[aa] = np.random.randint(0,self.x-1,(len(aa),1))                
				bb = np.where(xs<0)[0]                                                
				ys[aa] = np.random.randint(0,self.y-1,(len(aa),1))                
				ys[bb] = np.random.randint(0,self.y-1,(len(bb),1))                
				xs[bb] = np.random.randint(0,self.x-1,(len(bb),1))                
                
				dxs = np.rint(xs)
				dys = np.rint(ys)              
				# reset position of dots that reached the center
				if np.any(dys==np.rint(xc)):
					a = np.where(dys==np.rint(yc))[0]       
					xs[a] = np.random.randint(0,self.x-1,(len(a),1))                
					ys[a] = np.random.randint(0,self.y-1,(len(a),1))
				if np.any(dxs==np.rint(xc)):                       
					b = np.where(dxs==np.rint(xc))[0]                                   
					xs[b] = np.random.randint(0,self.x-1,(len(b),1))
					ys[b] = np.random.randint(0,self.y-1,(len(b),1))

				#re-calculate angle direction
				self.theta = np.arctan2(xc-xs,ys-yc)                       

			dxs = np.rint(xs)
			dys = np.rint(ys)
            
			# Revert to int so positions can be parsed correctly
			dxs = dxs.astype(int)
			dys = dys.astype(int)

	# File saving (for later loading into convnets)
	def fout(self):
		print "todo: implement save to file"

	# Display figure showing this motion
	def plot(self):
		fig = plt.figure()
		ax = plt.gca()
		im = plt.imshow(self.data[0,:,:], cmap='Greys_r', vmin=0, vmax=255)	
		def up(t):
			#set in motion
			im.set_array(self.data[t,:,:])		
			return im,
		ani = anim.FuncAnimation(fig, up, np.arange(self.t), interval=50, blit=True, repeat_delay=1000)
		plt.show()
		return ani

