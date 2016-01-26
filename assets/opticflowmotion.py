
#opticflowmotion.py
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
#		c = opticflowmotion(64,64,32,400,1)
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
class opticflowmotion():

	# Initialize
	def __init__(self,x,y,t,n,velocity):
		# Data size
		self.x = x
		self.y = y
		self.t = t
		self.data = np.zeros((x,y,t))

		# Dot num
		self.n = n
		
		# Speed 
		self.velocity = velocity			

		# Generate a new motion stimulus
	def gen(self):
		# Initialize
		self.data = np.zeros((self.x,self.y,self.t))
		
		#initial dots position
		xs = np.random.randint(0,self.x,(self.n,1))
		ys = np.random.randint(0,self.y,(self.n,1))
	
		#store initial position to re-initialize
		#dots out of boundaries
		xs0 = xs;
		ys0 = ys;
		
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
				self.data[xs[i],ys[i],t] = 255		
										
			## Move all dots							
			xs = np.rint(xs - self.velocity * np.sin(self.theta))
			ys = np.rint(ys + self.velocity * np.cos(self.theta))
				
			# Fix dots that go offscreen			
			xs[ys>=self.y] = xs0[ys>=self.y]
			ys[ys>=self.y] = ys0[ys>=self.y]
			ys[xs>=self.x] = ys0[xs>=self.x]
			xs[xs>=self.x] = xs0[xs>=self.x]			
			xs[ys<0] = xs0[ys<0]
			ys[ys<0] = ys0[ys<0]
			ys[xs<0] = ys0[xs<0]
			xs[xs<0] = xs0[xs<0]

			# Revert to int so positions can be parsed correctly
			xs = xs.astype(int)
			ys = ys.astype(int)

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

