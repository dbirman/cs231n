import numpy as np
import matplotlib.pyplot as plt
import random as rd
import math
import matplotlib.animation as anim

#############
# Rotate #
#############
#usage
#r = Rotational(32,32,16,200,30*1.0/180*np.pi,1,1,-1)
#r.gen()
#r.plot()

#direction : [1 -1] for clockwise/counterclockwise rotation

# Rotational motion in 360 degs
class Rotational():

	# Initialize
	def __init__(self,imageWidth,imageHeight,t,n_dots,rotational_speed,dot_radius,frames_per_second,direction):
		# Data size
		self.time = t
		self.imageWidth = imageWidth #in pixels
		self.imageHeight = imageHeight
		self.maxX = imageWidth/2
		self.maxY = imageHeight/2
		self.num_dots = n_dots
		self.speed = rotational_speed*direction #radians per second
		self.frames_per_second = frames_per_second
		self.dot_radius = dot_radius

	# Generate a new motion stimulus
	def gen(self):
		time = self.time 
		imageWidth = self.imageWidth
		imageHeight = self.imageHeight
		maxX = self.maxX 
		maxY = self.maxY 
		num_dots = self.num_dots 
		speed = self.speed 
		frames_per_second = self.frames_per_second 
		dot_radius = self.dot_radius 
		length = math.sqrt(math.pow(imageWidth,2)+math.pow(imageHeight,2))
		position = [[rd.random()*length - length/2,rd.random()*length - length/2] for i in range(num_dots)]
		#convert to polar coordinates
		for i in range(num_dots):
			r = np.sqrt(math.pow(position[i][0],2)+math.pow(position[i][1],2))
			theta = np.arctan2(position[i][1],position[i][0])
			position[i][:] = [r,theta]

		self.data = [[[0 for i in range(imageWidth)] for j in range(imageHeight)] for k in range(time*frames_per_second)]
		for t in range(time*frames_per_second):
			#convert to cartesian
			position_cartesian = [[position[i][0]*np.cos(position[i][1]),position[i][0]*np.sin(position[i][1])] for i in range(len(position))]
			for k in range(len(position_cartesian)):
				for x in range(int(round(position_cartesian[k][0]))-1-dot_radius,int(round(position_cartesian[k][0]))+dot_radius+1):
					for y in range(int(round(position_cartesian[k][1]))-1-dot_radius,int(round(position_cartesian[k][1]))+dot_radius+1):
						i = maxY - y
						j = x + maxX
						i = min(max(i,0),imageHeight-1) #constrain within range
						j = min(max(j,0),imageWidth-1) #constrain within range
						y_corr = maxY - i
						x_corr = j - maxX
						dist = math.sqrt(math.pow(x_corr-position_cartesian[k][0],2) + math.pow(y_corr-position_cartesian[k][1],2))
						self.data[t][i][j] = min(1,self.data[t][i][j]+math.exp(-math.pow(dist/dot_radius,2)));
			#rotate position
			for i in range(num_dots):
				position[i][1] = position[i][1] + speed/frames_per_second;

	# File saving (for later loading into convnets)
	def fout(self):
		print "todo: implement save to file"

	# Display figure showing this motion
	def plot(self):
		fig = plt.figure()
		ax = plt.gca()
		im = plt.imshow(self.data[0][:][:], cmap='Greys_r', vmin=0, vmax=1)	
		def up(t):
			im.set_array(self.data[t][:][:])
			return im,
		ani = anim.FuncAnimation(fig, up, np.arange(self.time*self.frames_per_second), interval=1000.0/self.frames_per_second, blit=True, repeat_delay=1000)
		plt.show()
		return ani
