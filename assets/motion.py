import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

#############
# TRANSLATE #
#############

# Translational motion in 360 degs
class Translate():

	# Initialize
	def __init__(self,x,y,t,n,velocity,theta):
		# Data size
		self.x = x
		self.y = y
		self.t = t
		self.data = np.zeros((x,y,t))
		# Dot num
		self.n = n
		# Speed 
		self.velocity = velocity
		# Angle in Radians np.pi (0 is down)
		self.theta = theta

	# Generate a new motion stimulus
	def gen(self):
		# Initialize
		self.data = np.zeros((self.x,self.y,self.t))
		xs = np.random.randint(0,self.x,(self.n,1))
		ys = np.random.randint(0,self.y,(self.n,1))
		# Repeatedly place and then move
		for t in np.arange(self.t):
			for i in np.arange(len(xs)):
				self.data[xs[i],ys[i],t] = 255
			# Move all dots
			xs = np.rint(xs + self.velocity * np.cos(self.theta))
			ys = np.rint(ys + self.velocity * np.sin(self.theta))
			# Fix dots that go offscreen
			xs[xs>=self.x] = xs[xs>=self.x]-self.x
			xs[xs<0] = xs[xs<0]+self.x
			ys[ys>=self.y] = ys[ys>=self.y]-self.y
			ys[ys<0] = ys[ys<0]+self.x
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
			im.set_array(self.data[:,:,t])
			return im,
		ani = anim.FuncAnimation(fig, up, np.arange(self.t), interval=50, blit=True, repeat_delay=1000)
		plt.show()
		return ani