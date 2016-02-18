import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

#############
# TRANSLATE #
#############

# Translational motion in 360 degs
class Translate():

	# Initialize
	def __init__(self,x,y,t,n,velocity,theta,coherence=1.0):
		# Data size
		self.x = x
		self.y = y
		self.t = t
		self.data = np.zeros((t,x,y))
		# Dot num
		self.n = n
		# Speed 
		self.velocity = velocity
		# Angle in Radians np.pi (0 is down)
		self.theta = theta
		# Coherence
		self.coherence = coherence

	# Generate a new motion stimulus
	def gen(self):
		# Initialize
		self.data = np.zeros((self.t,self.x,self.y),dtype='uint8')
		c = np.random.rand(self.n,1)<self.coherence
		notc = (c==False)[:,0]
		xs = np.random.randint(0,self.x,(self.n,1))
		ys = np.random.randint(0,self.y,(self.n,1))
		# Repeatedly place and then move
		for t in np.arange(self.t):
			for i in np.arange(len(xs)):
				self.data[t,xs[i],ys[i]] = 255
			# Move all coherent dots
			xs[c] = np.rint(xs[c] + self.velocity * np.cos(self.theta))
			ys[c] = np.rint(ys[c] + self.velocity * np.sin(self.theta))
			# Move all incoherent dots
			inc_thetas = np.random.rand(np.sum(notc),1)*np.pi*2
			xinc = self.velocity * np.cos(inc_thetas)
			yinc = self.velocity * np.sin(inc_thetas)
			xs[notc] = np.rint(np.add(xs[notc], xinc))
			ys[notc] = np.rint(np.add(ys[notc], yinc))
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
		im = plt.imshow(self.data[0,:,:], cmap='Greys_r', vmin=0, vmax=255)	
		def up(t):
			im.set_array(self.data[t,:,:])
			return im,
		ani = anim.FuncAnimation(fig, up, np.arange(self.t), interval=50, blit=True, repeat_delay=1000)
		plt.show()
		return ani
