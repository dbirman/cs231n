import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

class Translate():

	def __init__(self,x,y,t,n,velocity,theta):
		self.x = x
		self.y = y
		self.t = t
		self.n = n
		self.data = np.zeros((x,y,t))
		self.velocity = velocity
		self.theta = theta

	def gen(self):
		self.data = np.zeros((self.x,self.y,self.t))
		xs = np.random.randint(0,self.x,(self.n,1))
		ys = np.random.randint(0,self.y,(self.n,1))
		for t in np.arange(self.t):
			for x in xs:
				for y in ys:
					self.data[x,y,t] = 255
			# accumulate
			xs = np.rint(xs + self.velocity * np.cos(self.theta))
			ys = np.rint(ys + self.velocity * np.sin(self.theta))
			xs[xs>=self.x] = xs[xs>=self.x]-self.x
			xs[xs<0] = xs[xs<0]+self.x
			ys[ys>=self.y] = ys[ys>=self.y]-self.y
			ys[ys<0] = ys[ys<0]+self.x
			xs = xs.astype(int)
			ys = ys.astype(int)

	def fout(self):
		print "todo: implement save to file"

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
