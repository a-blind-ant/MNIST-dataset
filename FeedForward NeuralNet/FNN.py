import csv
import numpy as np
#######################################################################################################
#TRAINING INPUT#
filename = "mnist_train.csv"
images = []
with open(filename, 'r') as csvfile:
	csvreader = csv.reader(csvfile)
	for row in csvreader:
		images.append(row)
#######################################################################################################

L = 2	# Number of hidden layers
n = [20]*(L+2)	
n[1] = 121
n[2] = 61
# In the case of MNIST data set:
n[0] = 785
n[L+1] = 10

#######################################################################################################
# Neural net architecture and functions:

class neural_nets:
	#Constructor
	def __init__(self, L, n):
		self.n = [x for x in n]
		self.L = L
		ls_w = [1]*(L+2) #Weights do not exist for the input layer/ They 're [ones]

		for l in range(1,L+2):
			a = [[np.random.randn()*0.1 for i in range(n[l])] for j in range(n[l-1])]
			ls_w[l] = a

		self.ls_w = ls_w
		self.alpha = 0.211	#Learning rate of the network
		self.lamda = 0.00	#Hyperparameter for regularization in the cost function

	#Activation function
	def f(self, arg):
		ans = [1/(1 + np.exp(-x))	 for x in arg]
		return ans
			
	#Training the neural nets
	def train(self, x):
		if x == 0:
			return
		else:
			for i in range(7000):
				img = images[i][1:] + [1]
				label = images[i][0]
				network = self.calc_out(img)
				y_dash = network[L+1]
				y = [0]*10
				y[int(label)] = 1
				costly_list = [q-w for q,w in zip(np.multiply(([x-1 for x in y]),(np.log([1-x for x in y_dash]))) , np.multiply(y,np.log(y_dash)) )]
				ron = sum(costly_list)
				print(sum(costly_list))
				error = [0]*(L+2)
				error[L+1] = np.multiply([p-q for p,q in zip(network[L+1], y)], [x*(1-x) for x in network[L+1]])
				layer = L
				while layer>0:
					mat1 = np.dot((self.ls_w[layer+1]), error[layer+1])
					error[layer] = np.multiply(mat1, [x*(1-x) for x in network[layer]])
					#error[layer] = np.multiply(mat1, [int(x>0) for x in network[layer]])
					layer-=1
						
				#Weight Update:
				layer = L+1
				while layer>0:
					weights = [x for x in self.ls_w[layer]]
					weights = np.multiply(1-self.lamda*self.alpha, weights)
					temp = [[x] for x in network[layer-1]]
					temp2 = [[x] for x in error[layer]]
					subtract_me = np.transpose(np.multiply(self.alpha, (np.dot(temp2, np.transpose(temp)))))
					self.ls_w[layer] = np.subtract(weights, subtract_me)
					layer-=1
				if ron<0.0001:
					return		
			print(x)
			self.train(x-1)	 	

	def calc_out(self, img1):
		img3 = [int(x) for x in img1]
		mu = sum(img3)
		mu /= len(img3)
		sigma = sum(np.multiply(img3,img3))
		sigma /= len(img3)
		sigma = sigma - (mu**2)
		img = [(x-mu)/np.sqrt(sigma) for x in img3]
		img[len(img)-1] = 1
		z = [0]*(L+2)
		z[0] = [float(x) for x in img]
		for i in range(1,L+2):
			temp = np.dot(np.transpose(self.ls_w[i]), z[i-1])
			if i<L+1:
				temp[0] = 1
				z[i] = self.f(temp)
			else:
				temp = [np.exp(x) for x in temp]
				denom = sum(temp)
				z[i] = [x/denom for x in temp]
		return z


#################################################################################################
my_net = neural_nets(L, n)	#The network we are going to use.
my_net.train(30)
rows = [np.argmax(my_net.calc_out((img[1:])+[1])[L+1]) for img in images]
#################################################################################################

#WRITING OUTPUT ONTO A CSV FILE

filename = "mnist_test.csv"
images = []
with open(filename, 'r') as csvfile:
	csvreader = csv.reader(csvfile)
	for row in csvreader:
		images.append(row)

fields = ['id', 'label']
rows = [np.argmax(my_net.calc_out((img)+[1])[L+1]) for img in images]
filename = "output.csv"

with open(filename, 'w') as csvfile:
	csvwriter = csv.writer(csvfile)
	csvwriter.writerow(fields)
	for i in range(len(rows)):
		csvwriter.writerows([[i+1,rows[i]]])
