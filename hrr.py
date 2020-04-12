import torch

# Complex holographic reduced representation (HRR) with duplicatation
# to reduce retrieval noise. HRRs act as efficient key-value stores 
# with unbounded capacity but noisy reconstruction. 

# TODO: Consider making class explicitly immuatble.
class HRR:
	
	def __init__(self, real, im, num_copies=None, permutations=None):
		self.real = real
		self.im = im

		# TODO: Generate permutations (duplicates)
		self.num_copies = num_copies


	@property
	def modulus(self):
		return torch.sqrt(self.real ** 2 + self.im ** 2)


	@property
	def phase(self):
		# TODO: Handle zeros.
		return torch.atan(self.im / self.real)

	@property	
	def conj(self):
		return HRR(real=self.real, im=-1 * self.im)


	def bind(self, x):
		return HRR(
			real=self.real * x.real - self.im * x.im,
			im=self.real * x.im + self.im * x.real
		)


	def store(self, x):
		return HRR(
			real=self.real + x.real,
			im=self.im + x.im
		)


	def set(self, k, v):
		return self.store(self.bind(k, v))


	def get(self, k):
		# TODO: Compare conjugate retrieval to inverse
		# retrieval
		return k.conj * self


	def __add__(self, x):
		return self.store(x)


	def __mul__(self, x):
		return self.bind(x)
	
	
