import torch


class ComplexTensor:
	def __init__(self, real, im):
		self.real = real
		self.im = im


	@property
	def square_modulus(self):
		return self.real ** 2 + self.im ** 2


	@property
	def modulus(self):
		return torch.sqrt(self.square_modulus)


	@property
	def phase(self):
		# TODO: Handle zeros.
		return torch.atan(self.im / self.real)


	@property	
	def conj(self):
		return ComplexTensor(real=self.real, im=-1 * self.im)


	@property
	def inv(self):
		return (self.conj / self.square_modulus)


	def __add__(self, x):
		return ComplexTensor(real=self.real + x.real, im=self.im + x.im)


	def __mul__(self, x):
		return ComplexTensor(real=self.real * x.real - self.im * x.im,
		 					 im=self.real * x.im + self.im * x.real)


	def __truediv__(self, x):
		return ComplexTensor(self.real / x, self.im / x)