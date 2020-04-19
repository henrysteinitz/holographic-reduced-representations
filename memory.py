from torch.nn import Module


class NeuralMemory(Module):
	
	def forward(self, k, v=None):
		if v is None:
			return self.read(k)
		else:
			return self.write(k, v)


	def write(self, k, v):
		raise NotImplementedError


	def read(self, k):
		raise NotImplementedError