from complex import ComplexTensor
from memory import NeuralMemory
import torch

# Complex holographic reduced representation (HRR) with duplicatation
# to reduce retrieval noise. HRRs act as efficient key-value stores 
# with unbounded capacity but noisy reconstruction. 
class HRR(NeuralMemory):

	def __init__(self, data, num_copies=None, permutations=None):
		# TODO: Allow random initialization.
		self.data = data

		# TODO: Generate permutations (duplicates)
		self.num_copies = num_copies


	def write(self, k, v):
		self.data += k * v


	def read(self, k):
		# TODO: Average over permutations
		return k.inv * self.data


	def get_phase(self, k):
		# If we're only interested in recovering the phase,
		# we can use the conjugate instead of the inverse.
		# TODO: Average over permutations
		return (k.conj * self.data).phase