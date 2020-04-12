from hrr import HRR
import torch
import unittest


class HRRTests(unittest.TestCase):
	def test_single_pair(self):
		size = (40, 30)
		real = .01 * torch.rand(size)
		im = .01 * torch.rand(size)
		mem = HRR(real, im)

		k = HRR(2*torch.rand(size), 3*torch.rand(size))
		v = HRR(5*torch.ones(size), 2*torch.ones(size))
		mem = mem.set(k, v)

		restored = mem.get(k)
		torch.testing.assert_allclose(
			restored.real, 
			v.real, 
			rtol=0, 
			atol=.1
		)
		torch.testing.assert_allclose(
			restored.im, 
			v.im, 
			rtol=0, 
			atol=.1
		)
		restored_phase = mem.get_phase(k)
		torch.testing.assert_allclose(
			restored_phase, 
			v.phase, 
			rtol=0, 
			atol=.1
		)

	def test_multiple_pairs(self):
		size = (40, 30)
		real = .01 * torch.rand(size)
		im = .01 * torch.rand(size)
		mem = HRR(real, im)

		k = HRR(torch.rand(size), torch.rand(size))
		v = HRR(5*torch.ones(size), 2*torch.ones(size))
		mem = mem.set(k, v)

		k = HRR(torch.rand(size), torch.rand(size))
		v = HRR(5*torch.ones(size), 2*torch.ones(size))
		mem = mem.set(k, v)

		k = HRR(torch.rand(size), torch.rand(size))
		v = HRR(5*torch.ones(size), 2*torch.ones(size))
		mem = mem.set(k, v)

		restored_phase = mem.get_phase(k)
		torch.testing.assert_allclose(
			restored_phase, 
			v.phase, 
			rtol=0, 
			atol=.1
		)
		# TODO: Do we need duplicates to fix this test?

if __name__ == '__main__':
    unittest.main()