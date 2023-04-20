
import unittest
from SeisCL.python import ComputeRessource


class ComputeRessourceTester(unittest.TestCase):

    def test_opencl_ressources(self):
        resc = ComputeRessource()
        self.assertIsNotNone(resc.devices)
