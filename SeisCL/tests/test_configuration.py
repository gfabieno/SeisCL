import unittest
from SeisCL.python.configuration import Configuration


class TestConfiguration(unittest.TestCase):

    def test_set_backend(self):
        Configuration.set("backend", 'opencl')
        self.assertEqual(Configuration.get("backend"), 'opencl')
        Configuration.set("backend", 'numpy')
        self.assertEqual(Configuration.get("backend"), 'numpy')
        with self.assertRaises(ValueError):
            Configuration.set("backend", 'invalid')

    def test_import_variable_cl(self):
        from SeisCL.python.variable.opencl import VariableCL
        Configuration.set("backend", 'opencl')
        from SeisCL.python import Variable
        self.assertIsInstance(Variable(None, shape=(1,)), VariableCL)

    def test_Variable_numpy(self):
        from SeisCL.python.variable.variable import Variable as BaseVariable
        from SeisCL.python import Variable
        Configuration.set("backend", 'numpy')
        self.assertIsInstance(Variable(shape=(1,)), BaseVariable)

