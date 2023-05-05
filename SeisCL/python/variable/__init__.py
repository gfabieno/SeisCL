from SeisCL.python import Configuration
from .opencl import VariableCL
from .variable import Variable

#TODO: choose wiich type of variables according to Configuration

# from .variable import Variable as VariableNumpy
# class Variable(VariableNumpy):
#
#     def __new__(cls, *args, **kwargs):
#         if Configuration.get("backend") == 'opencl':
#             return VariableCL(*args, **kwargs)
#         elif Configuration.get("backend") == 'numpy':
#             return VariableNumpy(*args, **kwargs)
#         else:
#             raise ValueError("Invalid backend")
