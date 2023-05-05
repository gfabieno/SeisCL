

class Configuration:
    __conf = {
        "backend": "numpy",
        "mpi": False,
    }
    __setters = ["backend", "password"]

    __allowed = {
        "backend": ["opencl", "numpy"],
        "mpi": [True, False],
    }

    @staticmethod
    def get(name):
        return Configuration.__conf[name]

    @staticmethod
    def set(name, value):
        if name in Configuration.__setters:
            if name in Configuration.__allowed:
                if value not in Configuration.__allowed[name]:
                    raise ValueError("Allowed values for {} are {}".format(
                        name, Configuration.__allowed[name]))
            Configuration.__conf[name] = value
        else:
            raise NameError("Name not accepted in set() method")