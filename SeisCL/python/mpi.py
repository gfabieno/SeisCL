from  mpi4py import MPI

#TODO This is brainstorming how to do MPI domain decomposition

class MPIGRID:
    """
    A grid containing subgrids assigned to MPI processes
    Grid is divided along the last dimension of the grid
    """

    halosize = None # size of halo for communication


    def init(self, grid):
        """
        Initialize MPI if not already, then divides the grid into subgrids

        :param grid:
        :return:
        """

    def synhronize(self):
        "Performs the required communications between subgrids, non-blocking"


    def local_position(self):
        """
        Get the local
        :return:
        """

class StateKernelMPI:
    """
    Divide a kernel into several regions, make subgrid and manage data transfer
    """