import unittest
from .elastic import UpdateStress
from SeisCL.python import VariableCL as Variable
from SeisCL.python import ComputeRessource

class ElasticTester(unittest.TestCase):

    def setUp(self):

        nrec = 1
        nt = 3
        nab = 2
        self.dt = 0.00000000001
        self.dx = 1
        shape = (10, 10)
        self.resc = ComputeRessource()
        q = self.resc.queues[0]
        self.vp = Variable(q, shape=shape, initialize_method="random", pad=2)
        self.vs = Variable(q, shape=shape, initialize_method="random", pad=2)
        self.rho = Variable(q, shape=shape, initialize_method="random", pad=2)
        self.vx = Variable(q, shape=shape, initialize_method="random", pad=2)
        self.vz = Variable(q, shape=shape, initialize_method="random", pad=2)
        self.sxx = Variable(q, shape=shape, initialize_method="random", pad=2)
        self.szz = Variable(q, shape=shape, initialize_method="random", pad=2)
        self.sxz = Variable(q, shape=shape, initialize_method="random", pad=2)
        self.mu = Variable(q, shape=shape, initialize_method="random", pad=2)
        self.M = Variable(q, shape=shape, initialize_method="random", pad=2)
        self.muipkp = Variable(q, shape=shape, initialize_method="random", pad=2)
        self.vxout = Variable(q, shape=shape, initialize_method="random", pad=2)

    def test_scaledparameters(self):
        vp = self.vp; vs = self.vs; rho = self.rho
        dt = self.dt; dx = self.dx
        sp = ScaledParameters(dt, dx)
        self.assertLess(sp.backward_test(vp, vs, rho), 1e-12)
        self.assertLess(sp.linear_test(vp, vs, rho), 1e-05)
        self.assertLess(sp.dot_test(vp, vs, rho), 1e-12)

    def test_UpdateVelocity(self):
        vp = self.vp; vs = self.vs; rho = self.rho
        vx = self.vx; vz = self.vz
        sxx = self.sxx; szz = self.szz; sxz = self.sxz
        fun = UpdateVelocity()
        self.assertLess(fun.backward_test(rho, vx, vz, sxx, szz, sxz), 1e-12)
        self.assertLess(fun.linear_test(rho, vx, vz, sxx, szz, sxz), 1e-05)
        self.assertLess(fun.dot_test(rho, vx, vz, sxx, szz, sxz), 1e-12)

    def test_UpdateStress(self):
        M = self.M; mu = self.mu; muipkp = self.muipkp
        vx = self.vx; vz = self.vz
        sxx = self.sxx; szz = self.szz; sxz = self.sxz
        fun = UpdateStress(self.resc.queues[0], order=4)
        with self.subTest("backward"):
            self.assertLess(fun.backward_test(vx, vz, sxx, szz, sxz, M, mu, muipkp), 1e-6)
        with self.subTest("linear"):
            self.assertLess(fun.linear_test(vx, vz, sxx, szz, sxz, M, mu, muipkp), 1e-05)
        with self.subTest("dot_product"):
            self.assertLess(fun.dot_test(vx, vz, sxx, szz, sxz, M, mu, muipkp), 1e-12)

    def test_Cerjan(self):
        vx = self.vx; vz = self.vz
        fun = Cerjan()
        self.assertLess(fun.backward_test(vx, vz), 1e-12)
        self.assertLess(fun.linear_test(vx, vz), 1e-12)
        self.assertLess(fun.dot_test(vx, vz), 1e-12)

    def test_PoinSource(self):
        vx = self.vx
        fun = PointForceSource()
        self.assertLess(fun.backward_test(vx, 1, (0, 1)), 1e-12)
        self.assertLess(fun.linear_test(vx, 1, (0, 1)), 1e-12)
        self.assertLess(fun.dot_test(vx, 1, (0, 1)), 1e-12)

    def test_Geophone(self):
        vx = self.vx; vxout = self.vxout
        fun = Geophone()
        self.assertLess(fun.backward_test(vx, vxout, ((5, 5), (6, 6))), 1e-12)
        self.assertLess(fun.linear_test(vx, vxout, ((5, 5), (6, 6))), 1e-12)
        self.assertLess(fun.dot_test(vx, vxout, ((5, 5), (6, 6))), 1e-12)

    def test_elastic2d_propagator(self):
        grid = Grid(nd=2, nx=10, ny=None, nz=10, nt=3, dt=0.00000000001, dh=1.0,
                    nab=2, freesurf=True)
        shot = Shot([Source()], [Receiver(x=0), Receiver(x=1)], 0,
                    grid.nt, grid.dt)
        self.assertIsNot(shot.wavelet, None)
        acquisition = Acquisition(grid=grid, shots=[shot])
        propagator = Elastic2dPropagator(acquisition)
        shot = propagator.acquisition.shots[0]
        vp = propagator.vp; vs = propagator.vs; rho = propagator.rho
        @TapedFunction
        def prop(shot, vp, vs, rho):
            return propagator.propagate(shot, vp, vs, rho)
        self.assertLess(prop.backward_test(shot, vp, vs, rho), 1e-12)
        self.assertLess(prop.linear_test(shot, vp, vs, rho), 1e-05)
        self.assertLess(prop.dot_test(shot, vp, vs, rho), 1e-12)