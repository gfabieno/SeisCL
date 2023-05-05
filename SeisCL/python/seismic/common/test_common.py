import unittest
from SeisCL.python.seismic import Velocity2Lame, Velocity2LameGPU
from SeisCL.python.seismic import PointSources2DGPU, PointSources3DGPU, Acquisition, Shot, Source
from SeisCL.python import Variable, VariableCL, ComputeRessource


class TestVelocity2Lame(unittest.TestCase):

    def test_GPU_version(self):

        resc = ComputeRessource()
        q = resc.queues[0]
        for nd in range(1, 4):
            shape = (10, ) * nd
            vp = VariableCL(q, shape=shape, initialize_method="random", pad=2)
            vs = VariableCL(q, shape=shape, initialize_method="random", pad=2)
            rho = VariableCL(q, shape=shape, initialize_method="random", pad=2)
            v2l = Velocity2LameGPU(q)
            with self.subTest(nd=nd, msg="Velocity2LameGPU: backward test"):
                self.assertLess(v2l.backward_test(vp, vs, rho), 1e-06)
            with self.subTest(nd=nd, msg="Velocity2LameGPU: linear test"):
                self.assertLess(v2l.linear_test(vp, vs, rho), 1e-11)
            with self.subTest(nd=nd, msg="Velocity2LameGPU: dot product test"):
                self.assertLess(v2l.dot_test(vp, vs, rho), 1e-06)


#TODO finish that test case
class TestPointSources(unittest.TestCase):

    def test_GPU_version(self):

        resc = ComputeRessource()
        q = resc.queues[0]
        for nd in range(2, 4):

            if nd == 2:
                stypes = ["vz", "vx", "p"]
                srcfun = PointSources2DGPU(q)
            elif nd == 3:
                stypes = ["vz", "vy", "vx", "p"]
                srcfun = PointSources3DGPU(q)

            for stype in stypes:
                shape = (10, ) * nd
                dh = 1.5
                t = 2
                nt = 10
                pos = [(4*dh, ) * nd, (5*dh, ) * nd]
                sources = [Source(*p, type=stype) for p in pos]
                wavelet = VariableCL(q, shape=(nt,len(pos)), initialize_method="random")
                shot = Shot(sources, [], 0, 10, 1, wavelet=wavelet)
                fields = [VariableCL(q, shape=shape, initialize_method="random",
                                     pad=2) for _ in range(2*nd)]
                src_pos = srcfun.src_pos(shot, dh, shape)
                src_type = srcfun.src_type(shot)
                srcfun(*fields, shot.wavelet, src_pos, src_type, t)
                #TODO verify that the result is correct

