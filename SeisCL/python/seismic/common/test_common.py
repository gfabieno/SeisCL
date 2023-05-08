import unittest
import numpy as np
from SeisCL.python.seismic import Velocity2Lame, Velocity2LameGPU
from SeisCL.python.seismic import (PointSources2DGPU, PointSources3DGPU,
                                   GeophoneGPU2D, GeophoneGPU3D,
                                   Acquisition, Shot, Source, Receiver)
from SeisCL.python.seismic import Cerjan, CerjanGPU
from SeisCL.python.seismic import ArithmeticAveraging, ArithmeticAveraging2, HarmonicAveraging
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
                wavelet = VariableCL(q, shape=(nt,len(pos)),
                                     initialize_method="random")
                shot = Shot(sources, [], 0, 10, 1, wavelet=wavelet)
                fields = [VariableCL(q, shape=shape, initialize_method="random",
                                     pad=2) for _ in range(2*nd)]
                fieldsnp = [Variable(data=f.data.get()) for f in fields]
                src_pos = srcfun.src_pos(shot, dh, shape)
                src_type = srcfun.src_type(shot)
                srcfun(*fields, shot.wavelet, src_pos, src_type, t)
                args = srcfun.arguments(*fields, shot.wavelet, src_pos,
                                        src_type, t)
                argsnp = srcfun.arguments(*fieldsnp, shot.wavelet, src_pos,
                                          src_type, t)
                if stype in args:
                    f1 = args[stype].data.get().flatten()[src_pos.get()]
                    f2 = argsnp[stype].data.flatten()[src_pos.get()]
                    f2 += wavelet.data.get()[t, :]
                    self.assertTrue(np.allclose(f1, f2))
                elif stype == "p":
                    sxx = args["sxx"].data.get().flatten()[src_pos.get()]
                    sxx0 = argsnp["sxx"].data.flatten()[src_pos.get()]
                    sxx0 += wavelet.data.get()[t, :] / nd
                    self.assertTrue(np.allclose(sxx, sxx0))
                    if nd == 3:
                        syy = args["syy"].data.get().flatten()[src_pos.get()]
                        syy0 = argsnp["syy"].data.flatten()[src_pos.get()]
                        syy0 += wavelet.data.get()[t, :] / nd
                        self.assertTrue(np.allclose(syy, syy0))
                    szz = args["szz"].data.get().flatten()[src_pos.get()]
                    szz0 = argsnp["szz"].data.flatten()[src_pos.get()]
                    szz0 += wavelet.data.get()[t, :] / nd
                    self.assertTrue(np.allclose(szz, szz0))

            with self.subTest(nd=nd, msg="PointSourcesGPU: backward test"):
                self.assertLess(srcfun.backward_test(*fields, shot.wavelet,
                                                     src_pos, src_type, t),
                                1e-06)


class TestReceivers(unittest.TestCase):

    def test_GPU_version(self):

        resc = ComputeRessource()
        q = resc.queues[0]
        for nd in range(2, 4):

            if nd == 2:
                rtypes = ["vz", "vx", "p"]
                recfun = GeophoneGPU2D(q)
            elif nd == 3:
                rtypes = ["vz", "vy", "vx", "p"]
                recfun = GeophoneGPU3D(q)

            for rtype in rtypes:
                shape = (10, ) * nd
                dh = 1.5
                t = 2
                nt = 10
                pos = [(4*dh, ) * nd, (5*dh, ) * nd]
                receivers = [Receiver(*p, type=rtype) for p in pos]
                dmod = VariableCL(q, shape=(nt, len(pos)),
                                  initialize_method="random")
                shot = Shot([], receivers, 0, 10, 1, dmod=dmod)
                fields = [VariableCL(q, shape=shape, initialize_method="random",
                                     pad=2) for _ in range(2*nd)]
                rec_pos = recfun.rec_pos(shot, dh, shape)
                rec_type = recfun.rec_type(shot)
                recfun(*fields, shot.dmod, rec_pos, rec_type, t)
                args = recfun.arguments(*fields, shot.dmod, rec_pos, rec_type,
                                        t)
                if rtype in args:
                    f1 = args[rtype].data.get().flatten()[rec_pos.get()]
                    f2 = shot.dmod.data.get()[t, :]
                    self.assertTrue(np.allclose(f1, f2))
                elif rtype == "p":
                    f1 = shot.dmod.data.get()[t, :]
                    sxx = args["sxx"].data.get().flatten()[rec_pos.get()]
                    szz = args["szz"].data.get().flatten()[rec_pos.get()]
                    if nd == 3:
                        syy = args["syy"].data.get().flatten()[rec_pos.get()]
                        f2 = (sxx + syy + szz) / nd
                    else:
                        f2 = (sxx + szz) / nd
                    self.assertTrue(np.allclose(f1, f2))

            # with self.subTest(nd=nd, msg="PointSourcesGPU: backward test"):
            #     self.assertLess(recfun.backward_test(*fields, shot.wavelet,
            #                                          rec_pos, rec_type, t),
            #                     1e-06)
            with self.subTest(nd=nd, msg="GeophoneGPU: linear test"):
                self.assertLess(recfun.linear_test(*fields, shot.dmod, rec_pos,
                                                   rec_type, t),
                                1e-06)
            with self.subTest(nd=nd, msg="GeophoneGPU: linear test"):
                self.assertLess(recfun.dot_test(*fields, shot.dmod, rec_pos,
                                                rec_type, t),
                                1e-06)


class TestCerjan(unittest.TestCase):

    def test_Cerjan_GPU(self):
        resc = ComputeRessource()
        q = resc.queues[0]
        shape = (10, ) * 2
        vx = VariableCL(q, shape=shape, initialize_method="random", pad=2)
        vz = VariableCL(q, shape=shape, initialize_method="random", pad=2)
        fun = CerjanGPU(q)
        self.assertLess(fun.backward_test(vx, vz), 1e-08)
        self.assertLess(fun.linear_test(vx, vz), 1e-2)
        self.assertLess(fun.dot_test(vx, vz), 1e-06)

    def test_Cerjan_gpu_numpy(self):
        resc = ComputeRessource()
        q = resc.queues[0]
        shape = (10, ) * 2
        vxgpu = VariableCL(q, shape=shape, initialize_method="random", pad=2)
        vzgpu = VariableCL(q, shape=shape, initialize_method="random", pad=2)
        vxnp = Variable(data=vxgpu.data.get(), pad=2)
        vznp = Variable(data=vzgpu.data.get(), pad=2)
        self.assertTrue(np.allclose(vxgpu.data.get(), vxnp.data))
        self.assertTrue(np.allclose(vzgpu.data.get(), vznp.data))
        CerjanGPU(q)(vxgpu, vzgpu)
        Cerjan()(vxnp, vznp)
        self.assertTrue(np.allclose(vxgpu.data.get(), vxnp.data))
        self.assertTrue(np.allclose(vzgpu.data.get(), vznp.data))

    def test_Cerjan_numpy(self):
        shape = (10, ) * 2
        vx = Variable(shape=shape, initialize_method="random", pad=2)
        vz = Variable(shape=shape, initialize_method="random", pad=2)
        fun = Cerjan()
        self.assertLess(fun.backward_test(vx, vz), 1e-08)
        self.assertLess(fun.linear_test(vx, vz), 1e-2)
        self.assertLess(fun.dot_test(vx, vz), 1e-06)


class TestAveragin(unittest.TestCase):

    def test_ArithmeticAveraging(self):

        resc = ComputeRessource()
        q = resc.queues[0]
        for nd in [2, 3]:

            shape = (10,) * nd
            v = VariableCL(q, shape=shape, initialize_method="random",
                            pad=2)
            ave = VariableCL(q, shape=shape, initialize_method="random",
                             pad=2)
            fun = ArithmeticAveraging(q)
            with self.subTest(nd=nd, msg="ArithmeticAveraging backward test"):
                self.assertLess(fun.backward_test(v, ave, dx=1, dz=0), 1e-06)
            with self.subTest(nd=nd, msg="ArithmeticAveraging linear test"):
                self.assertLess(fun.linear_test(v, ave, dx=1, dz=0), 1e-2)
            with self.subTest(nd=nd, msg="ArithmeticAveraging dot test"):
                self.assertLess(fun.dot_test(v, ave, dx=1, dz=0), 1e-06)

    def test_ArithmeticAveraging2(self):

        resc = ComputeRessource()
        q = resc.queues[0]
        for nd in [2, 3]:

            shape = (10,) * nd
            v = VariableCL(q, shape=shape, initialize_method="random",
                           pad=2)
            ave = VariableCL(q, shape=shape, initialize_method="random",
                             pad=2)
            fun = ArithmeticAveraging2(q)
            with self.subTest(nd=nd, msg="ArithmeticAveraging backward test"):
                self.assertLess(fun.backward_test(v, ave, dx1=1, dz1=1,
                                                  dx2=1, dz2=1), 1e-06)
            with self.subTest(nd=nd, msg="ArithmeticAveraging linear test"):
                self.assertLess(fun.linear_test(v, ave, dx1=1, dz1=1,
                                                dx2=1, dz2=1), 1e-2)
            with self.subTest(nd=nd, msg="ArithmeticAveraging dot test"):
                self.assertLess(fun.dot_test(v, ave, dx1=1, dz1=1,
                                             dx2=1, dz2=1), 1e-06)

    def test_HarmonicAveraging(self):
        resc = ComputeRessource()
        q = resc.queues[0]
        for nd in [2, 3]:

            shape = (10,) * nd
            v = VariableCL(q, shape=shape, initialize_method="random",
                           pad=2)
            ave = VariableCL(q, shape=shape, initialize_method="random",
                             pad=2)
            fun = HarmonicAveraging(q)
            with self.subTest(nd=nd, msg="HarmonicAveraging backward test"):
                self.assertLess(fun.backward_test(v, ave, dx1=1, dz1=1,
                                                  dx2=1, dz2=1), 1e-06)
            with self.subTest(nd=nd, msg="HarmonicAveraging linear test"):
                self.assertLess(fun.linear_test(v, ave, dx1=1, dz1=1,
                                                dx2=1, dz2=1), 1e-2)
            with self.subTest(nd=nd, msg="HarmonicAveraging dot test"):
                self.assertLess(fun.dot_test(v, ave, dx1=1, dz1=1,
                                             dx2=1, dz2=1), 1e-06)