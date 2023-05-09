
from SeisCL.python import (FunctionGPU, ComputeRessource, ComputeGrid)
import numpy as np


class ArithmeticAveraging(FunctionGPU):

    def define_postr(self, v):
        nd = len(v.shape)
        if nd == 3:
            posstr1 = ", ".join(["g."+el for el in "zyx"])
            posstr2 = ", ".join(["g."+el+" + d"+el for el in "zyx"])
        elif nd == 2:
            posstr1 = ", ".join(["g."+el for el in "zx"])
            posstr2 = ", ".join(["g."+el+" + d"+el for el in "zx"])
        elif nd == 1:
            posstr1 = ", ".join(["g."+el for el in "z"])
            posstr2 = ", ".join(["g."+el+" + d"+el for el in "z"])
        return posstr1, posstr2

    def forward(self, v, vave, dx=0, dy=0, dz=0):

        posstr1, posstr2 = self.define_postr(v)
        src = """
        vave(%s)=0.5*(v(%s)+v(%s));
        """ % (posstr1, posstr1, posstr2)
        grid = ComputeGrid(shape=[s - 2*v.pad for s in v.shape],
                           queue=self.queue,
                           origin=[v.pad for _ in v.shape])
        self.callgpu(src, "forward", grid, v, vave, dx, dy, dz)
        return vave

    def linear(self, v, vave, dx=0, dy=0, dz=0):

        posstr1, posstr2 = self.define_postr(v)
        src = """
        vave_lin(%s)=0.5*(v_lin(%s)+v_lin(%s));
        """ % (posstr1, posstr1, posstr2)
        grid = ComputeGrid(shape=[s - 2*v.pad for s in v.shape],
                           queue=self.queue,
                           origin=[v.pad for _ in v.shape])
        self.callgpu(src, "linear", grid, v, vave, dx, dy, dz)
        return vave

    def adjoint(self, v, vave, dx=0, dy=0, dz=0):

            posstr1, posstr2 = self.define_postr(v)
            src = """
            v_adj(%s)+=0.5*vave_adj(%s);
            v_adj(%s)+=0.5*vave_adj(%s);
            vave_adj(%s)=0;
            """ % (posstr1, posstr1, posstr2, posstr1, posstr1)
            grid = ComputeGrid(shape=[s - 2*v.pad for s in v.shape],
                               queue=self.queue,
                               origin=[v.pad for _ in v.shape])
            self.callgpu(src, "adjoint", grid, v, vave, dx, dy, dz)
            return v, vave


class ArithmeticAveraging2(FunctionGPU):

    def posstr(self, v):
        nd = len(v.shape)
        if nd == 3:
            posstr1 = ", ".join(["g."+el for el in "zyx"])
            posstr2 = ", ".join(["g."+el+" + d"+el+"1" for el in "zyx"])
            posstr3 = ", ".join(["g."+el+" + d"+el+"2" for el in "zyx"])
            posstr4 = ", ".join(["g."+el+" + d"+el+"1" + " + d"+el+"2" for el
                                 in "zyx"])
        elif nd == 2:
            posstr1 = ", ".join(["g."+el for el in "zx"])
            posstr2 = ", ".join(["g."+el+" + d"+el+"1" for el in "zx"])
            posstr3 = ", ".join(["g."+el+" + d"+el+"2" for el in "zx"])
            posstr4 = ", ".join(["g."+el+" + d"+el+"1" + " + d"+el+"2"
                                 for el in "zx"])
        else:
            raise ValueError("Number of dimension must be 2 or 3")
        return posstr1, posstr2, posstr3, posstr4

    def forward(self, v, vave, dx1=0, dy1=0, dz1=0, dx2=0, dy2=0, dz2=0):

        posstr1, posstr2, posstr3, posstr4 = self.posstr(v)
        src = """
        vave(%s)=0.25*(v(%s)+v(%s)+v(%s)+v(%s));
        """ % (posstr1, posstr1, posstr2, posstr3, posstr4)
        grid = ComputeGrid(shape=[s - 2*v.pad for s in v.shape],
                        queue=self.queue,
                        origin=[v.pad for _ in v.shape])
        self.callgpu(src, "forward", grid, v, vave, dx1, dy1, dz1, dx2, dy2, dz2)
        return vave

    def linear(self, v, vave, dx1=0, dy1=0, dz1=0, dx2=0, dy2=0, dz2=0):

            posstr1, posstr2, posstr3, posstr4 = self.posstr(v)
            src = """
            vave_lin(%s)=0.25*(v_lin(%s)+v_lin(%s)+v_lin(%s)+v_lin(%s));
            """ % (posstr1, posstr1, posstr2, posstr3, posstr4)
            grid = ComputeGrid(shape=[s - 2*v.pad for s in v.shape],
                               queue=self.queue,
                               origin=[v.pad for _ in v.shape])
            self.callgpu(src, "linear", grid, v, vave, dx1, dy1, dz1, dx2, dy2, dz2)
            return vave

    def adjoint(self, v, vave, dx1=0, dy1=0, dz1=0, dx2=0, dy2=0, dz2=0):

        posstr1, posstr2, posstr3, posstr4 = self.posstr(v)
        src = """
        v_adj(%s)+=0.25*vave_adj(%s);
        v_adj(%s)+=0.25*vave_adj(%s);
        v_adj(%s)+=0.25*vave_adj(%s);
        v_adj(%s)+=0.25*vave_adj(%s);
        vave_adj(%s)=0;
        """ % (posstr1, posstr1, posstr2, posstr1, posstr3, posstr1, posstr4, posstr1, posstr1)
        grid = ComputeGrid(shape=[s - 2*v.pad for s in v.shape],
                           queue=self.queue,
                           origin=[v.pad for _ in v.shape])
        self.callgpu(src, "adjoint", grid, v, vave, dx1, dy1, dz1, dx2, dy2, dz2)
        return v


class HarmonicAveraging(FunctionGPU):

    def posstr(self, v):
        nd = len(v.shape)
        if nd == 3:
            posstr1 = ", ".join(["g."+el for el in "zyx"])
            posstr2 = ", ".join(["g."+el+" + d"+el+"1" for el in "zyx"])
            posstr3 = ", ".join(["g."+el+" + d"+el+"2" for el in "zyx"])
            posstr4 = ", ".join(["g."+el+" + d"+el+"1" + " + d"+el+"2" for el
                                 in "zyx"])
        elif nd == 2:
            posstr1 = ", ".join(["g."+el for el in "zx"])
            posstr2 = ", ".join(["g."+el+" + d"+el+"1" for el in "zx"])
            posstr3 = ", ".join(["g."+el+" + d"+el+"2" for el in "zx"])
            posstr4 = ", ".join(["g."+el+" + d"+el+"1" + " + d"+el+"2"
                                 for el in "zx"])
        else:
            raise ValueError("Number of dimension must be 2 or 3")
        return posstr1, posstr2, posstr3, posstr4

    def forward(self, v, vave, dx1=0, dy1=0, dz1=0, dx2=0, dy2=0, dz2=0):

        posstr1, posstr2, posstr3, posstr4 = self.posstr(v)
        eps = v.smallest
        src = """
        float d =0;
        float n = 0;
        if (v(%s) > %e){
            n+=1;
            d += 1.0 / v(%s);
        }
        if (v(%s) > %e){
            n+=0;
            d += 1.0 / v(%s);
        }
        if (v(%s) > %e){
            n+=1;
            d += 1.0 / v(%s);
        }
        if (v(%s) > %e){
            n+=1;
            d += 1.0 / v(%s);
        }
        vave(%s)= n / d;
        """ % (posstr1, eps, posstr1,
               posstr2, eps, posstr2,
               posstr3, eps, posstr3,
               posstr4, eps, posstr4,
               posstr1)
        grid = ComputeGrid(shape=[s - 2*v.pad for s in v.shape],
                           queue=self.queue,
                           origin=[v.pad for _ in v.shape])
        self.callgpu(src, "forward", grid, v, vave, dx1, dy1, dz1, dx2, dy2,
                     dz2)
        return vave

    def linear(self, v, vave, dx1=0, dy1=0, dz1=0, dx2=0, dy2=0, dz2=0):

        posstr1, posstr2, posstr3, posstr4 = self.posstr(v)
        eps = v.smallest
        src = """
            float d1 =0;
            float d2 =0;
            float n = 0;
            if (v(%s) > %e){
                n+=1;
                d1 += 1.0 / v(%s);
                d2 += 1.0/ pow(v(%s),2) * v_lin(%s);
            }
            if (v(%s) > %e){
                n+=1;
                d1 += 1.0 / v(%s);
                d2 += 1.0/ pow(v(%s),2) * v_lin(%s);
            }
            if (v(%s) > %e){
                n+=1;
                d1 += 1.0 / v(%s);
                d2 += 1.0/ pow(v(%s),2) * v_lin(%s);
            }
            if (v(%s) > %e){
                n+=1;
                d1 += 1.0 / v(%s);
                d2 += 1.0/ pow(v(%s),2) * v_lin(%s);
            }
            vave_lin(%s)= n / pow(d1, 2) * d2;
            """ % (posstr1, eps, posstr1, posstr1, posstr1,
                   posstr2, eps, posstr2, posstr2, posstr2,
                   posstr3, eps, posstr3, posstr3, posstr3,
                   posstr4, eps, posstr4, posstr4, posstr4,
                   posstr1)
        grid = ComputeGrid(shape=[s - 2*v.pad for s in v.shape],
                           queue=self.queue,
                           origin=[v.pad for _ in v.shape])
        self.callgpu(src, "linear", grid, v, vave, dx1, dy1, dz1, dx2, dy2,
                     dz2)
        return vave

    def adjoint(self, v, vave, dx1=0, dy1=0, dz1=0, dx2=0, dy2=0, dz2=0):

            posstr1, posstr2, posstr3, posstr4 = self.posstr(v)
            eps = v.smallest
            src = """
            float d1 = 0;
            float n = 0;
            if (v(%s) > %e){
                n+=1;
                d1 += 1.0 / v(%s);
            }
            if (v(%s) > %e){
                n+=1;
                d1 += 1.0 / v(%s);
            }
            if (v(%s) > %e){
                n+=1;
                d1 += 1.0 / v(%s);
            }
            if (v(%s) > %e){
                n+=1;
                d1 += 1.0 / v(%s);
            }
            
            if (v(%s) > %e){
                v_adj(%s) += n / pow(d1, 2) * 1.0/ pow(v(%s),2) * vave_adj(%s);
            }
            if (v(%s) > %e){
                v_adj(%s) += n / pow(d1, 2) * 1.0/ pow(v(%s),2) * vave_adj(%s);
            }
            if (v(%s) > %e){
                v_adj(%s) += n / pow(d1, 2) * 1.0/ pow(v(%s),2) * vave_adj(%s);
            }
            if (v(%s) > %e){
                v_adj(%s) += n / pow(d1, 2) * 1.0/ pow(v(%s),2) * vave_adj(%s);
            }
            vave_adj(%s) = 0;
                """ % (posstr1, eps, posstr1,
                       posstr2, eps, posstr2,
                       posstr3, eps, posstr3,
                       posstr4, eps, posstr4,
                       posstr1, eps, posstr1, posstr1, posstr1,
                       posstr2, eps, posstr2, posstr2, posstr1,
                       posstr3, eps, posstr3, posstr3, posstr1,
                       posstr4, eps, posstr4, posstr4, posstr1,
                       posstr1)
            grid = ComputeGrid(shape=[s - 2*v.pad for s in v.shape],
                            queue=self.queue,
                            origin=[v.pad for _ in v.shape])
            self.callgpu(src, "adjoint", grid, v, vave, dx1, dy1, dz1, dx2, dy2,
                        dz2)
            return vave



