import os, sys, numpy as np
sys.path.insert(0,"/userdata/u/gfabien/claude/SeisCL-dft/SeisCL/tests")
from gradient_common import *
WD="/userdata/u/gfabien/claude/dft-work/wd3"; os.makedirs(WD,exist_ok=True)
s0=make_seiscl(WD); p=homogeneous(s0)
tp={k:v.copy() for k,v in p.items()}; tp["vp"][25:35,25:35]+=300.
din=make_observed(s0,params=tp)
GF=np.array([11.,19.,27.])
for nf in (1,3):
    s=make_seiscl(WD,gradout=1,back_prop_type=2,gradfreqs=GF[:nf]); s.file_din=din
    s.set_forward(s.src_pos_all[3,:],p,withgrad=True); s.execute()
    g=s.read_grad()
    print("NFREQS=%d  max|gradvp|=%.6e  max|gradvs|=%.6e  max|gradrho|=%.6e  allzero=%s"
          % (nf,np.abs(g[0]).max(),np.abs(g[1]).max(),np.abs(g[2]).max(),
             all(np.all(x==0) for x in g)))
    np.save(os.path.join(WD,"cuda_nf%d.npy"%nf), np.array([x for x in g]))
