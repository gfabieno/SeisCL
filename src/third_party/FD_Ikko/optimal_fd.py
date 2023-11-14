'''
Copyright (c) 2022 Ikko Health LTD
See LICENSE file.
@author Albert Shalumov
'''
import matplotlib.pyplot as plt
try:
    import torch
    have_torch = True
except ImportError:
    have_torch = False
import numpy as np
from scipy.optimize._basinhopping import basinhopping
from scipy.optimize._differentialevolution import differential_evolution
from scipy.optimize._shgo import shgo
from scipy.optimize._dual_annealing import dual_annealing
import scipy

# This version doesn't enforce decaying oscillation conditions (i.e. a_i*a_(i-1)<0 and |a_i|>|a_(i-1)|)
skip_optimize=False # use saved weight
order=12 # FD order
crit_freq_err = 1e-3 # 0.1% error
force_a1=True # forces a1 such that disparity=1 as kh->0
mix_coeff = 0  # mix of disparity and frequency respone (0 - optimize disparity only)
constraint_reg_coeff = 1e6 # Penatly of decaying oscillation constraint (a_i*a_(i-1)<0 and |a_i|>|a_(i-1)|) (0 - disabled)
weight_reg_coeff = 0 # total weight regularization (0 - disabled)
# Different optimizers
optimizers = ['basin_hopping',
              'diff_evol',
              'shgo',
              'dual_annealing']
optimizer = 'diff_evol'
use_jac=False
# Discretization for optimization
N_samples = 2 ** 10
kx_const = np.linspace(1e-9, np.pi, N_samples, endpoint=True)
best_func=np.inf
min_val=np.inf
# Costs for samples outside threshold
# Format: (end of segment, symmetric threshold around 1, cost for fraction outside)
#    OR   (end of segment, (thrsh. below 1, thrsh. above 1), cost for fraction outside)
ranges = [(0, 0, 0),
          (np.pi/3-1, 5e-5, 1e7),
          (np.pi/3+1, 1e-4, 1e5),
          (2.3, 2e-4, 1e3),
          (2.6, 5e-4, 1e1),
          (np.pi, 9e-4, 1e0)]

# Inclusive iteration from a to b
def inc_range(a,b):
    for i in range(a,b+1):
        yield i

# Disparity for staggered grid finite difference
# delta = Sum_i=1^M(2coeff[i]*sin((2i-1)/2)/kh
# coeff - FD coefficients
# kh - given discretization of kh (0,pi] or None
# M - is half of FD order (eg FD12->M=6)
# N - number of samples in (0,pi] range (in case kh is None)
# Returns kh discretization, disparity
def CalcDispLin(coeff, kh=None, M=3, N=10000):
    if kh is None:
        kh = np.linspace(1e-7,np.pi,N)
    approx=0
    for m in inc_range(1,M):
        approx+=2*coeff[m-1]*np.sin(kh*(2*m-1)/2)
    delta = approx/kh
    return kh,delta

# Calculate frequency response error - use ideal derivative as reference
# coeff - FD coefficients
# kh - given discretization of kh (0,pi] or None
# M - is half of FD order (eg FD12->M=6)
# N - number of samples in (0,pi] range (in case kh is None)
# Returns absolute difference from ideal derivative
def CalcDervDevFFTErr(coeff, kh=None, M=3, N=10000):
    if kh is None:
        kh = np.linspace(1e-7,np.pi,N)
    fd = np.zeros(kh.shape[0]*2-2)
    fd[fd.shape[0]//2-coeff.shape[0]-1:fd.shape[0]//2+coeff.shape[0]-1]=np.concatenate([-np.flip(coeff),coeff])
    fd_fft_abs=np.abs(scipy.fft.rfft(fd))
    #ratio = fd_fft_abs/kh
    delta = np.abs(fd_fft_abs-kh)
    return kh,delta

# Return string ready for pasting in C++
def CoeffCppFormat(coeff):
    order = len(coeff) * 2
    s = 'constexpr float '
    for i in range(order // 2):
        s += f"fd{order}_c{i} = {coeff[i]}f, "
    s = s[:-2] + ';'
    return s

# Evaluation function - calculates fitness of a solution
def eval(*args):
    global min_val
    weight = np.array(args).flatten()
    # If using force_a1 then add a1 from limit calculation
    if force_a1:
        tmp=0
        for j in inc_range(2,order//2):
            tmp += weight[j-2]*(2*j-1)
        weight = np.append([1-tmp], weight)
    kx = kx_const
    _,delta = CalcDispLin(weight, kx, order//2, N=N_samples)
    _, delta_freq = CalcDervDevFFTErr(weight, kx, order // 2, N=N_samples)
    # Calculate and accumulate cost for all the defined ranges
    ret = 0
    for ind in range(1,len(ranges)):
        kx_from = ranges[ind-1][0]
        kx_to, thresh, penalty = ranges[ind]
        mask = (kx >= kx_from) * (kx < kx_to)
        if type(thresh) == tuple:
            thresh_below,thresh_above = thresh
            cost_below = (((1-delta) * mask).clip(0,np.inf) > thresh_below).sum() / (mask.sum()) * penalty
            cost_above = (((delta-1) * mask).clip(0,np.inf) > thresh_above).sum() / (mask.sum()) * penalty
            cost_disparity = cost_below+cost_above
            max_thresh = max(thresh_below,thresh_above)
            cost_freq = ((delta_freq * mask) > max_thresh).sum() / (mask.sum()) * penalty
        else:
            cost_disparity = ((np.abs(1 - delta) * mask) > thresh).sum() / (mask.sum()) * penalty
            cost_freq = ((delta_freq * mask) > thresh).sum() / (mask.sum()) * penalty
        ret += cost_disparity*(1-mix_coeff)+ mix_coeff*cost_freq

    # Oscillation constraints
    constraint_1 = 0
    constraint_2 = 0
    for i in range(order//2-1):
        constraint_1 += float((weight[i]* weight[i+1])>0)
        constraint_2 += float((np.abs(weight[i+1]) - np.abs(weight[i]))>0)
    ret += (constraint_2+constraint_1) * constraint_reg_coeff

    # Weight regularization
    ret += np.sum(weight ** 2) * weight_reg_coeff

    if ret<min_val:
        min_val = min(min_val, ret)
        print(min_val)
        print(repr(weight))
    return ret

if __name__=='__main__':
    if not skip_optimize:
        # if forcing a1 we need one less coefficient
        ind_shift = -1 if force_a1 else 0
        weight0 = np.random.random(order//2+ind_shift)
        # Fill ranges for coefficients
        start_val = 1.4
        bounds = []
        for i in range(order // 2):
            if i % 2 == 0:
                bounds.append((0 if i>0 else 1, start_val / 2 ** i))
            else:
                bounds.append((-start_val / 2 ** i, 0))
        if force_a1:
            bounds = bounds[1:]

        if optimizer=='basin_hopping':
            minimizer_kwargs = {"method":"L-BFGS-B", "jac":use_jac,"bounds":bounds}
            ret = basinhopping(eval, weight0, minimizer_kwargs=minimizer_kwargs, niter = int(5e3),disp=True,stepsize=0.2)
        elif optimizer=='diff_evol':
            # Use better population initialzation
            Np=150
            if have_torch:
                engine = torch.quasirandom.SobolEngine(order//2+ind_shift, scramble=True)
                init_pop = engine.draw(Np).numpy()
                for dim in range(order//2+ind_shift):
                    init_pop[:,dim] = init_pop[:,dim]*(bounds[dim][1]-bounds[dim][0])+bounds[dim][0]
                ret = differential_evolution(eval, bounds,maxiter=int(1e7),popsize=Np, tol=1e-7,disp=True, workers=20, strategy='best2bin', mutation=(0.5,1.99), recombination=0.99, init=init_pop, polish=True)
            else:
                ret = differential_evolution(eval, bounds,maxiter=int(1e7),popsize=Np, tol=1e-7,disp=True, workers=20, strategy='best2bin', polish=True)
        elif optimizer=='shgo':
            ret = shgo(eval, bounds,n=256, iters=10)
        elif optimizer=='dual_annealing':
            ret = dual_annealing(eval, bounds, x0=weight0, maxiter=int(3e4), no_local_search=True)
        weight = np.array(ret.x).flatten()
        if force_a1:
            tmp = 0
            for j in inc_range(2, order // 2):
                tmp += weight[j - 2] * (2 * j - 1)
            weight = np.append([1 - tmp], weight)
        weight.tofile('opt_fd.npy')
    # Load FD weights
    weights = np.fromfile('opt_fd.npy')
    # Print gamma and weights in C++ format
    print('------------------------------------------------')
    print(repr(weights))
    print('------------------------------------------------')
    print('Sum of abs:',np.abs(weights).sum())
    print(CoeffCppFormat(weights))
    # Plots
    N = int(1e6)
    kh,delta_fd6 = CalcDispLin([75 / 64, -25 / 384, 3 / 640], M=3, N=N)
    _,delta_holberg = CalcDispLin([1.1965, -0.078804, 0.0081781], M=3, N=N)
    _,delta_fd8 = CalcDispLin([1225. / 1024., -245. / 3072., 49. / 5120., -5. / 7168.], M=4, N=N)
    _, delta_fd12 = CalcDispLin([160083. / 131072., -12705. / 131072.,  22869. / 1310720.,	-5445. / 1835008.,  847. / 2359296.,  -63. / 2883584.], M=6, N=N)
    _,opt_w = CalcDispLin(weights, None, M=order//2, N=N)
    _, delta_fd12_fft = CalcDervDevFFTErr(np.array([160083. / 131072., -12705. / 131072., 22869. / 1310720., -5445. / 1835008., 847. / 2359296., -63. / 2883584.]), M=6, N=N)
    _, opt_w_fft = CalcDervDevFFTErr(weights, None, M=order // 2, N=N)

    fig = plt.gcf()
    plt.close(fig)

    plt.figure()
    plt.hlines(1,0,3,colors='k',linestyles='--')
    #plt.hlines([1+1e-4,1-1e-4],0,3,colors='k',linestyles='-.')
    #plt.hlines([1+1e-3,1-1e-3], 0, 3, colors='m', linestyles='-.')
    #plt.hlines([1-1e-2,1+1e-2], 0, 3, colors='g', linestyles='-.')
    #plt.hlines(1e-1, 0, 3, colors='y', linestyles='-.')
    plt.vlines(np.pi / 3, 1.05, 0.93, colors='k', linestyles='--')
    plt.plot(kh,delta_fd6,'r',label='$FD_6$')
    plt.plot(kh,delta_holberg,'r--',label='$Holberg_6$')
    plt.plot(kh,delta_fd8,'g',label='$FD_8$')
    plt.plot(kh, delta_fd12, 'm', label='$FD_{12}$')
    #plt.plot(kh,opt_w8,'b', label='Optimal weights')
    plt.plot(kh,opt_w,'b', label=f'Optimal weights$_{{{order}}}$')
    plt.ylabel('Ratio')
    plt.xlabel('kh')
    plt.legend(loc='upper right')
    plt.title(f'Dispersion $\delta$, $\\theta=0$')
    #plt.savefig('res.png')

    plt.figure()
    plt.hlines(0,0,3,colors='k',linestyles='--')
    plt.hlines(1e-4,0,3,colors='k',linestyles='-.')
    #plt.hlines(1e-3,0,3,colors='m',linestyles='-.')
    #plt.hlines(1e-2,0,3,colors='g',linestyles='-.')
    #plt.hlines(1e-1,0,3,colors='y',linestyles='-.')
    plt.vlines(np.pi / 3, 0, 0.1, colors='k', linestyles='--')
    plt.vlines(2*np.pi / 3, 0, 0.1, colors='k', linestyles='--')
    plt.plot(kh,np.abs(1-delta_fd6),'r',label='$FD_6$')
    plt.plot(kh,np.abs(1-delta_holberg),'r--',label='$Holberg_6$')
    plt.plot(kh,np.abs(1-delta_fd8),'g',label='$FD_8$')
    plt.plot(kh, np.abs(1 - delta_fd12), 'm', label='$FD_{12}$')
    plt.plot(kh,np.abs(1-opt_w),'b', label=f'Optimal weights$_{{{order}}}$')
    plt.ylabel('|1-Ratio|')
    plt.xlabel('kh')
    plt.legend(loc='upper right')
    plt.title(f'Dispersion $\delta$, $\\theta={{{0}}}$')
    #plt.savefig('res.png')

    plt.figure()
    plt.plot(kh,delta_fd12_fft,'r',label='$FD_12$')
    plt.plot(kh,opt_w_fft,'b',label='$Optimal$ ' + str(order))
    #plt.plot(kh,kh, 'k', label='Derivative')
    plt.ylabel('Ratio')
    plt.xlabel('kh')
    plt.legend(loc='upper right')
    plt.title(f'Ratio in freq. domain $\\theta=0$')

    # Points Per Wavenumber
    print('Points Per Wavenumber')
    print('------------------------------------------------')
    print('FD6:',2 * np.pi / kh[np.where((np.abs(1 - delta_fd6) < crit_freq_err) == False)[0][0]])
    print('FD8:', 2 * np.pi / kh[np.where((np.abs(1 - delta_fd8) < crit_freq_err) == False)[0][0]])
    print('Holberg6:', 2 * np.pi / kh[np.where((np.abs(1 - delta_holberg) < crit_freq_err) == False)[0][0]])
    print('FD12:', 2 * np.pi / kh[np.where((np.abs(1 - delta_fd12) < crit_freq_err) == False)[0][0]])
    if (np.abs(1 - opt_w) < crit_freq_err)[-1]==True:
        print(f'Optimized{order}:',2)
    else:
        print(f'Optimized{order}:', 2 * np.pi / kh[np.where((np.abs(1 - opt_w) < crit_freq_err) == False)[0][0]])

    plt.show()