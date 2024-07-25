"""
Script to perform the conversion between the quality factor Q and the tau
variable of the GLS. The class QTAU can be used to perform a least-squares fit
so the the relation between 1/Q and tau are as linear as possible.
"""

import numpy as np
import scipy.stats
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({'font.size': 9})

def alpha2q(alpha, v, f):

    q = np.pi * f / alpha / v
    return q

def q2alpha(q, v, f):

    alpha = np.pi * f / q / v
    return alpha

class QTAU(object):
    """
    Class to perform the conversion between the quality factor Q and the tau
    variable of the GSLS. The class QTAU can be used to perform a least-sqaures fit
    so the the relation between 1/Q and tau are as linear as possible.
    """
    def __init__(self, Qmin, Qmax, fmin, fmax, FL):
        """
        Parameters for the least squares linear fit between Q and tau.

        Qmin (float): Minimum Q value
        Qmax (float): Maximum A value
        fmin (float): Minimum frequency
        fmax (float): Maximum frequency
        FL (list): List of center freqency of each Maxwell body of the GSLS

        """
        self.Qmin = Qmin
        self.Qmax = Qmax
        self.fmin = fmin
        self.fmax = fmax
        self.taus = [1/t/2/np.pi for t in FL]

        tau, Qi, _, _, _ = self.calibrate_Qm1()
        self.interpQ = interp1d(np.log(Qi), np.log(tau))


    def Qprofile(self, tau, taus=None):
        if taus is None:
            taus = self.taus
        fact1 = 0
        fact2 = 0
        omega = 2 * np.pi * np.arange(self.fmin, self.fmax, 0.1)
        for jj in range(len(taus)):
            fact1 = fact1 + omega ** 2 * taus[jj] ** 2 / (
                    1.0 + omega ** 2 * taus[jj] ** 2)
            fact2 = fact2 + omega * taus[jj] / (
                    1.0 + omega ** 2 * taus[jj] ** 2)

        Q = (1 / tau + fact1) / fact2

        return omega, Q

    def Qplot(self, tau, name=None):

        fig, ax = plt.subplots(1, 1, figsize=[14 / 2.54, 9 / 2.54])
        for t in tau:
            omega, Q = self.Qprofile(t)
            ax.plot(omega / (2 * np.pi), 1 / Q, label=str(t))
        plt.xlabel('f (Hz)')
        plt.ylabel('1/Q')
        plt.yscale("log", basey=2)
        plt.legend(title='$\\tau$', loc=1)
        plt.grid()
        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
        if name:
            plt.savefig(name, dpi=300)
        else:
            plt.show()

    def calibrate_Qm1(self, plot=False, taus=None):

        tau_min = 1/self.Qmax/2
        tau_max = 1/self.Qmin*2
        tau = np.linspace(np.log(tau_min), np.log(tau_max), 40)
        if taus is None:
            taus = self.taus

        Q = np.zeros(len(tau))
        for ii in range(len(tau)):
            thisq = self.Qprofile(np.exp(tau[ii]), taus=taus)[-1]
            Q[ii] = np.average(1/thisq)
            Qvar = np.std(1/thisq)

        fit = scipy.stats.linregress(tau, np.log(Q))
        if plot is True:
            #plt.loglog(np.exp(tau), Q, color='black')
            plt.errorbar(np.exp(tau), Q, Qvar,  color='black')
            plt.plot(np.exp(tau), np.exp(fit.slope*tau + fit.intercept), color='red' )
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel('tau')
            plt.ylabel('$Q^{-1}$')
            plt.show()

        return np.exp(tau), Q, Qvar, np.sum(Qvar/Q), fit.slope


    def tau_optimize(self):

        def loss (t):
            tau, Q, Qvar, stds, slope = self.calibrate_Qm1(taus=t)

            return (slope-1.0)**2 + stds

        opt = minimize(loss, self.taus, bounds=[[1e-6, 0.3]]*len(self.taus))

        self.taus = opt.x
        tau, Qi, _, _, _ = self.calibrate_Qm1()
        self.interpQ = interp1d(np.log(Qi), np.log(tau))

    def get_FL(self):
        return [1/t/2/np.pi for t in self.taus]

    def Q2tau(self, Q):

        return np.exp(self.interpQ(np.log(1/Q)))

    def tau2Qi(self, tau, taus=None):

        if taus is None:
            taus = self.taus
        try:
            iter(tau)
        except TypeError:
            tau = [tau]

        Qi = np.zeros_like(tau)
        for ii, t in enumerate(tau):
            Q = self.Qprofile(np.exp(t), taus=taus)[-1]
            Qi[ii] = np.average(1/Q)

        return Qi



if __name__ == "__main__":

    qt = QTAU(7, 200, 15, 80, [5,  30, 60])
    qt.calibrate_Qm1(plot=True)
    print(qt.Q2tau(np.array([7, 20, 50])))
    qt.tau_optimize()
    qt.calibrate_Qm1(plot=True)
    print(qt.get_FL())
    print(qt.Q2tau(np.array([10, 15, 20])))
    qt.Qplot([0.01, 0.02, 0.04, 0.08, 0.16, 0.32],
             name="../report_2019/Figures/Qplot.png")

