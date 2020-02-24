# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:53:04 2020

@author: david

"""


import copy
import numpy as np
import matplotlib.pyplot as plt
import emcee as mc

def GelmanRubinTest(m:int, n:int, samples:list):
    '''  
    arguments: m -> number of MCMC chains, n -> length of chain (including burn-in), samples -> accepted MCMC samples
    
    returns: Gelman-Rubin ratio, close to 1 is better ie. < 1.1
    
    Description
    The Gelman-Rubin diagnostic uses an analysis of variance approach to assessing convergence. That is, it calculates both the between-
    chain variance (B) and within-chain variance (W), and assesses whether they are different enough to worry about convergence. Assuming m
    chains, each of length n
    
    reference: http://iacs-courses.seas.harvard.edu/courses/am207/blog/lecture-8.html
    
    '''
    m = m # number of chains
    n = int(0.8*n) # length of chain excluding burn-in samples
    thetas = [np.mean(sample[-n:]) for sample in samples]
    theta_bar_bar = np.mean([np.mean(sample[-n:]) for sample in samples])
    W_i = 0
    
    samples_thetas = list(zip(samples, thetas)) # (sample, theta)
    
    # Between chains variance
    B = n/(m-1)*np.sum([(st[1]-theta_bar_bar)**2 for st in samples_thetas])  # sum sq difference of chain theta and global theta

    # Within chain variance
    W = 1/m*np.sum([1/(n-1)*np.sum((st[0][-n:] - st[1])**2) for st in samples_thetas]) # sum sq difference of indiv theta and chain theta
    
    # posterior variance
    V = (n - 1)/n*W + 1/n*B

    # corrected Gelman-Rubin ratio: with degree freedom scaling to account for sampling variability
    d = m - 1
    
    # corrected Gelman-Rubin ratio: the potential scale reduction factor (PSRF)
    R = np.sqrt((d + 3)/(d + 1)*V/W)

#     print(B, W, V, R)
    return R

def MCMC(shape, scale, sigma=100, niters = np.linspace(3e4,1e5,3), thetas = np.arange(1500, 3501, 500)):
    '''
    arguments: 
    shape, scale -> shape and scale parameters of gamma distribution, sigma -> the width of the random perturbation (too wide, higher 
    rejection; too narrow, too many 'poorly' accepted samples
    niters -> list of number of iterations, thetas -> list of starting points
    
    prints: Trace plot, samples distribution and ACF plot, Gelman-Rubin convergence score
    
    '''
    from scipy.stats import gamma

    niters = niters
    cnt = 1

    fig = plt.figure(figsize=(20,20))

    for i in range(len(niters)):
        print("iter: %s" %niters[i])

        # select a sigma for the algo, 
        # a range too wide gives you lower chances of accepting samples but allows more room for the candidate data to roam
        # a range too narrow gives you higher chance of accepting samples but samples may not converge
        sigma = sigma

        # select different starting points for mu based on its range
        thetas = thetas
        sampless = [mh(niters[i], mu_, sigma, gamma, shape, scale).run() for mu_ in thetas]

        # Samples plot
        ax = fig.add_subplot(3,3,i+cnt)
        for samples in sampless:    
            ax.plot(samples, '--')
        ax.set_xlim([0, niters[i]])
        ax.set_ylabel('Theta')
        ax.set_xlabel('Iteration')
        ax.set_title('Accepted Theta Samples per Iteration')
        cnt += 1

        # Distribution plot
        ax = fig.add_subplot(3,3,i+cnt)
        last_n_samples = int(niters[i] * 0.8)
        for samples in sampless:
            ax.hist(samples[-last_n_samples:], bins=100, histtype='step', label=f'{niters[i]}')
        ax.set_xlim([0, 10000*sigma/100])
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Theta')
        ax.set_title('Distribution of Theta')
        cnt += 1

        # Auto-correlation plot
        ax = fig.add_subplot(3,3,i+cnt)
        for samples in sampless:
            ax.plot(mc.autocorr.function_1d(samples), '--')
        ax.set_xlim([0, niters[i]])
        ax.set_ylabel('ACF')
        ax.set_xlabel('Iteration')
        ax.set_title('Test for Autocorrelation')
        print("Gelman Rubin convergence ratio: %s" %GelmanRubinTest(len(thetas), len(sampless[0]), sampless))

    plt.show()
    return sampless


def numericalMLEGamma(x):
    '''
    Computes the MLE of a Gamma distribution
    argument: x -> list of gamma samples
    
    returns: alpha and beta -> parameters of a gamma distribution
    
    reference: https://tminka.github.io/papers/minka-gamma.pdf, https://github.com/tminka/fastfit/blob/master/gamma_fit.m
    '''
    from scipy.special import digamma, polygamma
    m = np.mean(x)
    s = np.log(m) - np.mean(np.log(x))
    a = 0.5/s
    for i in range(100):
        old_a = a
        g = np.log(a) - s - digamma(a)
        h = 1/a - polygamma(1,a) # first derivative of digamma
        a = 1/(1/a + g/(a**2*h))
        if (abs(a - old_a) < 1e-8):
            break
    b = m/a
    return a, b


class mh:
    '''
    Metropolis Hasting pseudo algorithm can be found at: http://www.mit.edu/~ilkery/papers/MetropolisHastingsSampling.pdf
    
    The Metropolis Hastings method runs for N iterations with different starting points - thetas (plausible values of the mean).
    It completes after N samples are accepted and returns those samples.
    
    This mcmc algorithm's target (proposal) is a gamma distribution and the transition is a normal distribution (random perturbation)
    
    reference: https://people.duke.edu/~ccc14/sta-663/MCMC.html
    
    '''
    def __init__(self, niters: int, theta: int, sigma: float, target, t_param1, t_param2=None):
        self.niters = niters
        self.theta = theta
        self.sigma = sigma
        self.target = target
        self.shape = t_param1 # shape
        self.scale = t_param2 # scale
        self.samples = []
        self.theta_p = None

    def _target_model(self, theta):
        return self.target.pdf(theta, self.shape, scale=self.scale)
    
    def run(self):
        print("Theta: %s" %self.theta)
        while len(self.samples) < self.niters:
            self.theta_p = max(self.theta + np.random.normal(0, self.sigma), 0) # q(x), if negative, assign 0

            T_next = self._target_model(self.theta_p) # π(x_cand) since we are using a symmetric transition probability, we can cancel away the transition term
            T_prev = self._target_model(self.theta)  #  π(x(i−1)) since we are using a symmetric transition probability, we can cancel away the transition term

            a = min(1, T_next/T_prev)
            u = np.random.uniform()
            if u < a:
                self.theta = copy.copy(self.theta_p)
                self.samples.append(self.theta)
           
        print("Mean of samples: %s, Stddev of samples: %s" %(np.sum(self.samples)/len(self.samples), np.std(self.samples)))
        return self.samples