# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:53:04 2020

@author: david

# adapted from https://people.duke.edu/~ccc14/sta-663/MCMC.html
"""


import copy
import numpy as np


class mh:
    '''
    Metropolis Hasting pseudo algorithm can be found at: http://www.mit.edu/~ilkery/papers/MetropolisHastingsSampling.pdf
    
    The Metropolis Hastings method runs for N iterations with a starting data point theta.
    It completes after N accepted samples are drawn and returns those samples.
    
    this mcmc algorithm's target is a gamma distribution
    and the transition is a normal distribution
    
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
#         if self.t_param1 is not None and self.t_param1 is not None: # two parameter model, shape and scale
#             return self.target.pdf(theta, self.t_param1, self.t_param2)
#         elif self.t_param2 is None: # one parameter model, shape only
#             return self.target.pdf(theta, self.t_param1) 
    
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