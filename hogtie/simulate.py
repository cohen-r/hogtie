#! usr/bin/env python

"""
generating data using ipcoal
"""

import ipcoal
import toytree
import pandas as pd
import numpy as np
from scipy.stats import zscore
from scipy.stats import chi2
from hogtie import MatrixParser

class Hogtie:
    """
    Compare data to expectations

    TO DO: Not sure standard deviation-based comparison is the best for
    log-likelihoods from a statistical point of view. Maybe something more like
    an AIC-type comparison or a likelihood-ratio test?
    """
    def __init__(self, tree, matrix, model=None, prior=0.5):
        self.tree = tree
        self.model = model
        self.prior = prior
        self.matrix = matrix

    def create_null(self):
        """
        Simulates SNPs across the input tree to create the null expectation for likelihood scores. 
        Deviation from the null will be flagged. 
        """

        #high ILS
        mod = ipcoal.Model(tree=self.tree)
        mod.sim_loci(nloci=1, nsites=100000)
        genos = mod.write_vcf()
        
        null_genos = genos.iloc[:, 9:].T

        #run the Binary State model on the matrix and get likelihoods
        null = MatrixParser(tree=self.tree, matrix=null_genos, model=self.model)
        null.matrix_likelihoods()
        self.null_likelihoods = null.likelihoods

        #get mean and sd of null likelihood expectations
        sigma = null.likelihoods.std()
        mu = null.likelihoods.mean()
        self.sigma = sigma
        self.mu = mu


    def significance_test(self):
        """
        identifies k-mer likelihoods that are +2 standard deviations outside of the mean
        from simulated likelihood z-score
        """
        #get the likelihood value that corresponds to the z-score of 3 in simulations
        #high_lik = self.mu + 2*self.sigma

        #get aic values for null
        aic_list_null = []
        for lik in self.null_likelihoods:
            aic = 2*lik+2*2
            aic_list_null.append(aic)

        aic_list_null = np.array(aic_list_null)
        aic_mean = aic_list_null.mean()
        aic_std = aic_list_null.std()

        print(aic_mean, aic_std)

        lik_calc = MatrixParser(tree=self.tree,
                               model=self.model,
                               prior=self.prior,
                               matrix=self.matrix
                               )

        lik_calc.matrix_likelihoods()

        #aic calculation
        aic_list = []
        for lik in lik_calc.likelihoods:
            aic = 2*lik+2*2
            aic_list.append(aic)

        mean_aic = 2*self.mu+2*2

        print(mean_aic)

        aic_sd = 2*self.sigma+2*2

        print(aic_sd)

        devs = [] #would prefer to append to an empty np.array
        for aic in aic_list:
            if aic >= (mean_aic+3):
                devs.append(1)
            else:
                devs.append(0)
        
        #self.deviations = devs

        #likelihood ratio test
        #p_values = []
        #or lik in lik_calc.likelihoods:
        #    G = 2 * (lik - self.mu)
        #    p_value = chi2.sf(G, 1) #what is the df in my case??
        #    p_values.append(p_value)

        #print(p_values)
        
        #I don't think we should expect much deviation among log-likelihoods
        #a better signficance test is probably a likelihood-ratio test
        #devs = [] #would prefer to append to an empty np.array
        #for like in lik_calc.likelihoods:
        #    if like >= high_lik:
        #        devs.append(1)
        #    else:
        #        devs.append(0)

        #self.deviations = devs

    # def genome_graph(self):
        """
        Graphs rolling average of likelihoods along the linear genome, identifies 
        regions that deviate significantly from null expectations

        TO DO: change color of outliers, integrate into Hogtie class instead of MatrixParser
        class
        """

        #self.likelihoods['rollingav']= self.likelihoods.rolling(10, win_type='triang').mean()
        
        #plot = toyplot.plot(
        #    self.likelihoods['rollingav'],
        #    width = 500,
        #    height=500,
        #    color = 'blue'
        #)

        #return plot

if __name__ == "__main__":
    testtree = toytree.rtree.unittree(ntips=10)
    mod1 = ipcoal.Model(tree=testtree, Ne=1e3, admixture_edges=[(3, 8, 0.5, 0.5)], nsamples=1)
    mod1.sim_loci(nloci=1, nsites=100000)
    genos1 = mod1.write_vcf()
    data=genos1.iloc[:, 9:].T
    test = Hogtie(tree=testtree, model='ARD', matrix=data)
    test.create_null()
    test.significance_test()