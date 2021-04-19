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
        self.std = null.likelihoods[0].std()
        self.mean = null.likelihoods[0].mean()


    def significance_test(self):
        """
        identifies k-mer likelihoods that are +2 standard deviations outside of the mean
        from simulated likelihood z-score
        """
        #get the likelihood value that corresponds to the z-score of 3 in simulations
        #high_lik = self.mu + 2*self.sigma

        #get aic values for null
        #aic_list_null = []
        #for lik in self.null_likelihoods:
        #    aic = 2*lik+2*2
        #    aic_list_null.append(aic)

        #aic_list_null = np.array(aic_list_null)
        #aic_mean = aic_list_null.mean()
        #aic_std = aic_list_null.std()

        #print(aic_mean, aic_std)

        lik_calc = MatrixParser(tree=self.tree,
                               model=self.model,
                               prior=self.prior,
                               matrix=self.matrix
                               )

        lik_calc.matrix_likelihoods()

        #likelihood ratio test
        p_values = []
        for lik in lik_calc.likelihoods[0]:
            lik_ratio = 2 * (lik - self.mean)
            p_value = chi2.sf(lik_ratio, df=1) #df set according to Felsenstein 2003 (pg.309)
            p_values.append(p_value)

        p_values = np.array(p_values)
        lik_calc.likelihoods['lrt_p_values'] = p_values
        
        #find deviations --> move deviations to genome_graph?
        devs = [] #would prefer to append to an empty np.array
        for value in lik_calc.likelihoods['lrt_p_values']:
            if value >= 0.05:
                devs.append(0)
            else:
                devs.append(1)

        devs = np.array(devs)
        lik_calc.likelihoods['deviation_score'] = devs

        print(lik_calc.likelihoods)
        self.likehood_df = like_calc.likelihoods

    def genome_graph(self):
        """
        Graphs rolling average of likelihoods along the linear genome, identifies 
        regions that deviate significantly from null expectations

        TO DO: change color of outliers, integrate into Hogtie class instead of MatrixParser
        class
        """

        self.likelihoods['rollingav']= self.likelihoods[1].rolling(10, win_type='triang').mean()
        
        plot = toyplot.plot(
            self.likelihoods['rollingav'],
            width = 500,
            height=500,
            color = 'blue'
        )

        return plot

if __name__ == "__main__":
    testtree = toytree.rtree.unittree(ntips=10)
    mod1 = ipcoal.Model(tree=testtree, Ne=1e3, admixture_edges=[(3, 8, 0.5, 0.5)], nsamples=1)
    mod1.sim_loci(nloci=1, nsites=100000)
    genos1 = mod1.write_vcf()
    data=genos1.iloc[:, 9:].T
    test = Hogtie(tree=testtree, model='ARD', matrix=data)
    test.create_null()
    test.significance_test()