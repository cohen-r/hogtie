#! usr/bin/env python

"""
generating null data and testing observations against it
"""

import ipcoal
import toytree
import toyplot
from loguru import logger
import numpy as np
import pandas as pd
from hogtie import DiscreteMarkovModel

class SimulateNull():
    """
    Compare data to expectations

    TO DO: 
    - integrate genealogy-based reordering function before running MatrixParser on null sim
    - why isn't sim_loci working within the function?
    - optimizing binary state model with simulation is giving really high likelihood scores
    """
    def __init__(self,
        tree,
        matrix,
        model=None,
        prior=0.5,
        null=None, #user can select high ILS, slow mutation rate, etc. (need to flesh this idea out more)
        significance_level=0): #integer multuplied by sd to get sig level
        
        if isinstance(tree, toytree.tree):
            self.tree = tree
        elif isinstance(tree, str):
            self.tree = toytree.tree(tree, tree_format=0)
        else: 
            raise Exception('tree must be either a newick string or toytree object')

        self.matrix = matrix
        self.model = model
        self.prior = prior
        self.significance_level = significance_level


        #derived model parameters
        self.treeheight = float(self.tree.treenode.height)

        #parameters to be set
        self.high_lik = 0

        #create an empty dataframe to store values
        self.likelihoods = pd.DataFrame()

        #high ILS, no introgression
        self.mod = ipcoal.Model(tree=self.tree, Ne=1e8)
        logger.info('Initiated model')

    #@property
    def experimental_likelihoods(self):
        """
        doc string
        """
        exp = BinaryStateModel(self.tree, self.matrix, self.model, self.prior)
        exp.optimize()
        logger.info(f'Optimized experimental values, alpha = {exp.alpha} and beta = {exp.beta}')
        
        #save new likelihood df
        log_liks = np.empty((0,len(exp.matrix.columns)),float)
        for i in exp.likelihoods['lik']:
            log_lik = -np.log(i)
            log_liks = np.append(log_liks, log_lik)

        #save to new dataframe    
        self.likelihoods['liks'] = exp.likelihoods['lik']
        self.likelihoods['negloglik'] = log_liks

    def null(self):
        """
        Simulates SNPs across the input tree to create the null expectation for likelihood
        scores and compares
        """
        self.mod.sim_snps(nsnps=10)
        null_genos = self.mod.write_vcf().iloc[:, 9:].T

        #make sure matrix has only 0's and 1's
        for col in null_genos:
            null_genos[col] = null_genos[col].replace([2,3],1)

        #run Binary State model on the matrix and get likelihoods
        null = BinaryStateModel(tree=self.tree, matrix=null_genos, model=self.model)
        null.optimize()

        #get negloglikes for the null values
        log_liks = np.empty((0,len(null.matrix.columns)),float)
        for i in null.likelihoods['lik']:
            log_lik = -np.log(i)
            log_liks = np.append(log_liks, log_lik)

        #get z -scores null likelihood expectations
        null_std = log_liks.std()
        null_mean = log_liks.mean()
        
        #get the likelihood value that corresponds 2 standard deviations above the null mean
        self.high_lik = null_mean + (self.significance_level * null_std)

        #devs = [] #would prefer to append to an empty np.array
        #for like in list(lik_calc.likelihoods[0]):
            #if like >= self.high_lik:
                #devs.append(1)
            #else:
                #devs.append(0)
        #lik_calc.likelihoods['deviation_score'] = np.array(devs)
    
    def genome_graph(self):
        """
        Graphs rolling average of likelihoods along the linear genome, identifies 
        regions that deviate significantly from null expectations

        TO DO: change color of outliers
        """

        self.likelihoods['rollingav']= self.likelihoods['negloglikes'].rolling(50).mean()

        likelihood_density_scores = []
        for i in self.likelihoods.index:
            end = i + 49
            dens = self.likelihoods.iloc[i:end,1]
            likelihood_density_scores.append(dens)
        
        a,b,c = toyplot.plot(
            likelihood_density_scores,
            width = 500,
            height=500,
            color = 'blue',
        )

        b.hlines(self.high_lik, style={"stroke": "red", "stroke-width": 2});

if __name__ == "__main__":
    testtree = toytree.rtree.unittree(ntips=10, treeheight=1e5)
    import os
    HOGTIEDIR = os.path.dirname(os.getcwd())
    file1 = os.path.join(HOGTIEDIR, "sampledata", "testmatrix.csv")
    file2 = os.path.join(HOGTIEDIR, "sampledata", "testtree.txt")    
    test = SimulateNull(tree=file2, model='ARD', matrix=file1)
    test.experimental_likelihoods()
    test.null()
    #print(test.experimental_likelihoods)
    #test.null()
    #test.genome_graph()
