#! usr/bin/env python

"""
generating null data and testing observations against it
"""

import ipcoal
import toytree
import toyplot
from loguru import logger
import numpy as np
from hogtie import MatrixParser

class SimulateNull():
    """
    Compare data to expectations

    TO DO: 
    - integrate genealogy-based reordering function before running MatrixParser on null sim
    - why isn't sim_loci working within the function?
    """
    def __init__(self, tree, matrix, model=None, prior=0.5):
        self.model = model
        self.prior = prior
        self.matrix = matrix

        if isinstance(tree, toytree.tree):
            self.tree = tree
        elif isinstance(tree, str):
            self.tree = toytree.tree(tree, tree_format=0)
        else: 
            raise Exception('tree must be either a newick string or toytree object')

        self.treeheight = float(self.tree.treenode.height)

        #high ILS
        self.mod = ipcoal.Model(tree=self.tree, Ne=(self.treeheight ** 3))

        logger.info('Initiated model')

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
        null = MatrixParser(tree=self.tree, matrix=null_genos, model=self.model)
        null.matrix_likelihoods()

        #get z -scores null likelihood expectations
        null_std = null.likelihoods[0].std()
        null_mean = null.likelihoods[0].mean()
        
        #get the likelihood value that corresponds 2 standard deviations above the null mean
        self.high_lik = null_mean + 2 * null_std

        lik_calc = MatrixParser(tree=self.tree,
                               model=self.model,
                               prior=self.prior,
                               matrix=self.matrix
                               )

        lik_calc.matrix_likelihoods()

        devs = [] #would prefer to append to an empty np.array
        for like in list(lik_calc.likelihoods[0]):
            if like >= self.high_lik:
                devs.append(1)
            else:
                devs.append(0)

        lik_calc.likelihoods['deviation_score'] = np.array(devs)
        
        self.likes = lik_calc.likelihoods
        return self.likes

    def genome_graph(self):
        """
        Graphs rolling average of likelihoods along the linear genome, identifies 
        regions that deviate significantly from null expectations

        TO DO: change color of outliers
        """

        self.likes['rollingav']= self.likes[0].rolling(50, win_type='triang').mean()
        
        a,b,c = toyplot.plot(
            self.likes['rollingav'],
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
    test = SimulateNull(tree=testtree, model='ARD', matrix=file1)
    test.null()
    #test.genome_graph()
