#! usr/bin/env python

"""
generating null data and testing observations against it
"""

import ipcoal
import toytree
import toyplot
from scipy.linalg import expm
from loguru import logger
import numpy as np
import pandas as pd
from hogtie import DiscreteMarkovModel

class NullDistribution:
    """
    Calculates null likelihoods using optimized alpha and beta values

    Now: uses ARD, will implement ER once functional
    """
    def __init__(self, tree, alpha, beta, prior=0.5):
        # store user inputs
        self.tree = tree
        #self.model = model
        self.prior_root_is_1 = prior
        self.alpha = alpha
        self.beta = beta
        
        # set likelihoods to 1 for data at tips, and None for internal
        #self.set_initial_likelihoods()
        
        #set qmat, assuming all rates different model
        self.qmat = np.array([
                [-alpha, alpha],
                [beta, -beta]
               ])

        self.likelihoods = np.empty((0,tree.ntips),float)

        #initiate model and simulate SNPs using the input species tree
        self.mod = ipcoal.Model(tree=self.tree, Ne=1e6)
        self.mod.sim_snps(10)
        logger.info('Initiated model, simulated SNPs')
        
    def set_initial_likelihoods(self, data):
        """
        Sets the observed states at the tips as attributes of the nodes.
        """
        # get values as lists of [0, 1] or [1, 0]
        values = ([float(1 - i), float(i)] for i in data)
        # get range of tip idxs (0-ntips)
        keys = range(0, len(data))
        # map values to tips {0:x, 1:y, 2:z...}
        valuesdict = dict(zip(keys, values))
        # set as .likelihood attributes on tip nodes.
        self.tree = self.tree.set_node_values(
            feature="likelihood", 
            values=valuesdict,
            default=None,
        )
        
    def node_conditional_likelihood(self, nidx):
        """
        Returns the conditional likelihood at a single node given the
        likelihood's of data at its child nodes.
        """
        # get the TreeNode 
        node = self.tree.idx_dict[nidx]
        # get transition probabilities over each branch length
        prob_child0 = expm(self.qmat * node.children[0].dist)
        prob_child1 = expm(self.qmat * node.children[1].dist)
        # likelihood that child 0 observation occurs if anc==0
        child0_is0 = (
            prob_child0[0, 0] * node.children[0].likelihood[0] + 
            prob_child0[0, 1] * node.children[0].likelihood[1]
        )
        # likelihood that child 1 observation occurs if anc==0
        child1_is0 = (
            prob_child1[0, 0] * node.children[1].likelihood[0] + 
            prob_child1[0, 1] * node.children[1].likelihood[1]
        )
        anc_lik_0 = child0_is0 * child1_is0
        # likelihood that child 0 observation occurs if anc==1
        child0_is1 = (
            prob_child0[1, 0] * node.children[0].likelihood[0] + 
            prob_child0[1, 1] * node.children[0].likelihood[1]
        )
        child1_is1 = (
            prob_child1[1, 0] * node.children[1].likelihood[0] + 
            prob_child1[1, 1] * node.children[1].likelihood[1]
        )
        anc_lik_1 = child0_is1 * child1_is1
        # set estimated conditional likelihood on this node
        node.likelihood = [anc_lik_0, anc_lik_1]
        
    def pruning_algorithm(self):
        """
        Traverse tree from tips to root calculating conditional 
        likelihood at each internal node on the way, and compute final
        conditional likelihood at root based on priors for root state.
        """
        # traverse tree to get conditional likelihood estimate at root.
        for node in self.tree.treenode.traverse("postorder"):
            if not node.is_leaf():
                self.node_conditional_likelihood(node.idx)
        # multiply root prior times the conditional likelihood at root
        root = self.tree.treenode
        lik = (
            (1 - self.prior_root_is_1) * root.likelihood[0] + 
            self.prior_root_is_1 * root.likelihood[1]
        )
        return -np.log(lik)

    def get_likelihoods(self):
        """
        mod must be an ipcoal model object
        objective: make a unique dataframe for each genealogy with the sites that follow
        that genealogy
        """
        vcf = self.mod.write_vcf().iloc[:,9:].T
        
        tree_list = []
        for idx in self.mod.df.index:
            genealogy = toytree.tree(self.mod.df.iloc[idx, 6], tree_format=0)
            tree_list.append(genealogy)
            
        for col in vcf.columns:
            data = vcf[col].reindex(tree_list[col].get_tip_labels())
            self.set_initial_likelihoods(data)

            log_lik = self.pruning_algorithm()
            self.likelihoods = np.append(self.likelihoods, log_lik)

    def get_dist(self):
        """
        """
        #get z -scores null likelihood expectations
        null_std = self.likelihoods.std()
        null_mean = self.likelihoods.mean()
        
        #get the likelihood value that corresponds 2 standard deviations above the null mean
        self.high_lik = null_mean + (self.significance_level * null_std)

    
    def genome_graph(self):
        """
        Graphs rolling average of likelihoods along the linear genome, identifies 
        regions that deviate significantly from null expectations

        TO DO: fix
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
    #get 'experimental' data and run the discretemarkovmodel to get alpha&beta
    TREE = toytree.rtree.baltree(ntips=12, treeheight=1e6)
    MODEL = ipcoal.Model(TREE, Ne=20000, mut=1e-8, seed=123)
    MODEL.sim_snps(10)
    DATA = MODEL.write_vcf().iloc[:, 9:]
    DATA[(DATA == 2) | (DATA == 3)] = 1
    TREE_ONE = TREE.mod.node_scale_root_height(1)
    TEST = DiscreteMarkovModel(TREE_ONE, DATA, 'ARD', prior=0.5)
    TEST.optimize()

    null_test=NullDistribution(TREE, TEST.alpha, TEST.beta)
    null_test.get_likelihoods()
    print(null_test.likelihoods)