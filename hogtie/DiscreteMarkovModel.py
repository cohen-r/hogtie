#!/usr/bin/env python

"""
Implements binary state markov model for ancestral character state reconstruction
"""

import numpy as np
import pandas as pd
import toytree
from scipy.optimize import minimize
from scipy.linalg import expm
from loguru import logger

class DiscreteMarkovModel:
    """
    Ancestral State Reconstruction for discrete binary state characters
    on a phylogeny ... 
    alternative names: DiscreteMarkovModel
    Discrete markov model implemented as described by Pagel (1994). This model
    is similar to that used in ace (R package, https://rdrr.io/cran/ape/man/ace.html); however, 
    unlike ace, hogtie assumes a uniform prior at the root. Hogtie is designed to run across
    large-matrices corresponding to genetic variants identified in sequence data (kmers, snps,
    transcripts, etc.) and allows for visualization of inferred character states and likelihoods
    along a tree and genome of interest, respectively. Either an equal rates (ER, transition 
    rate parameters are equal) or all rates different (ARD, transition rate parameters are unequal)
    can be selected.
    
    Parameters:
    ===========
    tree: newick string or toytree object
        species tree to be used. ntips = number of rows in data matrix
    data: pandas.dataframe object, csv (re-implement csv read-in)
        matrix of 1's and 0's corresponding to presence/absence data of the sequence variant at
        the tips of the input tree. Column number must equal the number of tips on the input tree. 
    model: str
         Either equal rates ('ER') or all rates different ('ARD').
    prior: float
        Prior probability that the root state is 1 (default=0.5). Flat, uniform prior is assumed.
    """

    def __init__(self, tree, data, model, prior=0.5):
          
        # store user inputs
        self.tree = tree
        self.data = data
        self.model = model
        self.prior_root_is_1 = prior

        # set to initial values based on the tree height units.
        self.qmat = None
        self.alpha = 1 / tree.treenode.height
        self.beta = 1 / tree.treenode.height

        assert set(data.columns) == set(tree.get_tip_labels()), (
            "data column names must match tree tip names\n"
            f"data: {data.columns}\n"
            f"tips: {tree.get_tip_labels()}"
        )

        # set likelihoods to 1 for data at tips, and None for internal
        self.unique = None
        self.inverse = None
        self.get_unique_data()
        self.set_node_arrays_to_tree()
        self.set_qmat()

        # results storage
        self.model_fit = {}
        self.log_likelihoods = np.zeros(self.data.shape, dtype=float)


    def set_qmat(self):
        """
        Instantaneous transition rate matrix (Q). 
        This returns the 
        matrix given the values currently set on .alpha and .beta.
        """
        if self.model == 'ER':
            self.qmat = np.array([
                [-self.alpha, self.alpha],
                [self.alpha,  -self.alpha],
                ])
        
        elif self.model == 'ARD':
            self.qmat = np.array([
                [-self.alpha, self.alpha],
                [self.beta, -self.beta]
               ])
        else:
            raise Exception("model must be specified as either 'ER' or 'ARD'")


    def set_node_arrays_to_tree(self):
        """
        Set observation states at the tips for all nodes based on the
        data in self.data using the column labels to align with tip 
        labels.
        """
        for node in self.tree.idx_dict.values():
            if node.is_leaf():
                # get column index from orig data
                cidx = self.data.columns.tolist().index(node.name)
                # get column from unique and enter to node as [1, 0]
                dat = self.unique[:, cidx]
                stack = np.column_stack([dat, 1 - dat])
                node.likelihood = stack
            else:
                node.likelihood = None


    def get_unique_data(self):
        """
        Gets matrix that contains only columns with unique pattern of 
        1's and 0's
        """
        # get unique patterns
        self.unique, self.inverse = np.unique(
            self.data,
            return_inverse=True,
            axis=0,
        )
        logger.debug(f"uniq array shape: {self.unique.shape}")


    def node_conditional_likelihood(self, node):
        """
        Returns the conditional likelihood at a single node given the
        likelihood's of data at its child nodes.
        """
        # get transition probabilities over each branch length
        prob_child0 = expm(self.qmat * node.children[0].dist)
        prob_child1 = expm(self.qmat * node.children[1].dist)

        # likelihood that child 0 observation occurs if anc==0
        child0_is0 = (
            prob_child0[0, 0] * node.children[0].likelihood[:, 0] + 
            prob_child0[0, 1] * node.children[0].likelihood[:, 1]
        )

        # likelihood that child 1 observation occurs if anc==0
        child1_is0 = (
            prob_child1[0, 0] * node.children[1].likelihood[:, 0] + 
            prob_child1[0, 1] * node.children[1].likelihood[:, 1]
        )
        anc_lik_0 = child0_is0 * child1_is0

        # likelihood that child 0 observation occurs if anc==1
        child0_is1 = (
            prob_child0[1, 0] * node.children[0].likelihood[:, 0] + 
            prob_child0[1, 1] * node.children[0].likelihood[:, 1]
        )
        child1_is1 = (
            prob_child1[1, 0] * node.children[1].likelihood[:, 0] + 
            prob_child1[1, 1] * node.children[1].likelihood[:, 1]
        )
        anc_lik_1 = child0_is1 * child1_is1

        # set estimated conditional likelihood on this node
        node.likelihood = np.column_stack([anc_lik_0, anc_lik_1])


    def pruning_algorithm(self):
        """
        Traverse tree from tips to root calculating conditional 
        likelihood at each internal node on the way, and compute final
        conditional likelihood at root based on priors for root state.
        """
        # traverse tree in order **from tips to root**
        # to get conditional likelihood estimate at root.
        for node in self.tree.treenode.traverse("postorder"):
            if not node.is_leaf():               
                self.node_conditional_likelihood(node)

        # multiply root prior times the conditional likelihood at root
        root = self.tree.treenode
        lik = (
            (1. - self.prior_root_is_1) * root.likelihood[:, 0] + 
            self.prior_root_is_1 * root.likelihood[:, 1]
        )
        
        return lik[self.inverse]
        

    def optimize(self):
        """
        Use maximum likelihood optimization to find the optimal alpha
        and beta model parameters to fit the data.
        TODO: max bounds could be set based on tree height. For smaller
        tree heights (e.g., 1) the max should likely be higher. If the 
        estimated parameters is at the max bound we should report a 
        logger.warning(message).
        """  
        if self.model == 'ARD':
            estimate = minimize(
                fun=optim_func,
                x0=np.array([self.alpha, self.beta]),
                args=(self,),
                method='L-BFGS-B',
                bounds=((1e-12, 500), (1e-12, 500)),
            )
        elif self.model == 'ER':
            estimate = minimize(
                fun=optim_func,
                x0=np.array([self.alpha]),
                args=(self,),
                method='L-BFGS-B',
                bounds=[(1e-12, 50)],
            )

        # store results
        self.alpha = estimate.x[0]
        self.beta = estimate.x[1] if self.model == "ARD" else np.nan
        self.model_fit = {
            "alpha": self.alpha,
            "beta": self.beta,
            "negLogLik": estimate.fun,
            "convergence": estimate.success,
        }

        # one last fit to the data using estimate parameters
        self.set_qmat()
        self.log_likelihoods = -np.log(self.pruning_algorithm())


def optim_func(params, model):
    """
    Function to optimize. Takes an iterable as the first argument 
    containing the parameters to be estimated (alpha, beta), and the
    BinaryStateModel class instance as the second argument.
    """
    if model.model == 'ARD':
        model.alpha, model.beta = params
    else:
        model.alpha = params[0]        
    model.set_qmat()
    liks = model.pruning_algorithm()
    return -np.log(liks).sum()


if __name__ == "__main__":
    import ipcoal
    from hogtie.utils import set_loglevel
    set_loglevel("DEBUG")
    import os
#
    #print("important-------")
    #print('toytree', toytree.__version__)
    #print('ipcoal', ipcoal.__version__)
    #print("----------------")
#
    ## GENERATE SOME BINARY DATA
    TREE = toytree.rtree.baltree(ntips=12, treeheight=1e6)
    #MODEL = ipcoal.Model(TREE, Ne=20000, mut=1e-8, seed=123)
    #MODEL.sim_snps(10)
#
    ## DATA is in alphanumeric tipname order
    #DATA = MODEL.write_vcf().iloc[:, 9:]
    #DATA[(DATA == 2) | (DATA == 3)] = 1
    #DATA = DATA.applymap(lambda x: x[0]).astype(int)

    # FIT PARAMS TO DATA on TREE with height=1
    #TREE_ONE = TREE.mod.node_scale_root_height(1)
    
    MODEL = ipcoal.Model(TREE, Ne=20000, mut=1e-8, seed=123)
    MODEL.sim_snps(10)
    DATA = MODEL.write_vcf().iloc[:, 9:]
    DATA[(DATA == 2) | (DATA == 3)] = 1

    # FIT PARAMS TO DATA on TREE with height=1
    TREE_ONE = TREE.mod.node_scale_root_height(1)
    
    TEST = DiscreteMarkovModel(TREE, DATA, 'ARD', prior=0.5)
    TEST.optimize()
    print(TEST.model_fit)
    #print(f"sum of edge lengths: {TREE_ONE.get_node_values('dist').sum()}")

    # # view results next to data.
    # TEST.data["loglik"] = TEST.log_likelihoods
    # print(TEST.data)