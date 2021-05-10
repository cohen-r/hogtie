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


class BinaryStateModel:
    """
    Ancestral State Reconstruction for discrete binary state characters
    on a phylogeny ... 

    alternative names: DiscreteMarkovModel

    The model is a discrete markov model implemented as described by Pagel (1994). This model
    is similar to that used in ace (R package, add: link); however, hogtie assumes a uniform
    prior at the root. Hogtie is designed to run across large-matrices corresponding to genetic
    variants identified in sequence data (kmers, snps, transcripts, etc.) and allows for visualization
    of inferred character states and likelihoods along a tree and genome of interest, respectively.

    Either an equal rates (ER, transition rate parameters are equal) or all rates different (ARD, 
    transition rate parameters are unequal) can be selected.

    Parameters
    ----------
    tree: newick string or toytree object
        species tree to be used. ntips = number of rows in data matrix
    data: ndarray
        array of integer binary data in order of node indices (0-ntips).
    model: str
         Either equal rates ('ER') or all rates different ('ARD').
    prior: float
        Prior probability that the root state is 1 (default=0.5). Flat, uniform prior is assumed.
    """

    def __init__(self, tree, matrix, model, prior=0.5):
      
        # store user inputs
        if isinstance(tree, toytree.tree):
            self.tree = tree
        elif isinstance(tree, str):
            self.tree = toytree.tree(tree, tree_format=0)
        else: 
            raise Exception('tree must be either a newick string or toytree object')


        if isinstance(matrix, pd.DataFrame):
            self.matrix = matrix  
        else:
            self.matrix = pd.read_csv(matrix, index_col=0)
        
        self.model = model
        self.prior_root_is_1 = prior

        # model parameters to be estimated (in ER model only alpha)
        # set to initial values based on the tree height units.
        self.alpha = 1 / tree.treenode.height
        self.beta = 1 / tree.treenode.height
        self.log_lik = 0.

        if len(self.matrix.index) != self.tree.ntips:
            raise Exception('Matrix row number must equal ntips on tree')

        
        #self.unique_matrix = pd.DataFrame(np.unique(self.matrix.to_numpy(), axis=1))

        # set likelihoods to 1 for data at tips, and None for internal
        #trees = []
        #for column in self.unique_matrix:
        #    tre = self.set_initial_likelihoods(data=self.unique_matrix[column], tree=self.tree)
        #    trees.append(tre)

        #self.tree_list = trees
        #print(self.tree_list)

        #for tre in self.tree_list:
        #    logger.debug(f'tip likelihoods are {tre.get_node_values("likelihood", True, True)}')

    @property
    def qmat(self):
        """
        Instantaneous transition rate matrix (Q). This returns the 
        matrix given the values currently set on .alpha and .beta.
        """
        if self.model == 'ER':
            qmat = np.array([
                [-self.alpha, self.alpha],
                [self.alpha,  -self.alpha],
                ])
        
        elif self.model == 'ARD':
            qmat = np.array([
                [-self.alpha, self.alpha],
                [self.beta, -self.beta]
               ])
        else:
           raise Exception("model must be specified as either 'ER' or 'ARD'")

        return qmat

    @property
    def unique_matrix(self):
        """
        Gets matrix that contains only columns with unique pattern of 1's and 0's
        """
        matrix_array = self.matrix.to_numpy()
        unique_array = np.unique(matrix_array, axis=1)
        unique_matrix = pd.DataFrame(unique_array)
        return unique_matrix

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
        tree = self.tree.set_node_values(
            feature="likelihood", 
            values=valuesdict,
            default=None,
        )
        return tree

        logger.debug(f"set tips values: {valuesdict}")


    def node_conditional_likelihood(self, tree, nidx):
        """
        Returns the conditional likelihood at a single node given the
        likelihood's of data at its child nodes.
        """
        # get the TreeNode 
        node = tree.idx_dict[nidx]

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
        logger.debug(f"node={nidx}; likelihood=[{anc_lik_0:.6f}, {anc_lik_1:.6f}]")
        node.likelihood = [anc_lik_0, anc_lik_1]

    def pruning_algorithm(self, tree):
        """
        Traverse tree from tips to root calculating conditional 
        likelihood at each internal node on the way, and compute final
        conditional likelihood at root based on priors for root state.
        """
        # traverse tree to get conditional likelihood estimate at root.
        for node in tree.treenode.traverse("postorder"):
            if not node.is_leaf():
                self.node_conditional_likelihood(tree, node.idx)

        # multiply root prior times the conditional likelihood at root
        root = tree.treenode
        lik = (
            (1 - self.prior_root_is_1) * root.likelihood[0] + 
            self.prior_root_is_1 * root.likelihood[1]
        )
        return lik

    def matrix_likelihoods(self):
        """
        Gets likelihoods for each column of the matrix
        """
        likelihoods = np.empty((0,len(self.matrix.columns)),float)
        for col in self.unique_matrix:
            tree = self.set_initial_likelihoods(data=self.unique_matrix[col])
            lik = self.pruning_algorithm(tree)

            for column in self.matrix:
                if list(self.matrix[column]) == list(self.unique_matrix[col]):
                    likelihoods = np.append(likelihoods, lik)

        #tree = tree.set_node_values(
            #'likelihood',
            #values={
                #node.idx: np.array(node.likelihood) / sum(node.likelihood)
                #for node in self.tree.idx_dict.values()
            #}
        #)     

        lik_sum = sum(likelihoods)
        return lik_sum

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
            bounds=((0, 50), (0, 50)),
            )
            #logger.info(estimate)

        # organize into a dict
            result = {
                "alpha": round(estimate.x[0], 6),
                "beta": round(estimate.x[1], 6), 
                "Lik": round(estimate.fun, 6),            
                "negLogLik": round(-np.log(-estimate.fun), 2),
                "convergence": estimate.success,
                }
            logger.info(result)

        elif self.model == 'ER':
            estimate = minimize(
                fun=optim_func,
                x0=np.array([self.alpha]),
                args=(self,),
                method='L-BFGS-B',
                bounds=[(0, 50)],
            )

            result = {
                "alpha": estimate.x[0],
                "Lik": estimate.fun,            
                "negLogLik": -np.log(-estimate.fun),
                "convergence": estimate.success,
                }
            logger.info(result)

        else:
            raise Exception('model must be specified as either ARD or ER')

        # get scaled likelihood values
        

    def draw_states(self):
        """
        Draw tree with nodes colored by state
        """
        drawing = self.tree.draw(
            width=400,
            height=300,
            layout='d',
            node_labels=("idx", 1, 1),
            node_sizes=15,
            node_style={"stroke": "black", "stroke-width": 2},
            node_colors=[
                toytree.colors[int(round(i[1]))] if isinstance(i, (list, np.ndarray))
                else "white" 
                for i in self.tree.get_node_values("likelihood", True, True)
            ],
        )
        return drawing

    def get_likelihoods(self):
        """
        Gets a dataframe of likelihoods 1 column x nrows in self.matrix that contains likelihoods
        for each column calculated with the optimized params
        """
        pass




def optim_func(params, model):
    """
    Function to optimize. Takes an iterable as the first argument 
    containing the parameters to be estimated (alpha, beta), and the
    BinaryStateModel class instance as the second argument.
    """
    if model.model == 'ARD':
        model.alpha, model.beta = params
        lik = model.matrix_likelihoods()

    else:
        model.alpha = params[0]
        lik = model.matrix_likelihoods()
    
    return -lik


if __name__ == "__main__":

    from hogtie.utils import set_loglevel
    set_loglevel("DEBUG")
    TREE = toytree.rtree.imbtree(ntips=10, treeheight=1000)

    import os
    HOGTIEDIR = os.path.dirname(os.getcwd())
    tree1 = toytree.rtree.unittree(ntips=10)
    file1 = os.path.join(HOGTIEDIR, "sampledata", "testmatrix.csv")
    mod = BinaryStateModel(TREE, file1, 'ER')
    mod.optimize()

    #DATA = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    #od = BinaryStateModel(TREE, DATA, 'ARD')
    #mod.optimize()

    #DATA = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    #mod = BinaryStateModel(TREE, DATA, 'ER')
    #mod.optimize()
