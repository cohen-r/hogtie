#! usr/bin/env python

"""
Runs BinaryStateModel on matrix of binary character state data for gwas data for the input tree.
"""


import numpy as np
import toytree
import toyplot
import pandas as pd
from scipy.optimize import minimize
from loguru import logger
from hogtie import BinaryStateModel


class MatrixParser():
    """
    Runs BinaryStateModel on matrix columns, returns a likelihood score for each column.
    The matrix should correspond to presence/absence data corresponding to sequence variants (this
    could be kmers, snps, transcripts, etc.).

    Parameters
    ----------
    tree: newick string or toytree object
        species tree to be used. ntips = number of rows in data matrix
    matrix: pandas.dataframe object, csv
        matrix of 1's and 0's corresponding to presence/absence data of the sequence variant at the tips of 
        the input tree. Row number must equal tip number. 
    model: str
        Either equal rates ('ER') or all rates different ('ARD')
    prior: float
        Prior probability that the root state is 1 (default=0.5). Flat, uniform prior is assumed.

    """
    def __init__(self, 
        tree,               #must be Toytree class object
        matrix = None,      #must be pandas DataFrame class object
        model = None,
        prior = 0.5    
        ):

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
        self.prior = prior

        self.alpha = 1 / tree.treenode.height
        self.beta = 1 / tree.treenode.height

        #for i in self.matrix:
        #  if i != 1 or 0:
        #        raise ValueError('Only valid trait values are 0 and 1')

    @property
    def unique_matrix(self):
        """
        Gets matrix that contains only columns with unique pattern of 1's and 0's
        """
        matrix_array = self.matrix.to_numpy()
        unique_array = np.unique(matrix_array, axis=1)
        unique_matrix = pd.DataFrame(unique_array)
        return unique_matrix

    def matrix_likelihood(self):
        """
        Gets likelihoods for each column of the matrix
        """
        likelihoods = np.empty((0,len(self.matrix.columns)),float)
        for column in self.unique_matrix:
            out = BinaryStateModel(self.tree, self.unique_matrix[column], self.model, self.prior)
            lik = out.pruning_algorithm()

            for col in self.matrix:
                if list(self.matrix[col]) == list(self.unique_matrix[column]):
                    likelihoods = np.append(likelihoods, lik)

        lik_sum = sum(likelihoods)
        self.likelihoods = pd.DataFrame(likelihoods)

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
        # logger.info(estimate)

        # organize into a dict
            result = {
                "alpha": round(estimate.x[0], 6),
                "beta": round(estimate.x[1], 6), 
                "Lik": round(estimate.fun, 6),            
                #"negLogLik": round(-np.log(-estimate.fun), 2),
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
                #"negLogLik": estimate.fun,
                "convergence": estimate.success,
                }
            logger.info(result)

        else:
            raise Exception('model must be specified as either ARD or ER')

        # get scaled likelihood values
        #self.log_lik = result["negLogLik"]
        #self.tree = self.tree.set_node_values(
        #    'likelihood',
        #    values={
        #        node.idx: np.array(node.likelihood) / sum(node.likelihood)
        #        for node in self.tree.idx_dict.values()
        #    }
        #)

def optim_func(params, model):
    """
    Function to optimize. Takes an iterable as the first argument 
    containing the parameters to be estimated (alpha, beta), and the
    BinaryStateModel class instance as the second argument.
    """
    if model.model == 'ARD':
        model.alpha, model.beta = params
        lik = model.matrix_likelihood()

    else:
        model.alpha = params[0]
        lik = model.matrix_likelihood()
    
    return lik       
    
if __name__ == "__main__":
    from hogtie.utils import set_loglevel
    set_loglevel("DEBUG")
    
    import os
    HOGTIEDIR = os.path.dirname(os.getcwd())
    tree1 = toytree.rtree.unittree(ntips=10)
    file1 = os.path.join(HOGTIEDIR, "sampledata", "testmatrix.csv")
    testmatrix = MatrixParser(tree=tree1, matrix=file1, model='ER')
    testmatrix.optimize()
