#! usr/bin/env python

"""
Runs BinaryStateModel on matrix of binary character state data for gwas data for the input tree.
"""


import numpy as np
import toytree
import toyplot
import pandas as pd #assuming matrix will be a pandas df
from loguru import logger
from hogtie import BinaryStateModel


class MatrixParser:
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

        for i in self.matrix:
          if i != 1 or 0:
                raise ValueError('Only valid trait values are 0 and 1')

    @property
    def unique_matrix(self):
        """
        Gets matrix that contains only columns with unique pattern of 1's and 0's
        """
        matrix_array = self.matrix.to_numpy()
        unique_array = np.unique(matrix_array, axis=1)
        unique_matrix = pd.DataFrame(unique_array)
        return unique_matrix

    def matrix_likelihoods(self):
        """
        Gets likelihoods for each column of the matrix
        """
        likelihoods = np.empty((0,len(self.matrix.columns)),float)
        for column in self.unique_matrix:
            data = self.unique_matrix[column]
            out = BinaryStateModel(self.tree, data, self.model, self.prior)
            out.optimize()
        
            lik = out.log_lik

            for col in self.matrix:
                if list(self.matrix[col]) == list(self.unique_matrix[column]):
                    likelihoods = np.append(likelihoods, lik)

        self.likelihoods = pd.DataFrame(likelihoods)
        
    
if __name__ == "__main__":
    import os
    HOGTIEDIR = os.path.dirname(os.getcwd())
    tree1 = toytree.rtree.unittree(ntips=10)
    file1 = os.path.join(HOGTIEDIR, "sampledata", "testmatrix.csv")
    testmatrix = MatrixParser(tree=tree1, matrix=file1, model='ARD')
    testmatrix.matrix_likelihoods()
    print(testmatrix.likelihoods)
