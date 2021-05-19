#! usr/bin/env python

"""
generating null data and testing observations against it
"""

import ipcoal
import toytree
import toyplot
from scipy.optimize import minimize
from scipy.linalg import expm
from loguru import logger
import numpy as np


class NullDistribution:
    """
    Parameterizes the model by simulating a distribution of genealogies and optimizing
    transition rate parameters in a Discrete Markov Model 

    TO DO:
    -this class uses ARD, will implement ER once functional
    -check on alpha/beta calculations
    -Do we need to optimize this or should we just use the optimized alpha/beta from DiscreteMarkovModel
    -Or, alternatively, should the alpha/beta from this be used in DiscreteMarkovModel
    """
    def __init__(self, tree, significance_level=0, prior=0.5):
        # store user inputs
        self.tree = tree
        self.significance_level = significance_level
        #self.model = model
        self.prior_root_is_1 = prior

        #set alpha & beta (to be optimized)
        self.qmat = None
        self.alpha = 1 / tree.treenode.height
        self.beta = 1 / tree.treenode.height

        #initiate model and simulate SNPs using the input species tree
        self.mod = ipcoal.Model(tree=self.tree, Ne=1e6)
        self.mod.sim_snps(10)
        logger.info('Initiated model, simulated SNPs')

        #assign tip values to each genealogy
        self.reorder()
        self.set_qmat()

    def set_initial_likelihoods(self, tree, data):
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
        tree = tree.set_node_values(
            feature="likelihood", 
            values=valuesdict,
            default=None,
        )
        return tree

    def reorder(self):
        """
        mod must be an ipcoal model object
        objective: make a unique dataframe for each genealogy with the sites that follow
        that genealogy
        """
        vcf = self.mod.write_vcf().iloc[:,9:].T
        
        tree_list = []
        #likelihoods = np.empty((0,self.tree.ntips),float)

        for idx in self.mod.df.index:
            genealogy = toytree.tree(self.mod.df.iloc[idx, 6], tree_format=0)
            tree_list.append(genealogy)
            
        for col in vcf.columns:
            data = vcf[col].reindex(tree_list[col].get_tip_labels()).to_numpy()
            tree_list[col] = self.set_initial_likelihoods(tree_list[col], data)

        self.tree_list = tree_list

    def set_qmat(self):
        """
        Instantaneous transition rate matrix (Q). 
        This returns the 
        matrix given the values currently set on .alpha and .beta.
        """
        
        self.qmat = np.array([
            [-self.alpha, self.alpha],
            [self.beta, -self.beta]
           ])
       
        
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


    def get_likelihoods(self):
        """
        """
        likelihoods = np.empty((0,self.tree.ntips),float)

        for tree in self.tree_list:
            lik = self.pruning_algorithm(tree)
            likelihoods = np.append(likelihoods, lik)

        return likelihoods

    def optimize(self):
        """
        Optimizing across the distribution of 
        """  
        
        estimate = minimize(
            fun=optim_func,
            x0=np.array([self.alpha, self.beta]),
            args=(self,),
            method='L-BFGS-B',
            bounds=((1e-12, 500), (1e-12, 500)),
        )
        
        # store results
        self.alpha = estimate.x[0]
        self.beta = estimate.x[1]
        self.model_fit = {
            "alpha": self.alpha,
            "beta": self.beta,
            "negLogLik": estimate.fun,
            "convergence": estimate.success,
        }

        # one last fit to the data using estimate parameters
        self.set_qmat()
        self.log_likelihoods = -np.log(self.get_likelihoods())

    def genome_graph(self):
        #"""
        #Graphs rolling average of likelihoods along the linear genome, identifies 
        #regions that deviate significantly from null expectations
#
        #TO DO: fix
        #"""
#       #get z -scores null likelihood expectations
        null_std = null.likelihoods.std()
        null_mean = null.likelihoods.mean()
    
        #get the likelihood value that corresponds 2 standard deviations above the null mean
        high_lik = null_mean + (self.significance_level * null_std)
        #self.likelihoods['rollingav']= self.likelihoods['negloglikes'].rolling(50).mean()
#
        #likelihood_density_scores = []
        #for i in self.likelihoods.index:
            #end = i + 49
            #dens = self.likelihoods.iloc[i:end,1]
            #likelihood_density_scores.append(dens)
#        
        a,b,c = toyplot.plot(
            likelihood_density_scores,
            width = 500,
            height=500,
            color = 'blue',
        )
#
        b.hlines(high_lik, style={"stroke": "red", "stroke-width": 2});

def optim_func(params, model):
    """
    Function to optimize. Takes an iterable as the first argument 
    containing the parameters to be estimated (alpha, beta), and the
    BinaryStateModel class instance as the second argument.
    """
    model.alpha, model.beta = params
    model.set_qmat()
    liks = model.get_likelihoods()
    return -np.log(liks).sum()
    
    

if __name__ == "__main__":
    #get 'experimental' data and run the discretemarkovmodel to get alpha&beta
    TREE = toytree.rtree.baltree(ntips=12, treeheight=1e6)
    TREE_ONE = TREE.mod.node_scale_root_height(1)
    null_test=NullDistribution(TREE_ONE)
    null_test.optimize()
    print(null_test.model_fit)
    print(null_test.log_likelihoods)
