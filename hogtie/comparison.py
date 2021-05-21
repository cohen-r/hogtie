#! usr/bin/env python

"""
runs hogtie functionalities on inputs, generates outputs
"""

import pandas as pd
import toyplot
from scipy.stats import zscore
from hogtie.DiscreteMarkovModel import DiscreteMarkovModel
from hogtie.simulate import NullDistribution


class Hogtie:
    """
    Run DiscreteMarkovModel on the experimental data and NullDistribution
    to parameterize the input species tree, then compare the data to
    expectations and graph it
    """
    def __init__(self, tree, data, model, prior=0.5, significance_level=0):
        #save user inputs
        self.tree = tree
        self.data = data
        self.model = model
        self.prior = prior
        self.significance_level = significance_level

        #make dataframe to be filled in as analyses are run
        self.output = pd.DataFrame()

    def run(self):
        """
        Run analyses, build output dataframe
        """
        exp = DiscreteMarkovModel(self.tree, self.data, self.model, self.prior)
        exp.optimize()

        null = NullDistribution(self.tree, self.model)
        null.optimize()

        self.output["liks"] = exp.log_likelihoods
        #self.output["z_scores"] = zscore(df['rollingav'],nan_policy='omit') #z-score of what?

        null_std = null.log_likelihoods.std()
        null_mean = null.log_likelihoods.mean()
    
        #get the likelihood value that falls the number of sd's away from the mean
        #indicated by the significance value
        high_lik = null_mean + (self.significance_level * null_std)
        ##self.likelihoods['rollingav']= self.likelihoods['negloglikes'].rolling(50).mean()



    #def genome_graph(self):
        ##"""
        ##Graphs rolling average of likelihoods along the linear genome, identifies 
        ##regions that deviate significantly from null expectations
##
        ##TO DO: fix
        ##"""
##      #get z -scores null likelihood expectations
##
        ##likelihood_density_scores = []
        ##for i in self.likelihoods.index:
            ##end = i + 49
            ##dens = self.likelihoods.iloc[i:end,1]
            ##likelihood_density_scores.append(dens)
##        
        #a,b,c = toyplot.plot(
            #likelihood_density_scores,
            #width = 500,
            #height=500,
            #color = 'blue',
        #)
##
        #b.hlines(high_lik, style={"stroke": "red", "stroke-width": 2});

if __name__ == "__main__":
    import toytree
    import ipcoal
    TREE = toytree.rtree.baltree(ntips=12, treeheight=1e6)
    MODEL = ipcoal.Model(TREE, Ne=20000, mut=1e-8, seed=123)
    MODEL.sim_snps(10)
    DATA = MODEL.write_vcf().iloc[:, 9:]
    DATA[(DATA == 2) | (DATA == 3)] = 1
    TREE_ONE = TREE.mod.node_scale_root_height(1)
    test = Hogtie(TREE, DATA, 'ARD')
    test.run()
    print(test.output)

