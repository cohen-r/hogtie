#! usr/bin/env python

"""
runs hogtie functionalities on inputs, generates outputs
"""

class Hogtie:
    """
    
    """
    def __init__(self):
        pass

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

if __name__ == "__main__":

