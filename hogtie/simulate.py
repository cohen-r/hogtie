#! usr/bin/env python

"""
generating data using ipcoal
"""

import ipcoal
import toytree

# if this is meant to be executed then you should put it in main
# otherwise it would be run if this file was imported. 
if __name__ == "__main__":
    tree = toytree.rtree.unittree(ntips=10, treeheight=1e5)
    mod = ipcoal.Model(tree=tree, Ne=1e6, admixture_edges=[(3, 8, 0.5, 0.5)], nsamples=1)
    mod.sim_loci(1, nsites=10000)
    genos = mod.write_vcf()
    genos.iloc[:, 9:].T