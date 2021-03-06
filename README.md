# HoGTIE (Horizontal Gene Transfer Identification Engine)

First thought to only occur widely in prokaryotic organisms, recent work has shown the prevalence of horizontal gene transfer (HGT) across the tree of life. Identifying regions that contain HGT, though, is difficult in eukaryotic organisms and particularly difficult in plants due to lack of high quality reference genomes and highly variable genomic structure. HoGTIE uses a combination of k-mer-based and likelihood methods to identify genomic regions and sequences that contain HGT in a recipient species by:

1. Treating k-mer presence/absence as a binary character
2. Modeling evolution via a discrete Markov Model, traversing backwards along an input phylogeny to estimate ancestral character states of k-mer presence/absence
3. Identifying k-mers whose distribution along the tips the input phylogeny deviate from expectations. Expectations are set by parameterizing a species tree model through simulations

 The combination of these methodologies allows for comparison of potentially diverse genome structures and sizes in the absence of well-resolved reference genomes across the input phylogeny. HoGTIE visualizes regions of predicted HGT by flagging presence/absence patterns that deviate significantly from expectations along a graph of the linearized recipient genome. 


### In development
HoGTIE is under active development. If you would like to check out or contribute to the code, HoGTIE can be installed locally via:

```
# coming soon...
# conda install hogtie -c conda-forge -c bioconda

# for now, install dependencies (toytree, ipcoal, numpy, pandas, scipy, sys, argparse, os, loguru) and
# do dev installation with pip

git clone https://github.com/cohen-r/hogtie
cd hogtie
pip install -e .

```

If interested in using HoGTIE on a CLI, test your installation by running the command below. The test matrix data is simulated SNP data. HoGTIE will read in the testdata and write the results to a file in an output directory.

```
hogtie --tree sampledata/testtree.txt --data sampledata/testmatrix.csv --model ARD
```

HoGTIE will also run on the API for visualization of likelihood scores and ancestral character states along a tree. These capabilities are currently in development and being tested in the [working example in the notebooks folder](https://github.com/cohen-r/hogtie/blob/main/notebooks/working_example.ipynb), which can be accessed in a jupyter notebook after pip installation.