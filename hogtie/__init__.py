#!/usr/bin/env python

"""
Binary state character phylogenetic likelihood calculations
"""


from hogtie.utils import set_loglevel
from hogtie.binary_state_model import DiscreteMarkovModel
from hogtie.matrixlike import MatrixParser
from hogtie.simulate import SimulateNull

__version__ = "0.0.4"

set_loglevel("DEBUG")
