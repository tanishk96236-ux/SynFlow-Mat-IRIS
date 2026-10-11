"""SynFlow-Mat: physics- and synthesis-constrained generative framework for materials discovery.

Engines (see SynFlow-Mat_Project_Blueprint.pdf):
  DAO-P  -- energy / stability predictor (EGNN)     <- implemented here
  DAO-G  -- structure generator (Flow Matching+DiT) <- not started
  DAO-S  -- synthesis recipe predictor (Seq2Seq)    <- feasibility spike only
"""

import os

# Two copies of Intel's OpenMP runtime end up in this environment -- one inside
# numpy's MKL, one inside torch -- and MKL responds by calling abort() on the
# process. No traceback, no exception, just exit code 4 at the moment torch is
# imported. It has cost this project twice: 68 tests that silently never ran
# for weeks, and a reaction check that died before printing anything.
#
# Every script sets this at its top, but only the scripts that remembered. Here
# it covers anything that imports synflow at all, and it has to happen before
# any submodule pulls in torch -- which is why this file imports nothing.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
