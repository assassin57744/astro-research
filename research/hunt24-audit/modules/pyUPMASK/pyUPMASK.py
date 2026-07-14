import logging
import os
from pathlib import Path
import numpy as np
from astropy.stats import RipleysKEstimator
import time as t
from modules.pyUPMASK.modules import outer
from modules.pyUPMASK.modules.dataIO import (
    readFiles,
    readINI,
    dread,
    dmask,
    dxynorm,
    dwrite,
)
import multiprocessing as mp

logger = logging.getLogger("AstroPipeline.pyUPMASK")

def dataProcess(
    ID,
    xy,
    data,
    data_err,
    verbose,
    OL_runs,
    parallel_flag,
    parallel_procs,
    resampleFlag,
    PCAflag,
    PCAdims,
    GUMM_flag,
    GUMM_perc,
    KDEP_flag,
    IL_runs,
    N_membs,
    N_cl_max,
    clust_method,
    clRjctMethod,
    C_thresh,
    cl_method_pars,
):
    """ """
    start_t = t.time()

    # TODO this should be handled by the logging() module
    # Set print() according to the 'verbose' parameter
    if verbose == 0:
        prfl = open(os.devnull, "w")
    else:
        prfl = None

    # Print input parameters to screen
    if parallel_flag:
        logger.info("Parallel runs      : {}".format(parallel_flag))
        logger.info("Processes          : {}".format(parallel_procs))
    if PCAflag:
        logger.info("Apply PCA          : {}".format(PCAflag))
        logger.info(" PCA N_dims        : {}".format(PCAdims))
    if GUMM_flag:
        logger.info("Apply GUMM         : {}".format(GUMM_flag))
        logger.info(" GUMM percentile   : {}".format(GUMM_perc))
    if KDEP_flag:
        logger.info("Obtain KDE probs   : {}".format(KDEP_flag))

    logger.info("Outer loop runs    : {}".format(OL_runs))
    logger.info("Inner loop runs    : {}".format(IL_runs))
    logger.info("Stars per cluster  : {}".format(N_membs))
    logger.info("Maximum clusters   : {}".format(N_cl_max))
    logger.info("Clustering method  : {}".format(clust_method))
    if cl_method_pars:
        for key, val in cl_method_pars.items():
            logger.info(" {:<17} : {}".format(key, val))

    Kest = RipleysKEstimator(area=1, x_max=1, y_max=1, x_min=0, y_min=0)


    # Arguments for the Outer Loop
    OLargs = (
        ID,
        xy,
        data,
        data_err,
        resampleFlag,
        PCAflag,
        PCAdims,
        GUMM_flag,
        GUMM_perc,
        KDEP_flag,
        IL_runs,
        N_membs,
        N_cl_max,
        clust_method,
        clRjctMethod,
        Kest,
        C_thresh,
        cl_method_pars,
        prfl,
    )

    # TODO: Breaks if verbose=0
    if parallel_flag is True:
        if parallel_procs == "None":
            # Use *almost* all the cores
            N_cpu = mp.cpu_count() - 1
        else:
            N_cpu = int(parallel_procs)
        with mp.Pool(processes=N_cpu) as p:
            manager = mp.Manager()
            KDE_vals = manager.dict({})
            probs_all = p.starmap(OLfunc, [(OLargs, KDE_vals) for _ in range(OL_runs)])

    else:
        KDE_vals = {}
        probs_all = []
        for _ in range(OL_runs):
            logger.debug("--------------------------------------------------------")
            logger.debug("OL run {}".format(_ + 1))
            # The KDE_vals dictionary is updated after each OL run
            probs, KDE_vals = outer.loop(*OLargs, KDE_vals)
            probs_all.append(probs)

            p_dist = [
                (np.mean(probs_all, 0) > _).sum() for _ in (0.5, 0.75, 0.9, 0.95, 0.99)
            ]
            logger.debug(
                "P>(.5, .75, .9, .95, .99): {}, {}, {}, {}, {}".format(*p_dist),
                # file=prfl,
            )

    elapsed = t.time() - start_t
    if elapsed > 60.0:
        elapsed, ms_id = elapsed / 60.0, "minutes"
    else:
        ms_id = "seconds"
    logger.info("Time consumed: {:.1f} {}".format(elapsed, ms_id))

    return probs_all


def OLfunc(args, KDE_vals):
    """
    Here to handle the parallel runs.
    """
    probs, _ = outer.loop(*args, KDE_vals)
    return probs
