import numpy as np
import matplotlib.pyplot as plt
import enspara
import pickle
import enspara
from enspara import ra
from enspara.msm import MSM, builders, implied_timescales
from functools import partial
from glob import glob
import pyemma as pe
from natsort import natsorted,ns
from tqdm import tqdm

isoforms = ["myh11-1br2","myh9-5i4e","myh7-5n6a","myh2-5n6a","myh7b-hs"]
#isoforms = ["myh7-5n6a"]
#isoforms = ["ampc"]
#isoforms = ["myh7b-hs"]
frames_per_ns = 50

lagtime = frames_per_ns*ns
centers = [50,100,250,500,750,1000]
#centers = [500] 
#centers = [200]
#centers = [500] 
#centers = [50,100,200,300,400,500,750,1000]

lag_times = {
    'myh11-1br2': 5, 
    'myh9-5i4e': 8,
    'myh7-5n6a': 5,
    'myh2-5n6a': 5,
    'myh2-6fsa': 2.5,
    'myh7b-hs': 5,
}

n_clusters = {
    'myh11-1br2': 50,
    'myh9-5i4e': 100,
    'myh7-5n6a': 100,
    'myh2-5n6a': 50,
    'myh2-6fsa': 100,
    'myh7b-hs': 100,
}


for isoform in isoforms:
    lagtime = lag_times[isoform]
    centers = n_clusters[isoform]
    assignment_files = glob(f"{isoform}/{isoform}*backbone-chi1-chis-dihedrals-bleb-pocket-charmm36-sims-lag-500-tica-reduced-k-{centers}-cluster-dtrajs.h5")
    print(assignment_files)

    for k,assigns in zip([centers],assignment_files):
            print(assigns)
            assignments = pe.coordinates.load(assigns)
            for i, trj in enumerate(assignments):
                assignments[i] = trj.astype(int).flatten()
            print(f"assign file: {assigns}")
            print(f"on isoform {isoform} center {k}")
            msm = pe.msm.estimate_markov_model(assignments, lag=lagtime*frames_per_ns, reversible=True)
            with open(f"{isoform}/{isoform}-backbone-all-chis-dihedrals-bleb-pocket-charmm36-sims-tica-lag-500-k-{k}-msm-lag-{lagtime}ns-msm.pickle","wb") as f:
                pickle.dump(msm,f)
                eq_probs = msm.pi
                print("isoform,k.max_eq_prob",isoform,k,max(eq_probs))
                with open(f"{isoform}/{isoform}-backbone-all-chis-dihedrals-bleb-pocket-charmm36-sims-tica-lag-500-k-{k}-msm-lag-{lagtime}ns-eq-probs.pickle","wb") as fh:
                    pickle.dump(eq_probs,fh)
