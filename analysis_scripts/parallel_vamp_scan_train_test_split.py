import pyemma.coordinates as coor
import numpy as np
import pyemma
import os
import pyemma.plots as pyemma_plots
import matplotlib.pyplot as plt
import jug
import time

@jug.TaskGenerator
def cluster_subset(tica_filename, trj_len_cutoff, n_clusters, trial, output_stem, test_size=0.5):
    from sklearn.model_selection import train_test_split

    tica_trjs = coor.load(tica_filename)
    # short_tica_trjs = [t for t in tica_trjs if len(t) < trj_len_cutoff]
    long_tica_trjs = [t for t in tica_trjs if len(t) >= trj_len_cutoff]

    #print("short: ",np.array(short_tica_trjs).shape)
    #print("long: ",np.array(long_tica_trjs).shape)

    # Train test split
    # everything is a "long traj" for myh7b-hs
    train_tica_trjs, test_tica_trjs = train_test_split(long_tica_trjs, test_size=0.5)

    # short_train_tica_trjs, short_test_tica_trjs = train_test_split(short_tica_trjs, test_size=test_size)


    ## uncomment if using trj length cutoffs.
    # train_tica_trjs.extend(short_train_tica_trjs)
    # test_tica_trjs.extend(short_test_tica_trjs)

    # remove unneeded variables from memory
    #del short_train_tica_trjs
    #del short_test_tica_trjs
    del long_tica_trjs
    #del short_tica_trjs

    # time clustering
    start = time.time()

    # determine the amount of jobs available for this
    if 'LSB_DJOB_NUMPROC' in os.environ:
        num_processors = int(os.environ['LSB_DJOB_NUMPROC'])
    else:
        num_processors = None
    print(num_processors)

    clustering = pyemma.coordinates.cluster_kmeans(
        data=train_tica_trjs, k=n_clusters, max_iter=200, n_jobs=num_processors)
    end = time.time()
    print(f'clustering took {end - start} seconds with k={n_clusters} with {num_processors} processors')

    train_assignments = clustering.dtrajs
    train_assignment_filename = f'{output_stem}-k-{n_clusters}-split-{trial}-train-dtrajs.npy'

    # make directory if it does not already exist
    os.makedirs(os.path.dirname(train_assignment_filename), exist_ok=True)

    np.save(train_assignment_filename, train_assignments)

    test_assignments = clustering.transform(test_tica_trjs)
    test_assignment_filename = f'{output_stem}-k-{n_clusters}-split-{trial}-test-dtrajs.npy'
    np.save(test_assignment_filename, test_assignments)

    return (train_assignment_filename, test_assignment_filename)

@jug.TaskGenerator
def vamp2_score(assignment_filenames, lag_time):
    train_assignment_filename, test_assignment_filename = assignment_filenames
    dtrajs_train = list(np.load(train_assignment_filename, allow_pickle=True))
    dtrajs_test = list(np.load(test_assignment_filename, allow_pickle=True))
    dtrajs_test = [np.concatenate(t) for t in dtrajs_test]

    # VAMP-2
    # time clustering
    try:
        start = time.time()
        pyemma_msm = pyemma.msm.estimate_markov_model(dtrajs_train, lag=lag_time, score_method='VAMP2', score_k=10)
        end = time.time()
        print(f'MSM fitting took {end-start} for {os.path.basename(train_assignment_filename)}')

        start = time.time()
        vamp2_train_score = pyemma_msm.score(dtrajs_train, score_method='VAMP2', score_k=10)
        end = time.time()
        print(f'VAMP scoring took {end-start} for {os.path.basename(train_assignment_filename)}')

        vamp2_test_score = pyemma_msm.score(dtrajs_test, score_method='VAMP2', score_k=10)

        return (vamp2_train_score, vamp2_test_score)
    except:
        print(f'trial failed for {train_assignment_filename}')
        return (np.nan, np.nan)

@jug.TaskGenerator
def save_vamp2_scores(vamp2_scores, output_basename):
    print(vamp2_scores)
    vamp2_train_scores = [[s[0] for s in scores_for_k] for scores_for_k in vamp2_scores]
    vamp2_test_scores = [[s[1] for s in scores_for_k] for scores_for_k in vamp2_scores]
    np.save(f'{output_basename}-train.npy', vamp2_train_scores)
    np.save(f'{output_basename}-test.npy', vamp2_test_scores)
    return None


specs = [
    #{
    #    'protein': 'ampc',
    #    'tica_filename': '/home/research/j.lotthammer/j.lotthammer/tica-clustering/ampc/INSERT',
    #    'n_clustercenters': [50,100, 250, 500, 750, 1000, 2000],
    #    'lag': 100,
    #    'output_dir': '/project/bowmanlab/j.lotthammer/tica-clustering/ampc/vamp-scan',
    #},
    {
        'protein': 'myh11-1br2',
        'tica_filename': '/project/bowmore/j.lotthammer/tica-clustering/myh11-1br2/myh11-1br2-backbone-chi1-chis-dihedrals-bleb-pocket-charmm36-sims-lag-500-tica-reduced.h5',
        'n_clustercenters': [50,100, 250, 500, 750, 1000], #, 2000],
        'lag': 500,
        'output_dir': '/project/bowmore/j.lotthammer/tica-clustering/myh11-1br2/vamp-scan'
    },
    {
        'protein': 'myh7-5n6a',
        'tica_filename': '/project/bowmore/j.lotthammer/tica-clustering/myh7-5n6a/myh7-5n6a-backbone-chi1-chis-dihedrals-bleb-pocket-charmm36-sims-lag-500-tica-reduced.h5',
        'n_clustercenters': [50,100, 250, 500, 750, 1000], #2000],
        'lag': 500,
        'output_dir': '/project/bowmore/j.lotthammer/tica-clustering/myh7-5n6a/vamp-scan'
    },
    {
        'protein': 'myh2-5n6a',
        'tica_filename': '/project/bowmore/j.lotthammer/tica-clustering/myh2-5n6a/myh2-5n6a-backbone-chi1-chis-dihedrals-bleb-pocket-charmm36-sims-lag-500-tica-reduced.h5',
        'n_clustercenters': [50,100, 250, 500, 750, 1000], #, 2000],
        'lag': 500,
        'output_dir': '/project/bowmore/j.lotthammer/tica-clustering/myh2-5n6a/vamp-scan'
    },
    {
        'protein': 'myh9-5i4e',
        'tica_filename': '/project/bowmore/j.lotthammer/tica-clustering/myh9-5i4e/myh9-5i4e-backbone-chi1-chis-dihedrals-bleb-pocket-charmm36-sims-lag-500-tica-reduced.h5',
        'n_clustercenters': [50,100, 250, 500, 750, 1000], #2000],
        'lag': 500,
        'output_dir': '/project/bowmore/j.lotthammer/tica-clustering/myh9-5i4e/vamp-scan'
    },
    {
        'protein': 'myh7b-hs',
        'tica_filename': '/project/bowmore/j.lotthammer/tica-clustering/myh7b-hs/myh7b-hs-backbone-chi1-chis-dihedrals-bleb-pocket-charmm36-sims-lag-500-tica-reduced.h5',
        'n_clustercenters': [50,100, 250, 500, 750, 1000], #2000],
        'lag': 500,
        'output_dir': '/project/bowmore/j.lotthammer/tica-clustering/myh7b-hs/vamp-scan'
    },
]

n_trials = 10
trj_len_cutoff = 50 * 5 # in frames (50 ns * 5 frames / ns)

for spec in specs:
    tica_filename = spec['tica_filename']
    output_dir = spec['output_dir']
    output_stem = f"{output_dir}/{os.path.basename(tica_filename).split('.')[0]}"
    print(output_stem)
    vamp2_scores = []
    for k in spec['n_clustercenters']:
        scores_for_k = []
        for trial in range(n_trials):
            assignment_filenames = cluster_subset(
                tica_filename, trj_len_cutoff, k, trial, output_stem)
            scores_for_k.append(vamp2_score(assignment_filenames, spec['lag']))
        vamp2_scores.append(scores_for_k)

    scan_description = '-'.join(str(k) for k in spec['n_clustercenters'])
    output_basename = f"{output_stem}-vamp-scan-lagtime-{spec['lag']}-k-{scan_description}"
    save_vamp2_scores(vamp2_scores, output_basename)

