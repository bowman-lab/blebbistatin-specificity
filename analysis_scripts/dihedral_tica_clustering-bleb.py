import pyemma
import mdtraj as md
import numpy as np
from glob import glob
import os
import jug
import pickle

@jug.TaskGenerator
def mini_cluster(tica_filename, stride, n_clusters=50, overwrite=False):
    import pyemma.coordinates as coor

    cluster_filename = f"{os.path.splitext(tica_filename)[0]}-k-{n_clusters}-mini-cluster-object.h5"
    if os.path.exists(cluster_filename) and not overwrite:
        print('cluster file already exists')
        return cluster_filename

    tica_data = coor.load(tica_filename, stride=stride)
    cl = coor.cluster_kmeans(tica_data, k=n_clusters, max_iter=250)

    cluster_filename = f"{os.path.splitext(tica_filename)[0]}-k-{n_clusters}-mini-cluster-object.h5"
    
    # since overwrite is broken, delete file and create new
    if os.path.exists(cluster_filename):
        os.remove(cluster_filename)
    cl.save(cluster_filename, overwrite=True)

    cluster_dtraj_filename = f"{os.path.splitext(tica_filename)[0]}-k-{n_clusters}-cluster-dtrajs.h5"
    # since overwrite is broken, delete file and create new
    if os.path.exists(cluster_dtraj_filename):
        os.remove(cluster_dtraj_filename)
    print('writing cluster file')
    cl.write_to_hdf5(cluster_dtraj_filename)
    
    return cluster_filename

@jug.TaskGenerator
def tica_reduce(feature_filename, lag_time, tica_filename, var_cutoff=0.9,
                save_tica_obj=False, overwrite=False):
    import pyemma.coordinates as coor

    # if output and we are not overriding, return filename
    if os.path.exists(tica_filename) and not overwrite:
        print('tica file already exists')
        return tica_filename

    data = coor.source(feature_filename)
    # print(f'Data shape: {data[0].shape}')
    # print(f'Number of trajectories: {len(data)}')

    # I assume lag is in frames
    tica = coor.tica(data=data, lag=lag_time, kinetic_map=False, commute_map=True)

    tica.var_cutoff = var_cutoff

    np.save(tica_filename.replace('tica-reduced.h5', 'tica-cumvar.npy'), tica.cumvar)

    # save out eigenvectors to get a sense of which features are being selected
    np.save(tica_filename.replace('tica-reduced.h5', 'tica-eigenvectors.npy'), tica.eigenvectors)

    print('Number of dimentions saved is: ', tica.dimension())

    tica.write_to_hdf5(tica_filename, overwrite=True)

    if save_tica_obj:
        tica.save(tica_filename.replace('tica-reduced.h5', 'tica-object.pkl'))
    return tica_filename


@jug.TaskGenerator
def backbone_dihedral_featurize(traj_path, top_path, OUT_STEM, feature_filename,
    selstr="", stride=1):

    pdb = md.load(top_path)

    feat = pyemma.coordinates.featurizer(pdb)
    feat.add_backbone_torsions(selstr=selstr, cossin=True, periodic=False)

    # save out description of features
    np.save(feature_filename.replace('features.h5', 'feature-descriptions.npy'), feat.describe())

    traj_list = glob(f'{traj_path}/*.xtc')

    print(feat)

    # reader = pyemma.coordinates.source(traj_list, features=feat, stride=stride)

    os.makedirs(OUT_STEM, exist_ok=True)

    # h5_opt = {
    #     'chunks': True,
    #     'compression': 'gzip',
    #     'scaleoffset': 0,
    #     'shuffle': True,
    # }
    # reader.write_to_hdf5(feature_filename) #, h5_opt=h5_opt)

    return feature_filename

@jug.TaskGenerator
def backbone_sidechain_dihedral_featurize(traj_path, top_path, OUT_STEM, feature_filename,
    selstr="", stride=1, chi_selstr=None):
    # Allows for different residue selections between phi, psi and chi
    if chi_selstr is None:
        chi_selstr = selstr

    pdb = md.load(top_path)

    feat = pyemma.coordinates.featurizer(pdb)
    feat.add_backbone_torsions(selstr=selstr, cossin=True, periodic=False)
    feat.add_sidechain_torsions(selstr=chi_selstr, cossin=True, periodic=False, which='chi1')

    # save out description of features
    os.makedirs(OUT_STEM, exist_ok=True)
    np.save(feature_filename.replace('features.h5', 'feature-descriptions.npy'), feat.describe())

    traj_list = sorted(glob(f'{traj_path}/*.xtc'))

    print(feat)

    reader = pyemma.coordinates.source(traj_list, features=feat, stride=stride)
    print(reader)
    reader.write_to_hdf5(feature_filename)

    return feature_filename

@jug.TaskGenerator
def backbone_sidechain_dihedral_featurize_paths(traj_paths, top_path, OUT_STEM, feature_filename,
    selstr="", stride=1, chi_selstr=None, include_sidechains=True, which_chis='chi1'):

    print(f"Beginning featurization for {feature_filename} \n")

    # Allows for different residue selections between phi, psi and chi
    if chi_selstr is None:
        chi_selstr = selstr

    # if feature filename already exists return
    if os.path.exists(feature_filename):
        print('feature file already exists')
        return feature_filename

    pdb = md.load(top_path)

    feat = pyemma.coordinates.featurizer(pdb)
    feat.add_backbone_torsions(selstr=selstr, cossin=True, periodic=False)
    if include_sidechains:
        feat.add_sidechain_torsions(selstr=chi_selstr, cossin=True, periodic=False, which=which_chis)

    # save out description of features
    os.makedirs(OUT_STEM, exist_ok=True)
    np.save(feature_filename.replace('features.h5', 'feature-descriptions.npy'), feat.describe())

    traj_list = sorted(list(np.concatenate([glob(traj_path) for traj_path in traj_paths])))
    print(f"trajectory_list: {traj_list}\n")
    reader = pyemma.coordinates.source(traj_list, features=feat, stride=stride, chunksize=200000)
    reader.write_to_hdf5(feature_filename)

    return feature_filename



# CHARMM36 Blebbistatin Pocket Sims (PPS Apo Isoforms)
# in principle could add cpu sims but dont feel like rerunning everything bc its slow
trajectory_paths = {
    'myh11-1br2': [
        '/project/bowmanlab/j.lotthammer/Simulations/myosin/specificity/pps-bleb-isoforms/myh11/run*/frame0_masses.xtc',
        '/project/bowmanlab/j.lotthammer/Simulations/folding_at_home/highland2/trajectories/processed/myh11-1br2/aligned-18321/run-*/clone-*/*.xtc',
        "/project/bowmanlab/ameller/simulations/long-pps-simulations/myh11/run*/frame0_masses.xtc",
    ],
    'myh7-5n6a': [
        '/project/bowmanlab/j.lotthammer/Simulations/myosin/specificity/pps-bleb-isoforms/myh7/run*/frame0_masses.xtc',
        '/project/bowmanlab/j.lotthammer/Simulations/folding_at_home/highland2/trajectories/processed/myh7-5n6a/aligned-18319/run-*/clone-*/*.xtc',
        "/project/bowmanlab/ameller/simulations/long-pps-simulations/myh7/run*/frame0_masses.xtc",
        ],
    'myh9-5i4e': [
        '/project/bowmanlab/j.lotthammer/Simulations/myosin/specificity/pps-bleb-isoforms/myh9/run*/frame0_masses.xtc',
        '/project/bowmanlab/j.lotthammer/Simulations/folding_at_home/highland2/trajectories/processed/myh9-5I4E/aligned-18320/run-*/clone-*/*.xtc',
        "/project/bowmanlab/ameller/simulations/long-pps-simulations/myh9/run*/frame0_masses.xtc",
    ],
    'myh2-5n6a': [
        '/project/bowmanlab/j.lotthammer/Simulations/myosin/specificity/pps-bleb-isoforms/myh2/run*/frame0_masses.xtc',
        '/project/bowmanlab/j.lotthammer/Simulations/folding_at_home/highland2/trajectories/processed/myh2-5n6a/aligned-18318/run-*/clone-*/*.xtc',
       "/project/bowmanlab/ameller/simulations/long-pps-simulations/myh2/run*/frame0_masses.xtc",
    ],
    'myh7b-hs': [
        '/project/bowmanlab/j.lotthammer/Simulations/myosin/myosin-7b/act_site_redo/run*/frame0_masses.xtc',
    ],
}

topologies = {
    'myh11-1br2': '/project/bowmanlab/j.lotthammer/Simulations/myosin/specificity/pps-bleb-isoforms/myh11/input_files3/myh11-1br2-holo-prot-masses.pdb',
    'myh2-5n6a': '/project/bowmanlab/j.lotthammer/Simulations/myosin/specificity/pps-bleb-isoforms/myh2/input_files3/myh2-5n6a-holo-prot-masses.pdb',
    'myh7-5n6a': '/project/bowmanlab/j.lotthammer/Simulations/myosin/specificity/pps-bleb-isoforms/myh7/input_files3/myh7-5n6a-holo-prot-masses.pdb',
    'myh9-5i4e': '/project/bowmanlab/j.lotthammer/Simulations/myosin/specificity/pps-bleb-isoforms/myh9/input_files3/myh9-5I4E-holo-prot-masses.pdb',
    'myh7b-hs': '/project/bowmanlab/j.lotthammer/Simulations/myosin/myosin-7b/act_site_redo/eq/myh7b-5n6a-holo-prot-masses.pdb',
}


# Alignment is the same as the one used for Amber sims
#pocket_resis_dict = pickle.load(open(
#    "/project/bowmanlab/j.lotthammer/notebooks/bleb-fast-isoform-alignment.pickle",
#    'rb'))
#conserved_pocket_resis_dict = pickle.load(open(
#    "/project/bowmanlab/j.lotthammer/notebooks/bleb-pocket-conserved-residue-alignment.pickle",
#    "rb"))

# Alignment is the same as the one used for Amber sims
pocket_resis_dict = pickle.load(open(
    "/project/bowmore/ameller/projects/notebooks/tmp/bleb-pocket-residue-alignment-with-7b.pickle",
    'rb'))

conserved_pocket_resis_dict = pickle.load(open(
    "/project/bowmore/ameller/projects/notebooks/tmp/bleb-pocket-conserved-residue-alignment-with-7b.pickle",
    "rb"))


to_cluster = {}

for protein in trajectory_paths.keys():
    print(protein)
    to_cluster[protein] = {
        'traj_paths': trajectory_paths[protein],
        'top_path': topologies[protein],
        'stride': 1,
        'selstr': ' or '.join(f'residue {r}' for r in pocket_resis_dict[protein]),
        'chi_selstr': ' or '.join(f'residue {r}' for r in conserved_pocket_resis_dict[protein]),
        'description': 'bleb-pocket-charmm36-sims',
        'which_chis': 'chi1',
        'tica-lags': [500],
        'k': [50,100,250,500,750,1000],
    }
    
    if protein == 'myh2-6fsa':
        to_cluster[protein]['selstr'] = ' or '.join(f'residue {r}' for r in pocket_resis_dict['myh2-5n6a'])
        to_cluster[protein]['chi_selstr'] = ' or '.join(f'residue {r}' for r in conserved_pocket_resis_dict['myh2-5n6a'])
    elif protein == 'myh7b-hs-5n6a':
        to_cluster[protein]['selstr'] = ' or '.join(f'residue {r}' for r in pocket_resis_dict['myh7b-hs'])
        to_cluster[protein]['chi_selstr'] = ' or '.join(f'residue {r}' for r in conserved_pocket_resis_dict['myh7b-hs'])
    else:
        to_cluster[protein]['selstr'] = ' or '.join(f'residue {r}' for r in pocket_resis_dict[protein])
        to_cluster[protein]['chi_selstr'] = ' or '.join(f'residue {r}' for r in conserved_pocket_resis_dict[protein])

var_cutoff = 0.9
for protein, specs in to_cluster.items():
    OUT_STEM = f"/project/bowmore/j.lotthammer/tica-clustering/{protein}"
    traj_paths = specs['traj_paths']
    top_path = specs['top_path']
    selstr = specs['selstr']
    stride = specs['stride']
    description = specs['description']
    
    traj_list = sorted(list(np.concatenate([glob(traj_path) for traj_path in traj_paths])))
    os.makedirs(OUT_STEM, exist_ok=True)
    with open(f'{protein}/traj_list.txt', 'w') as f:
            for traj_name in traj_list:
                    f.write(traj_name+'\n')

    if 'chi_selstr' in specs.keys():
        output_filename = f"{OUT_STEM}/{protein}-backbone-{specs['which_chis']}-dihedrals-{description}-features.h5"
        if 'which_chis' in specs.keys():
            feature_filename = backbone_sidechain_dihedral_featurize_paths(traj_paths, top_path,
                                                                           OUT_STEM, output_filename,
                                                                           selstr=selstr, stride=stride,
                                                                           chi_selstr=specs['chi_selstr'],
                                                                           which_chis=specs['which_chis'])
        else:
            # default is chi1
            feature_filename = backbone_sidechain_dihedral_featurize_paths(traj_paths, top_path,
                                                                           OUT_STEM, output_filename,
                                                                           selstr=selstr, stride=stride,
                                                                           chi_selstr=specs['chi_selstr'])
    else:
        output_filename = f'{OUT_STEM}/{protein}-backbone-dihedrals-{description}-features.h5'
        feature_filename = backbone_sidechain_dihedral_featurize_paths(traj_paths, top_path,
                                                                       OUT_STEM, output_filename,
                                                                       selstr=selstr, stride=stride,
                                                                       include_sidechains=False)

    for lag_time in specs['tica-lags']:
        if 'chi_selstr' in specs.keys():
            if 'which_chis' in specs.keys():
                tica_filename = f"{OUT_STEM}/{protein}-backbone-{specs['which_chis']}-chis-dihedrals-{description}-lag-{lag_time}-tica-reduced.h5"
            else:
                tica_filename = f"{OUT_STEM}/{protein}-backbone-chi1-dihedrals-{description}-lag-{lag_time}-tica-reduced.h5"
        else:
            tica_filename = f"{OUT_STEM}/{protein}-backbone-dihedrals-{description}-lag-{lag_time}-tica-reduced.h5"
        tica_filename = tica_reduce(feature_filename, lag_time, tica_filename, var_cutoff, save_tica_obj=True)

        if 'k' in specs.keys():
            for k in specs['k']:
                mini_cluster(tica_filename, stride, n_clusters=k)
