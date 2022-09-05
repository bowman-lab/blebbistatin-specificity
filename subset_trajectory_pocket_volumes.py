import jug
import numpy as np
import mdtraj as md
from glob import glob
import os
import numba as nb

@nb.njit(fastmath=True,parallel=True)
def calc_distance(vec_1, vec_2, sq_dist_cutoff):
    dists=np.zeros((vec_1.shape[0], vec_2.shape[0]), dtype=np.bool_)
    for i in nb.prange(vec_1.shape[0]):
        for j in range(vec_2.shape[0]):
            dists[i,j]=(((vec_1[i,0]-vec_2[j,0])**2+(vec_1[i,1]-vec_2[j,1])**2+(vec_1[i,2]-vec_2[j,2])**2) < 
                          sq_dist_cutoff)

    return dists

@jug.TaskGenerator
def pocket_vols_to_h5(outfname, pocket_volumes, indexer=None):
    import os
    import tables
    from tqdm import tqdm

    import numpy as np

    if not os.path.isdir(os.path.dirname(outfname)):
        os.mkdir(os.path.dirname(outfname))

    # if file already exists delete it so we don't combine
    # existing file with new file
    if os.path.isfile(outfname):
        os.remove(outfname)

    compression = tables.Filters(complevel=9, complib='zlib', shuffle=True)
    n_zeros = len(str(len(pocket_volumes))) + 1

    with tables.open_file(outfname, 'a') as handle:

        print("writing array to", outfname)
        for i, pocket_vols in enumerate(tqdm(pocket_volumes)):

            data = np.array(jug.bvalue(pocket_vols))
            if indexer is not None:
                data = indexer(data)

            atom = tables.Atom.from_dtype(data.dtype)
            tag = 'array_' + str(i).zfill(n_zeros)

            if tag in handle.root:
                logger.warn(
                    'Tag %s already existed in %s. Overwriting.' %
                    (tag, outfname), RuntimeWarning)
                handle.remove_node('/', name=tag)

            node = handle.create_carray(
                where='/', name=tag, atom=atom,
                shape=data.shape, filters=compression)
            node[:] = data

            del data

    return outfname

def _get_pockets_helper(
        struct, grid_spacing, probe_radius, min_rank, min_cluster_size):
    from enspara import geometry

    pocket_cells = geometry.get_pocket_cells(
        struct, grid_spacing=grid_spacing, probe_radius=probe_radius,
        min_rank=min_rank)
    sorted_pockets, sorted_cluster_mapping = geometry.cluster_pocket_cells(
        pocket_cells, grid_spacing=grid_spacing,
        min_cluster_size=min_cluster_size)
    pockets_as_mdtraj = geometry.xyz_to_mdtraj(
        sorted_pockets, cluster_ids=sorted_cluster_mapping)
    return pockets_as_mdtraj

@jug.TaskGenerator
def calculate_pocket(trj_filename, top, xtal_path, xtal_name,
                     alignment_pickle, gene,
                     region_atom_indices_dict_pickle,
                     region_key, ligand_name, ligand_binding_chainid=0,
                     stride=1, min_rank=6, min_cluster_size=3,
                     grid_spacing=0.07, probe_radius=0.07, dist_cutoff=0.25,
                     continuity_requirement=None):

    from enspara import geometry
    from joblib import Parallel, delayed
    from joblib import parallel_backend
    import pickle
    import scipy
    import time

    region_atom_indices = np.load(region_atom_indices_dict_pickle, allow_pickle=True).item()[region_key]

    trj = md.load(trj_filename, stride=stride, top=top, atom_indices=region_atom_indices)
    if 'LSB_DJOB_NUMPROC' in os.environ:
        num_processors = int(os.environ['LSB_DJOB_NUMPROC'])
    else:
        num_processors = 1

    # for debugging on head node
    # num_processors = 16
    print(f'starting pocket calculations for {trj_filename}')
    print(trj.xyz.shape)
    # print(f'number of processors: {num_processors}')
    print(f'grid spacing {grid_spacing}')
    # Calculate pockets
    t0 = time.perf_counter()

    # os.system("taskset -p 0xff %d" % os.getpid())

    with parallel_backend('multiprocessing', n_jobs=num_processors):
        pockets = Parallel(verbose=10)\
            (delayed(_get_pockets_helper)(struct, grid_spacing, probe_radius, min_rank, min_cluster_size) for struct in trj)
    # pockets = geometry.get_pockets(trj, grid_spacing=grid_spacing, min_rank=min_rank,
    #                                min_cluster_size=min_cluster_size,
    #                                probe_radius=probe_radius, n_procs=num_processors)
    print(f"calculating pockets took {time.perf_counter() - t0: .2f} seconds"
          f" for a {trj.top.n_residues} residue protein with trajectory length {trj.xyz.shape[0]}"
          f" using {num_processors} cores")

    ext = os.path.splitext(alignment_pickle)[-1].lower()
    if ext == '.npy':
        pocket_resis = np.load(alignment_pickle, allow_pickle=True).item()
    else:
        pocket_resis = np.load(alignment_pickle, allow_pickle=True)

    holo_xtal = md.load(xtal_path)

    ref_atom_indices = np.concatenate([trj.top.select('residue %s and backbone and not element H'
                                                      % res)
                                       for res in pocket_resis[gene]])
    atom_indices = [holo_xtal.top.select('residue %s and name %s and chainid %s'
                                         % (pocket_resis[xtal_name][pocket_resis[gene].index(trj.top.atom(aid).residue.resSeq)],
                                            trj.top.atom(aid).name,
                                            ligand_binding_chainid))[0]
                    for aid in ref_atom_indices]

    assert all([trj.top.atom(a).name == holo_xtal.top.atom(b).name
                for a, b in zip(ref_atom_indices, atom_indices)])

    ligand_pocket_volumes = []
    for i in range(trj.xyz.shape[0]):
        # Superimpose holo onto structures
        superimposed = holo_xtal.superpose(trj[i],
                                           ref_atom_indices=ref_atom_indices,
                                           atom_indices=atom_indices)

        # Count number of grid points within cutoff distance of ligand
        ligand_aids = superimposed.top.select("resname '%s'" % ligand_name)
        ligand_xyz = superimposed.xyz[0][ligand_aids]

        sq_dist_cutoff = dist_cutoff ** 2
        dists = calc_distance(ligand_xyz, pockets[i].xyz[0], sq_dist_cutoff)
        # determine number of grid points within cutoff distance of at least one ligand heavy atom
        ligand_pocket_volume = np.sum(np.sum(dists, axis=0) > 0)

        if continuity_requirement:
            if ligand_pocket_volume > 1:

                indices_pockets = np.where(np.sum(dists, axis=0) > 0)[0]
                # Spatially cluster grid points using scipy
                cluster_labels = scipy.cluster.hierarchy.fclusterdata(
                    pockets[i].atom_slice(indices_pockets).xyz.reshape(-1, 3),
                    t=continuity_requirement, criterion='distance')
                unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
                ligand_pocket_volumes.append(np.max(counts))
                # if saving pockets
                # indices_pockets = indices_pockets[cluster_labels == unique_clusters[np.argmax(counts)]]
            else:
                # either a 1 (in which case one cannot calculate a distance matrix) or 0
                ligand_pocket_volumes.append(ligand_pocket_volume)


    return ligand_pocket_volumes

# Blebbsitatin Pocket Simulations CHARMM36
# may want to run with stride=1
specs = {
    'myh11-1br2': {
        'trajectory_paths': '/project/bowmore/j.lotthammer/tica-clustering/myh11-1br2/traj_list.txt',
        # 'trajectory_paths': 'myh11-1br2-oci-traj-list.txt',
        'top_path': '/project/bowmanlab/j.lotthammer/Simulations/myosin/specificity/pps-bleb-isoforms/myh11/input_files3/myh11-1br2-holo-prot-masses.pdb',
        'alignment_pickle': "/project/bowmore/ameller/projects/notebooks/tmp/bleb-pocket-0.5nm-residue-alignment.npy",
        'gene': 'myh11',
        'region_atom_indices_dict_pickle': 'bleb-pocket-region-atom-indices-dict.npy',
        'region_key': 'myh11-1br2',
        'xtal_path': '/project/bowmore/ameller/projects/notebooks/tmp/1YV3.pdb',
        'xtal_name': '1YV3',
        'ligand_name': 'BIT',
        'stride': 1,

    },
    'myh2-5n6a': {
        'trajectory_paths': '/project/bowmore/j.lotthammer/tica-clustering/myh2-5n6a/traj_list.txt',
        # 'trajectory_paths': 'myh2-5n6a-oci-traj-list.txt',
        'top_path': '/project/bowmanlab/j.lotthammer/Simulations/myosin/specificity/pps-bleb-isoforms/myh2/input_files3/myh2-5n6a-holo-prot-masses.pdb',
        'alignment_pickle': "/project/bowmore/ameller/projects/notebooks/tmp/bleb-pocket-0.5nm-residue-alignment.npy",
        'gene': 'myh2',
        'region_atom_indices_dict_pickle': 'bleb-pocket-region-atom-indices-dict.npy',
        'region_key': 'myh2-5n6a',
        'xtal_path': '/project/bowmore/ameller/projects/notebooks/tmp/1YV3.pdb',
        'xtal_name': '1YV3',
        'ligand_name': 'BIT',
        'stride': 1,
    },
    'myh7b-hs-5n6a': {
        'trajectory_paths': '/project/bowmore/j.lotthammer/tica-clustering/myh7b-hs/traj_list.txt',
        'top_path': '/project/bowmanlab/j.lotthammer/Simulations/myosin/myosin-7b/act_site_redo/eq/myh7b-5n6a-holo-prot-masses.pdb',
        'alignment_pickle': "/project/bowmore/ameller/projects/notebooks/tmp/bleb-pocket-0.5nm-residue-alignment.npy",
        'gene': 'myh7b',
        'region_atom_indices_dict_pickle': '/project/bowmore/ameller/projects/notebooks/tmp/bleb-pocket-region-atom-indices-dict.npy',
        'region_key': 'myh7b-5n6a',
        'xtal_path': '/project/bowmore/ameller/projects/notebooks/tmp/1YV3.pdb',
        'xtal_name': '1YV3',
        'ligand_name': 'BIT',
        'stride': 1,
    },
    # 'myh2-1br2': {
    #     'trajectory_paths': '/project/bowmanlab/ameller/tica-clustering/myh2-1br2/trajectory_list.txt',
    #     'top_path': '/project/bowmanlab/ameller/simulations/long-pps-simulations/myh2-1br2/input_files/myh2-1br2-holo-prot-masses.pdb',
    #     'alignment_pickle': "/project/bowmore/ameller/projects/notebooks/tmp/bleb-pocket-0.5nm-residue-alignment.npy",
    #     'gene': 'myh2',
    #     'region_atom_indices_dict_pickle': '/project/bowmore/ameller/projects/notebooks/tmp/bleb-pocket-region-atom-indices-dict.npy',
    #     'region_key': 'myh2-1br2',
    #     'xtal_path': '/project/bowmore/ameller/projects/notebooks/tmp/1YV3.pdb',
    #     'xtal_name': '1YV3',
    #     'ligand_name': 'BIT',
    #     'stride': 1,
    # },
    'myh7-5n6a': {
        'trajectory_paths': '/project/bowmore/j.lotthammer/tica-clustering/myh7-5n6a/traj_list.txt',
        # 'trajectory_paths': 'myh7-5n6a-oci-traj-list.txt',
        'top_path': '/project/bowmanlab/j.lotthammer/Simulations/myosin/specificity/pps-bleb-isoforms/myh7/input_files3/myh7-5n6a-holo-prot-masses.pdb',
        'alignment_pickle': "/project/bowmore/ameller/projects/notebooks/tmp/bleb-pocket-0.5nm-residue-alignment.npy",
        'gene': 'myh7',
        'region_atom_indices_dict_pickle': 'bleb-pocket-region-atom-indices-dict.npy',
        'region_key': 'myh7-5n6a',
        'xtal_path': '/project/bowmore/ameller/projects/notebooks/tmp/1YV3.pdb',
        'xtal_name': '1YV3',
        'ligand_name': 'BIT',
        'stride': 1,
    },
    'myh9-5i4e': {
        'trajectory_paths': '/project/bowmore/j.lotthammer/tica-clustering/myh9-5i4e/traj_list.txt',
        # 'trajectory_paths': 'myh9-5i4e-oci-traj-list.txt',
        'top_path': '/project/bowmanlab/j.lotthammer/Simulations/myosin/specificity/pps-bleb-isoforms/myh9/input_files3/myh9-5I4E-holo-prot-masses.pdb',
        'alignment_pickle': "/project/bowmore/ameller/projects/notebooks/tmp/bleb-pocket-0.5nm-residue-alignment.npy",
        'gene': 'myh9',
        'region_atom_indices_dict_pickle': 'bleb-pocket-region-atom-indices-dict.npy',
        'region_key': 'myh9-5i4e',
        'xtal_path': '/project/bowmore/ameller/projects/notebooks/tmp/1YV3.pdb',
        'xtal_name': '1YV3',
        'ligand_name': 'BIT',
        'stride': 1,
    },
    'myh2-6fsa': {
        'trajectory_paths': '/project/bowmanlab/ameller/tica-clustering/myh2-6fsa/trajectory_list.txt',
        'top_path': '/project/bowmanlab/j.lotthammer/Simulations/myosin/specificity/FASTSpecificPockets-myh2-CK-571-CHARMM/msm/prot_masses.pdb',
        'alignment_pickle': "bleb-fast-isoform-alignment.pickle",
        'gene': 'myh7',
        'region_atom_indices_dict_pickle': 'bleb-pocket-region-atom-indices-dict.npy',
        'region_key': 'myh2-5n6a',
        'xtal_path': '/project/bowmore/ameller/projects/notebooks/tmp/1YV3.pdb',
        'xtal_name': '1YV3',
        'ligand_name': 'BIT',
        'stride': 1,
    },
}

min_rank = 6
min_cluster_size = 3
# finer grid spacing is generally better but leads to memory overhead
grid_spacing = 0.07
# set to twice the grid spacing
# probe_radius = 0.14
probe_radius = 0.14
dist_cutoff = 0.25
# can also be set to None if we are fine with separated pocket clusters
continuity_requirement = 0.15
out_dir = 'subset-trajectory-ligand-pocket-volumes'

for protein, spec in specs.items():
    traj_list = np.loadtxt(spec['trajectory_paths'], dtype=str)
    pocket_volumes = ([
        calculate_pocket(trj_filename, spec['top_path'], spec['xtal_path'],
                         spec['xtal_name'], spec['alignment_pickle'], spec['gene'],
                         spec['region_atom_indices_dict_pickle'],
                         spec['region_key'], spec['ligand_name'],
                         stride=spec['stride'],
                         min_rank=min_rank, min_cluster_size=min_cluster_size,
                         grid_spacing=grid_spacing, probe_radius=probe_radius,
                         dist_cutoff=dist_cutoff,
                         continuity_requirement=continuity_requirement)
        for trj_filename in traj_list
    ])
    out_filename = (f"{out_dir}/{protein}-{spec['xtal_name']}-{spec['ligand_name']}-ligsite-pocket-volumes"
                    f"-stride-{spec['stride']}-min-rank-{min_rank}-"
                    f'min-cluster-size-{min_cluster_size}-grid-spacing-{grid_spacing}'
                    f'-probe-radius-{probe_radius}-'
                    f'max-dist-from-ligand-{dist_cutoff}.h5')
    pocket_vols_to_h5(out_filename, pocket_volumes)