import json
from Bio.PDB import *
import Bio
import numpy as np
import matplotlib
import tqdm
from multiprocessing import Pool

def read_json(json_path):
    data_dict = json.loads(open(json_path,'r').read())
    return data_dict

def read_jsonl(jsonl_path):
    f = open(jsonl_path,'r').readlines()
    for line in f:
        print(line)
        break

def get_pdb(pdb_id):
    pdb_list = PDBList()
    pdb_list.retrieve_pdb_file(pdb_id, pdir = '../../data/PDB/', file_format = 'mmCif')

def download_datasets():
    pdb_list = read_json('../../data/chain_set_splits.json')
    for chain in pdb_list['train']:
        ent_id = chain.split('.')[0]
        get_pdb(ent_id)
    for chain in pdb_list['test']:
        ent_id = chain.split('.')[0]
        get_pdb(ent_id)
    for chain in pdb_list['validation']:
        ent_id = chain.split('.')[0]
        get_pdb(ent_id)


def backbone_angles(pre_atom, atom, next_atom):
    if pre_atom == None:
        phi = 0
        psi = calc_dihedral(atom['N'].get_vector(), atom['CA'].get_vector(), atom['C'].get_vector(),next_atom['N'].get_vector())
        omega = calc_dihedral(atom['CA'].get_vector(), atom['C'].get_vector(), next_atom['N'].get_vector(), next_atom['CA'].get_vector())
    elif next_atom == None:
        phi = calc_dihedral(pre_atom['C'].get_vector(), atom['N'].get_vector(), atom['CA'].get_vector(), atom['C'].get_vector())
        psi = 0
        omega = 0
    else:
        phi = calc_dihedral(pre_atom['C'].get_vector(), atom['N'].get_vector(), atom['CA'].get_vector(), atom['C'].get_vector())
        psi = calc_dihedral(atom['N'].get_vector(), atom['CA'].get_vector(), atom['C'].get_vector(),next_atom['N'].get_vector())
        omega = calc_dihedral(atom['CA'].get_vector(), atom['C'].get_vector(), next_atom['N'].get_vector(), next_atom['CA'].get_vector())
    
    return (phi, psi, omega) 

def calc_dihedral(v1, v2, v3, v4):
    angle = Bio.PDB.vectors.calc_dihedral(v1,v2,v3,v4)
    return angle

def calc_dist(coord1, coord2):
    diff = coord1 - coord2
    return np.linalg.norm(diff)

def calc_contact(Ca_dict):
    seq_len = len(Ca_dict)
    distance_mtx = np.zeros((seq_len,seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            dist = calc_dist(Ca_dict[i],Ca_dict[j])
            distance_mtx[i][j] = dist
    return distance_mtx



def cif_reader(pdb_file,chain_id):
    atom_dict = dict()
    res_dict = dict()
    Ca_dict = dict()
    seq_len = 0 
    
    pdb_id = pdb_file.split('/')[-1].split('.')[0]
    parser = MMCIFParser(QUIET=True)
    struct = parser.get_structure(pdb_id, pdb_file)
    
    for chain in struct.get_chains():
        if chain.id == chain_id:
            for res in chain.get_residues():
                if Polypeptide.is_aa(res, standard = True):
                    try:
                        Ca_dict[seq_len] = res['CA'].get_vector().get_array()
                        res['N'].get_vector()
                        res['C'].get_vector()
                    except:
                        continue
                    res_dict[seq_len] = Polypeptide.three_to_one(res.get_resname())
                    atom_dict[seq_len] = dict()
                else:
                    continue
                atom_dict[seq_len] = res
                seq_len += 1
    
    angle_dict = dict()
    for i in range(0,seq_len):
        pre_res = None
        current_res = None
        next_res = None
        current_res = atom_dict[i]
        if i > 0:
            pre_res = atom_dict[i-1]
        if i < (seq_len - 1):
            next_res = atom_dict[i+1]
        angles = backbone_angles(pre_res, current_res, next_res)
        angle_dict[i] = angles
    
    dist_mtx = calc_contact(Ca_dict)
    return res_dict, Ca_dict, angle_dict, dist_mtx

def aa2onehot(aa):
    return

def save_feat(chain_info):
    data_dir = '../../data/PDB/'
    ent = chain_info.split('.')[0]
    chain_id = chain_info.split('.')[1]
    feats = cif_reader(data_dir + ent + '.cif', chain_id)

def feature_gen():
    chain_dict = read_json('../../data/chain_set_splits.json')
    with Pool(processes = 15) as p:
        list(tqdm.tqdm(p.imap(save_feat, chain_dict['train']), total =len(chain_dict['train'])))

    return


if __name__ == '__main__':
    feature_gen()
    
