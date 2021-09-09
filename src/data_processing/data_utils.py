import json
from Bio.PDB import *
import Bio
import numpy as np
import matplotlib
import tqdm
from multiprocessing import Pool
import torch
import os

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
    try:
        struct = parser.get_structure(pdb_id, pdb_file)
    except:
        return None
    for chain in struct.get_chains():
        if chain.id == chain_id:
            for res in chain.get_residues():
                if Polypeptide.is_aa(res, standard = True):
                    try:
                        res['CA'].get_vector()
                        res['N'].get_vector()
                        res['C'].get_vector()
                    except:
                        continue
                    Ca_dict[seq_len] = res['CA'].get_vector().get_array()
                    res_dict[seq_len] = Polypeptide.three_to_one(res.get_resname())
                    atom_dict[seq_len] = dict()
                else:
                    continue
                atom_dict[seq_len] = res
                seq_len += 1
            break

    Ca_dir = Ca_direction(Ca_dict)
    Sc_dir = side_chain_direction(atom_dict)

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
    return res_dict, Ca_dict, angle_dict, dist_mtx, Ca_dir, Sc_dir

def aa2onehot(seq):
    aa2index = {
        'A':0,
        'C':1,
        'D':2,
        'E':3,
        'F':4,
        'G':5,
        'H':6,
        'I':7,
        'K':8,
        'L':9,
        'M':10,
        'N':11,
        'P':12,
        'Q':13,
        'R':14,
        'S':15,
        'T':16,
        'V':17,
        'W':18,
        'Y':19
    }
    length = len(seq)
    onehot = np.zeros((length,20))
    for i, aa in enumerate(seq):
        idx = aa2index[aa]
        onehot[i][idx] = 1
    return onehot

def unit_vec(vec):
    '''
    compute the unit vector in the direction of the input vector
    '''
    return(np.divide(vec, np.linalg.norm(vec)))

def Ca_direction(Ca_dict):
    seq_len = len(Ca_dict)
    fw_vec = []
    bw_vec = []
    bw_vec.append(np.zeros(3))
    for i in Ca_dict:
        if i < seq_len - 1:
            fw_vec.append(unit_vec(Ca_dict[i+1] - Ca_dict[i]))
        if i > 0:
            bw_vec.append(unit_vec(Ca_dict[i-1] - Ca_dict[i]))
    fw_vec.append(np.zeros(3))
    
    Ca_direction = np.concatenate((fw_vec, bw_vec), axis = -1)
    
    return Ca_direction

def side_chain_direction(atom_dict):
    Sc_dir = []
    for i in atom_dict:
        C = atom_dict[i]['C'].get_vector().get_array()
        Ca = atom_dict[i]['CA'].get_vector().get_array()
        N = atom_dict[i]['N'].get_vector().get_array()

        c, n = unit_vec(C - Ca), unit_vec(N - Ca)
        bisector = unit_vec(c + n)
        perp = unit_vec(np.cross(c, n))
        vec = -bisector * np.sqrt(1/3) - perp * np.sqrt(2/3)
        Sc_dir.append(vec)
    Sc_dir  = np.array(Sc_dir)
    return Sc_dir

def save_feat(chain_info):
    save_path = '../../data/features/'
    data_dir = '../../data/PDB/'
    ent = chain_info.split('.')[0]
    chain_id = chain_info.split('.')[1]
    feats = cif_reader(data_dir + ent + '.cif', chain_id)
    if feats == None:
        return
    seq = ''
    Ca_coord = []
    angles = []
    for i in feats[0]:
        seq += feats[0][i]
        Ca_coord.append(feats[1][i])
        angles.append(feats[2][i])
    Ca_coord = torch.from_numpy(np.array(Ca_coord)).half()
    one_hot = torch.from_numpy(aa2onehot(seq)).half()
    angles = torch.from_numpy(np.array(angles)).half()
    dist_mtx = torch.from_numpy(feats[3]).half()
    vec_feat = torch.cat([torch.from_numpy(feats[4]).half(), torch.from_numpy(feats[5]).half()], dim = -1)

    assert one_hot.size(0) <= 500, 'sequence too long'

    feat_dict = {'seq':one_hot, 'Ca_coord':Ca_coord, 'torsion_angles':angles, 'distance':dist_mtx, 'vec_feats':vec_feat}
    
    torch.save(feat_dict, save_path+ent+'-'+chain_id)

def feature_gen():
    chain_dict = read_json('../../data/chain_set_splits.json')

    pool = Pool(processes = 24)
    result_list_tqdm = []
    for result in tqdm.tqdm(pool.imap_unordered(save_feat, chain_dict['validation']), total=len(chain_dict['validation'])):
        result_list_tqdm.append(result)

def data_check():
    data_path = '../../data/features/' 
    chain_dict = read_json('../../data/chain_set_splits.json')
    with open('../../data/train.txt', 'w') as f:
        for chain_id in chain_dict['train']:
            feat_path = data_path + '-'.join(chain_id.split('.'))
            if os.path.exists(feat_path):
                f.write('-'.join(chain_id.split('.')) + '\n')
    
def rot_mtx(alpha, beta, gamma):
    z = np.array(
        [[np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha), np.cos(alpha), 0],
        [0,0,1]])
    y = np.array(
        [[np.cos(beta), 0, np.sin(beta)],
        [0,1,0],
        [-np.sin(beta), 0, np.cos(beta)]])
    x = np.array(
        [[1,0,0],
        [0, np.cos(gamma), -np.sin(gamma)],
        [0, np.sin(gamma), np.cos(gamma)]])
    return z @ y @ x



if __name__ == '__main__':
    feature_gen()

