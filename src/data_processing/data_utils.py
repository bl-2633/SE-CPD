import json
from Bio.PDB import *


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

if __name__ == '__main__':
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
    
