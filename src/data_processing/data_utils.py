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

def get_pdb(pdb_id,pdir = ''):
    pdb_list = PDBList()
    pdb_list.retrieve_pdb_file(pdb_id, pdir = '.', file_format = 'mmCif')

def parse_pdb(pdb_path):
    



if __name__ == '__main__':
    parse_pdb('./216l.cif')
