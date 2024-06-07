import numpy as np
from rdkit import Chem
import torch
import pickle
import sys
sys.path.append('..')
num_atom_feat = 34
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom,explicit_H=False,use_chirality=True):
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']  # 10-dim
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3,
                              Chem.rdchem.HybridizationType.SP3D,
                              Chem.rdchem.HybridizationType.SP3D2,
                              'other']   # 6-dim
    results = one_of_k_encoding_unk(atom.GetSymbol(),symbol) + \
                  one_of_k_encoding(atom.GetDegree(),degree) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]  # 10+7+2+6+1=26

    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])   # 26+5=31
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 31+3 =34
    return results

def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)+np.eye(adjacency.shape[0])

def mol_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        raise RuntimeError("SMILES cannot been parsed!")
    atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    adj_matrix = adjacent_matrix(mol)
    return atom_feat, adj_matrix

def get_pretrain_prot(prot2id, protein, max_len=1000):
    prot_idx = []
    for i in range(len(protein)-2):
        prot_idx.append(prot2id.get(protein[i:(i+3)], len(id2prot)+1))
    return prot_idx

def get_pretrain_smi(smi2id, smi, max_len=100):
    smi_idx = [smi2id.get(i, len(smi2id)+1) for i in smi]
    return smi_idx

if __name__ == "__main__":
    from utils.word2vec import seq_to_kmers, get_protein_embedding
    from gensim.models import Word2Vec
    with open(f"../data/DrugBank2021.csv","r") as f:
        data_list = f.read().strip().split('\n')

    #data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)

    id2smi, smi2id, smi_embed = np.load('../data/pretrain_embed/smi2vec.npy', allow_pickle=True)
    id2prot, prot2id, pro_embed = np.load('../data/pretrain_embed/prot2vec.npy', allow_pickle=True)

    drug1, target1, compounds, adjacencies,proteins,interactions,smi_ids,prot_ids = [], [], [], [], [], [],[],[]
    model = Word2Vec.load("../data/pretrain_embed/word2vec_30.model")
    for no, data in enumerate(data_list):
        print('/'.join(map(str, [no + 1, N])))
        #_, _, smiles, sequence, interaction = data.strip().split(' ')
        _, _, smiles, sequence, interaction = data.strip().split(' ')
        try:
            smi_id = get_pretrain_smi(smi2id, smiles)
            smi_ids.append(torch.LongTensor(smi_id))
            prot_id = get_pretrain_prot(prot2id, sequence)
            prot_ids.append(torch.LongTensor(prot_id))
            atom_feature, adj = mol_features(smiles)
            protein_embedding = get_protein_embedding(model, seq_to_kmers(sequence))
            label = np.array(interaction, dtype=np.float32)
            atom_feature = torch.FloatTensor(atom_feature)
            adj = torch.FloatTensor(adj)
            protein = torch.FloatTensor(protein_embedding)
            label = torch.LongTensor(label)
            compounds.append(atom_feature)
            adjacencies.append(adj)
            proteins.append(protein)
            interactions.append(label)

        except:
            print('Error:', no)
            continue

    dataset = list(zip(compounds, adjacencies, proteins, interactions, smi_ids, prot_ids))
    with open(f"../data/DrugBank2021.pickle", "wb") as f:
        pickle.dump(dataset, f)
    print('The preprocess of dataset has finished!')