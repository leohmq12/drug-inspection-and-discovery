from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import IPythonConsole, MolstoGridImage
import numpy as np
import tensorflow as tf
from tensorflow import keras

RDLogger.DisableLog("rdApp.*")

csv_filepath = tf.keras.utils.get_file("qm9.csv", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv")

data = []

#Loop through rows and data of the imported .csv file
with open(csv_filepath, 'r') as f:
    for line in f.readlines()[1:]:
        data.append(line.split(',')[1])

#Generative AI to extract data to generate a new molecule
smiles = data[100]
print("SMILES:", smiles)
molecule = Chem.MolFromSmiles(smiles)
print("Number of heavy atoms:", molecule.GetNumHeavyAtoms())

#Helper functions to facilitate the model to understand the molecule
atom_mapping = {
    'C': 0, 
    0 : 'C',
    'N' : 1,
    1 : 'N',
    'O' : 2,
    2 : 'O',
    'F' : 3,
    3 : 'F',
}

#Using numbers to moderate the mapping for model to understand
bond_mapping = {
    "SINGLE": 0,
    0: Chem.BondType.SINGLE,
    "DOUBLE": 1,
    1: Chem.BondType.DOUBLE,
    "TRIPLE": 2,
    2: Chem.BondType.TRIPLE,
    "AROMATIC": 3,
    3: Chem.BondType.AROMATIC,
}
#This helps the string identifier mapping the bonds to a number
# and rdkit representation of the contents of the bond types

numofatoms = 9 #Maximum number
numofatomtypes = 4 + 1 
numofbondtypes = 4 + 1
latentdim = 64 #Latent Space size

# The + 1 in above two functions is to facilitate the missing bonds and atom types
# to not lose any information

#Smiles Conversion to Visual Representation
def smiles_to_graphs(smiles):
    #conversion
    molecule = Chem.MolFromSmiles(smiles)

#bond type encoding between atoms
adjacency = np.zeros((numofbondtypes, numofatoms, numofatoms ), 'float32')
#atom types encoding
features = np.zeros ((numofatoms, numofatomtypes), 'float32')

#looping each atom through the molecule

for atom in molecule.GetAtoms():
    #Index grabbing
    j = atom.GetIdx()
    #Item type grabbing
    atomtype = atom_mapping[atom.GetSymbol()]
    #grabbing indexes with features and inserting 1 in the atom type columns
    #so the new generated row would look like this [0,0,1,0,0]
    features[j]= np.eye(numofatomtypes)[atomtype]

# Using  loops to go through the atoms next to the ones we passed through
for neighbor in atom.GetNeighbors():
    #index grabbing of neighbor atoms
    k = neighbor.GetIdx()
    #capturing bonds between atoms and neighbor atoms
    bond = molecule.GetBondBetweenAtoms(j,k)
    #String name conversion of rdkit to get integer index
    bondtypeindex = bond_mapping[bond.GetBondType().name]
    #Replacing the value to 1 where bonds meet with each other
    adjacency[bondtypeindex, [j,k], [k,j]] = 1
    #Summing columns of atom dimensions and if the sum is zero
    #in any column then putting 1 in that column indicating bond absence
    adjacency[-1,np.sum(adjacency,axis=0)==0] = 1
    #if there is no atom after summing up the columns then go to last row
    #to set the index to 1 pointing that there is no atom
    features[np.where(np.sum(features, axis=1) == 0)[0], -1] = 1
    #future usage tensors grabbing
            
return adjacency, features

def unpack_graph(graph):
    #graph unpacking
    adjacency, features = graph
    #RWMol object editing
    molecule = Chem.RWMol()

#Non-atoms removal and atoms with no bonds
index_keeping = np.where((np.argmax(features, axis=1) != numofatomtypes-1) &(np.sum(adjacency[:-1], axis=(0,1)) != 0))[0]
#rows and colums relevancy that correspond to relevant bonds and atoms
features = features[index_keeping]
adjacency = adjacency[:, index_keeping,:][:,:,index_keeping]

#index looping through each atom to retrieve the string symbol
for atomtypeindex in np.argmax(features, axis = 1):
    atom = Chem.Atom(atom_mapping[atomtypeindex])
    _ = molecule.AddAtom(atom)

#Looping through each bond and adding it to the molecule
#emphasizing array in top triangle and ignoring duplicates
#using tuples with indices where 1 is present and unpacked
(bondsjk, atomj, atomk) = np.where(np.triu(adjacency)==1)

#condition checking before adding bonds inbetween atoms
for (bondsjk, atomj, atomk) in zip(bondsjk, atomj, atomk):
    #if atoms are same
    if atomj == atomk or bondsjk == numofbondtypes - 1:
        continue
    bondtype = bond_mapping[bondsjk]
    molecule.AddBond(int(atomj), int(atomk), bondtype)

#molecule checks if it is breaking any chemistry rules
flag = Chem.SanitizeMol(molecule, catchErrors=True)
#if sanitization fails
if flag != Chem.SanitizeFlags.SANITIZE_NONE:
    return None
return molecule

#array generation for training data
adjacencymatrix, featuresmatrix = [], []

#looping through data by appending adjacency tensor to adjacency array
#and features tensor to features array

for smiles in data[::10]:
    adjacency, featuers = smiles_to_graphs(smiles)
    adjacencymatrix.append(adjacency)
    featuresmatrix.append(features)
    #converting arrays into numpy arrays
    adjacencymatrix = np.array(adjacencymatrix)
    featuresmatrix = np.array(featuresmatrix)
    #printing the values
    print("adjacencymatrix.shape = ", adjacencymatrix.shape)
    print("featuresmatrix.shape = ", featuresmatrix.shape)
