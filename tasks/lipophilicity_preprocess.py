import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
import torch
from torch.nn.utils.rnn import pad_sequence
import os
import re


SMILES_TOKEN_PATTERN = re.compile(
    r"\[[^\]]+\]"                 # bracket expressions
    r"|Br|Cl|Si|Se|Na|Li|Al|Ca|Mg|Zn|Cu|Fe|Sn|Ag|Hg"
    r"|%[0-9]{2}"                 # two-digit ring labels
    r"|[A-Z]"                     # one-letter aliphatic atoms
    r"|[bcnops]"                  # aromatic atoms
    r"|[0-9]"                     # one-digit ring labels
    r"|[().=#\$+\-\\/@:]"         # SMILES operators
)

def get_scaffold_split(df, train_frac=0.8):
    """Perform a scaffold split on the Lipophilicity dataset."""
    smiles = df['SMILES'].tolist()
    labels = df['label'].tolist()

    # Compute scaffolds for each molecule
    scaffolds = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            scaffolds.append(MurckoScaffold.GetScaffoldForMol(mol))
        else:
            scaffolds.append(None)

    # Create a mapping from scaffold to molecules
    scaffold_sets = {}
    for i, scaf in enumerate(scaffolds):
        if scaf is None: continue
        s_smiles = Chem.MolToSmiles(scaf)
        if s_smiles not in scaffold_sets:
            scaffold_sets[s_smiles] = []
        scaffold_sets[s_smiles].append(i)

    # Split the scaffolds themselves
    all_scaffolds = list(scaffold_sets.keys())
    train_scaffs, test_scaffs = train_test_split(all_scaffolds, train_size=train_frac, random_state=42)

    train_indices = []
    test_indices = []
    for scaf in train_scaffs:
        train_indices.extend(scaffold_sets[scaf])
    for scaf in test_scaffs:
        test_indices.extend(scaffold_sets[scaf])

    # Handle molecules without scaffolds (put them in training for simplicity)
    non_scaffold_indices = [i for i, s in enumerate(scaffolds) if s is None]
    train_indices.extend(non_scaffold_indices)

    return train_indices, test_indices

# def smiles_to_sequence(smiles, vocab):
#     """Convert a SMILES string to a sequence of integers based on the vocabulary."""
#     return [vocab.get(char, 0) for char in smiles] # 0 is usually <UNK> or padding

def smiles_to_sequence(smiles, vocab):
    return [
        vocab.get(token, vocab["<UNK>"])
        for token in tokenize_smiles(smiles)
    ]

def tokenize_smiles(smiles: str) -> list[str]:
    tokens = SMILES_TOKEN_PATTERN.findall(smiles)

    # Detect malformed or unsupported characters instead of silently dropping them.
    reconstructed = "".join(tokens)
    if reconstructed != smiles:
        raise ValueError(
            f"Could not tokenize SMILES {smiles!r}: "
            f"reconstructed {reconstructed!r}"
        )

    return tokens

# def create_vocab(df):
#     """Create a vocabulary from all characters in the SMILES dataset."""
#     all_chars = set()
#     for s in df['SMILES']:
#         all_chars.update(list(s))

#     # Sort for consistency and add <PAD> at 0
#     sorted_chars = sorted(list(all_chars))
#     vocab = {char: i + 1 for i, char in enumerate(sorted_chars)}
#     vocab['<PAD>'] = 0
#     return vocab
def create_vocab(df):
    all_tokens = set()

    for smiles in df["SMILES"]:
        all_tokens.update(tokenize_smiles(smiles))

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for token in sorted(all_tokens):
        vocab[token] = len(vocab)

    return vocab

def smiles_to_morgan(smiles, radius=2, nBits=2048):
    """Convert a SMILES string to a Morgan fingerprint (numpy array)."""
    from rdkit.Chem import rdFingerprintGenerator
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits, dtype=np.float32)

    # New RDKit API using MorganGenerator to avoid deprecation warnings
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
    fp = gen.GetFingerprint(mol)

    arr = np.zeros((nBits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def get_morgan_fps(df, indices, radius=2, nBits=2048):
    """Generate Morgan fingerprints for a set of indices in the dataframe."""
    from rdkit.Chem import rdFingerprintGenerator
    smiles = df['SMILES'].iloc[indices].tolist()

    # Initialize generator once for efficiency
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)

    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            fps.append(np.zeros(nBits, dtype=np.float32))
        else:
            fp = gen.GetFingerprint(mol)
            arr = np.zeros((nBits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
    return np.array(fps)

def preprocess_data(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. Scaffold Split
    train_idx, test_idx = get_scaffold_split(df)
    print(f"Scaffold split: Train={len(train_idx)}, Test={len(test_idx)}")

    # 2. Vocabulary
    vocab = create_vocab(df)
    print(f"Vocabulary size: {len(vocab)}")

    # 3. Convert to sequences
    def process_row(row):
        seq = smiles_to_sequence(row['SMILES'], vocab)
        return seq, row['label']

    train_data = [process_row(df.iloc[i]) for i in train_idx]
    test_data = [process_row(df.iloc[i]) for i in test_idx]

    return train_data, test_data, vocab

class SMILESDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.samples = [
            (
                torch.tensor(sequence, dtype=torch.long),
                torch.tensor(label, dtype=torch.float32),
            )
            for sequence, label in data
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
        
# keep the length of the sequences in the batch to the maximum length of the batch
def collate_smiles_batch(batch):
    """Pad each batch only to its longest SMILES sequence."""
    sequences, labels = zip(*batch)

    input_ids = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0,  # <PAD>
    )
    attention_mask = input_ids.ne(0)
    labels = torch.stack(labels)

    return input_ids, attention_mask, labels

if __name__ == "__main__":
    csv_path = "data/lipophilicity.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
    else:
        train, test, vocab = preprocess_data(csv_path)
        print("Preprocessing complete.")
        print(f"Size of vocabulary: {len(vocab)}")
        print(f"Example train sample sequence length: {len(train[0][0])}")
        print(f"Example target value: {train[0][1]}")
        # check minimum and max length of sequences
        train_lengths = [len(seq) for seq, _ in train]
        print(f"Train sequence lengths: min={min(train_lengths)}, max={max(train_lengths)}")

        # # Now check with morgan fingerprints
        # X = get_morgan_fps(pd.read_csv(csv_path), list(range(len(pd.read_csv(csv_path)))))
        # y = pd.read_csv(csv_path)['label'].values.astype(np.float32)
        # print(f"Morgan fingerprints shape: {X.shape}, labels shape: {y.shape}")
        # X_lengths = [len(fp) for fp in X]
        # print(f"Morgan fingerprint lengths: min={min(X_lengths)}, max={max(X_lengths)}")
        
        # print(tokenize_smiles("CC(Cl)C(=O)O"))
        # print(tokenize_smiles("C1=CC=[NH+]C=C1"))

        #df = pd.read_csv(csv_path)
        for k in range(5):
            idx = train[k][0]
            label = train[k][1]
            print(f"Train sample {k}: sequence length={len(idx)}, target={label}")
