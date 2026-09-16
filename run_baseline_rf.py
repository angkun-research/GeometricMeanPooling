import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

def get_scaffold_split(df, train_frac=0.8):
    """Perform a scaffold split on the Lipophilicity dataset."""
    smiles = df['SMILES'].tolist()
    
    scaffolds = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            scaffolds.append(MurckoScaffold.GetScaffoldForMol(mol))
        else:
            scaffolds.append(None)
            
    scaffold_sets = {}
    for i, scaf in enumerate(scaffolds):
        if scaf is None: continue
        s_smiles = Chem.MolToSmiles(scaf)
        if s_smiles not in scaffold_sets:
            scaffold_sets[s_smiles] = []
        scaffold_sets[s_smiles].append(i)
        
    all_scaffolds = list(scaffold_sets.keys())
    train_scaffs, test_scaffs = train_test_split(all_scaffolds, train_size=train_frac, random_state=42)
    
    train_indices = []
    test_indices = []
    for scaf in train_scaffs:
        train_indices.extend(scaffold_sets[scaf])
    for scaf in test_scaffs:
        test_indices.extend(scaffold_sets[scaf])
        
    non_scaffold_indices = [i for i, s in enumerate(scaffolds) if s is None]
    train_indices.extend(non_scaffold_indices)
    
    return train_indices, test_indices

def smiles_to_fp(smiles, gen):
    """Convert SMILES to Morgan Fingerprint using the provided generator."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2048) # Matches nBits=2048
    # Get fingerprint as a bit vector and convert to numpy array
    fp = gen.GetFingerprint(mol)
    return np.array(fp)

if __name__ == "__main__":
    csv_path = "data/lipophilicity.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        exit(1)

    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    train_idx, test_idx = get_scaffold_split(df)
    print(f"Scaffold split: Train={len(train_idx)}, Test={len(test_idx)}")

    # Create the Morgan Generator once (Modern API)
    # radius=2 corresponds to ECFP4
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    
    print("Generating Morgan Fingerprints...")
    def get_fp_and_label(idx):
        row = df.iloc[idx]
        return smiles_to_fp(row['SMILES'], gen), row['label']

    X_train, y_train = [], []
    for i in train_idx:
        fp, label = get_fp_and_label(i)
        X_train.append(fp)
        y_train.append(label)
    
    X_test, y_test = [], []
    for i in test_idx:
        fp, label = get_fp_and_label(i)
        X_test.append(fp)
        y_test.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print(f"Feature shape: {X_train.shape}")

    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n" + "="*30)
    print(f"Baseline Results (Morgan FP + RF)")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R^2: {r2:.4f}")
    print("="*30)
