import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski


DESCRIPTOR_FILE = "data/processed/drug_descriptors.csv"
SMILES_FILE = "data/processed/drug_smiles.csv"


missing_drugs = {
    119875: "[NH3][Pt]([NH3])(Cl)Cl",
    753082: "CCCS(=O)(=O)NC1=C(C(=C(C=C1)F)C(=O)C2=CNC3=C2C=C(C=N3)C4=CC=C(C=C4)Cl)F"
}


def calculate_descriptors(nsc, smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            f"Could not parse SMILES for NSC {nsc}"
        )

    return {
        "NSC": nsc,
        "MolWt": Descriptors.MolWt(mol),
        "ExactMolWt": Descriptors.ExactMolWt(mol),
        "MolLogP": Crippen.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Lipinski.NumHDonors(mol),
        "NumHAcceptors": Lipinski.NumHAcceptors(mol),
        "NumRotatableBonds": Lipinski.NumRotatableBonds(mol),
        "HeavyAtomCount": Lipinski.HeavyAtomCount(mol),
        "RingCount": Lipinski.RingCount(mol),
        "NumAromaticRings": Lipinski.NumAromaticRings(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "MolMR": Crippen.MolMR(mol)
    }


# Load existing files

descriptors = pd.read_csv(
    DESCRIPTOR_FILE,
    low_memory=False
)

smiles_df = pd.read_csv(
    SMILES_FILE,
    low_memory=False
)


# Normalize NSC IDs

descriptors["NSC"] = pd.to_numeric(
    descriptors["NSC"],
    errors="coerce"
).astype("Int64")

smiles_df["NSC"] = pd.to_numeric(
    smiles_df["NSC"],
    errors="coerce"
).astype("Int64")


new_descriptors = []


for nsc, smiles in missing_drugs.items():

    print(f"Processing NSC {nsc}...")

    # Add SMILES
    if nsc not in set(smiles_df["NSC"]):

        smiles_df = pd.concat(
            [
                smiles_df,
                pd.DataFrame(
                    [{
                        "NSC": nsc,
                        "SMILES": smiles
                    }]
                )
            ],
            ignore_index=True
        )

    # Add descriptors
    if nsc not in set(descriptors["NSC"]):

        result = calculate_descriptors(
            nsc,
            smiles
        )

        new_descriptors.append(result)


# Add new descriptor rows

if new_descriptors:

    descriptors = pd.concat(
        [
            descriptors,
            pd.DataFrame(new_descriptors)
        ],
        ignore_index=True
    )


# Sort

descriptors = descriptors.sort_values(
    "NSC"
).reset_index(drop=True)

smiles_df = smiles_df.sort_values(
    "NSC"
).reset_index(drop=True)


# Save

descriptors.to_csv(
    DESCRIPTOR_FILE,
    index=False
)

smiles_df.to_csv(
    SMILES_FILE,
    index=False
)


print("\nUpdated files successfully!")

print(
    "Drug SMILES:",
    smiles_df.shape
)

print(
    "Drug descriptors:",
    descriptors.shape
)

print("\nAdded drugs:")

print(
    descriptors[
        descriptors["NSC"].isin(
            [119875, 753082]
        )
    ].to_string(index=False)
)
