from __future__ import annotations

import pandas as pd
from pathlib import Path


def clean_swiss_postcodes(
    input_csv: str,
    output_csv: str,
    sep: str = ";",
) -> pd.DataFrame:
    """
    Nettoie le CSV officiel des localités suisses et produit une table agrégée
    par code postal (PLZ4), avec coordonnées pondérées par Adressenanteil.

    Sortie :
        postcode | e | n | canton | locality

    Règles :
    - postcode = PLZ4 sur 4 chiffres
    - e, n = moyenne pondérée par Adressenanteil
    - canton, locality = ligne dominante (plus grand poids)
    """

    input_path = Path(input_csv)
    output_path = Path(output_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    df = pd.read_csv(input_path, sep=sep, dtype=str)

    required_cols = [
        "PLZ4",
        "Ortschaftsname",
        "Kantonskürzel",
        "Adressenanteil",
        "E",
        "N",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le CSV : {missing}\n"
            f"Colonnes trouvées : {list(df.columns)}"
        )

    # Standardisation des champs texte
    df["PLZ4"] = df["PLZ4"].astype(str).str.strip().str.zfill(4)
    df["Ortschaftsname"] = df["Ortschaftsname"].astype(str).str.strip()
    df["Kantonskürzel"] = df["Kantonskürzel"].astype(str).str.strip()

    # Nettoyage de Adressenanteil, ex: "84.138 %" -> 84.138
    df["Adressenanteil_num"] = (
        df["Adressenanteil"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    df["Adressenanteil_num"] = pd.to_numeric(
        df["Adressenanteil_num"], errors="coerce"
    )

    # Nettoyage des coordonnées
    df["E_num"] = pd.to_numeric(
        df["E"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    df["N_num"] = pd.to_numeric(
        df["N"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )

    # Supprime les lignes inutilisables
    df = df.dropna(subset=["PLZ4", "Adressenanteil_num", "E_num", "N_num"]).copy()

    if df.empty:
        raise ValueError("Aucune ligne exploitable après nettoyage.")

    # Poids entre 0 et 1
    df["weight"] = df["Adressenanteil_num"] / 100.0

    # Coordonnées pondérées
    df["E_weighted"] = df["E_num"] * df["weight"]
    df["N_weighted"] = df["N_num"] * df["weight"]

    # Agrégation pondérée par postcode
    grouped = (
        df.groupby("PLZ4", as_index=False)
        .agg(
            weight_sum=("weight", "sum"),
            e_weighted_sum=("E_weighted", "sum"),
            n_weighted_sum=("N_weighted", "sum"),
        )
    )

    # Sécurité : évite division par zéro
    grouped = grouped[grouped["weight_sum"] > 0].copy()

    grouped["e"] = grouped["e_weighted_sum"] / grouped["weight_sum"]
    grouped["n"] = grouped["n_weighted_sum"] / grouped["weight_sum"]

    # Locality + canton dominants = ligne au poids max pour chaque postcode
    dominant_rows = (
        df.sort_values(
            by=["PLZ4", "weight", "Ortschaftsname"],
            ascending=[True, False, True],
        )
        .drop_duplicates(subset=["PLZ4"], keep="first")
        .loc[:, ["PLZ4", "Kantonskürzel", "Ortschaftsname"]]
        .rename(
            columns={
                "PLZ4": "postcode",
                "Kantonskürzel": "canton",
                "Ortschaftsname": "locality",
            }
        )
    )

    result = (
        grouped.rename(columns={"PLZ4": "postcode"})
        .merge(dominant_rows, on="postcode", how="left")
        .loc[:, ["postcode", "e", "n", "canton", "locality"]]
        .sort_values("postcode")
        .reset_index(drop=True)
    )

    # Arrondi léger pour lisibilité
    result["e"] = result["e"].round(3)
    result["n"] = result["n"].round(3)

    # Création du dossier de sortie si nécessaire
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, sep=";", index=False, encoding="utf-8")

    return result


if __name__ == "__main__":
    # À adapter selon ton fichier réel
    input_file = "localites_suisse.csv"
    output_file = "ch_postcodes_weighted.csv"

    try:
        result_df = clean_swiss_postcodes(input_file, output_file)
        print(f"Fichier généré : {output_file}")
        print(f"Nombre de codes postaux agrégés : {len(result_df)}")
        print("\nAperçu :")
        print(result_df.head(10).to_string(index=False))
    except Exception as e:
        print(f"Erreur : {e}")