"""
cd "C:\Travail Master\CodePAC\Tools"
py .\building_k.py
"""
import pandas as pd
import numpy as np

# ============================================================
# PARAMÈTRES
# ============================================================

CSV_GEOM = r"C:\Travail Master\CodePAC\Data\geneve_batiments_geom.csv"
CSV_SURF = r"C:\Travail Master\CodePAC\Data\geneve_batiments_surfaces.csv"

SEP = ";"

# Seuils minimaux
MIN_AREA = 20.0
MIN_PERIM = 15.0
MIN_SURFACE_HS = 20.0
MIN_LEVELS = 1

# Bornes plausibles globales pour k
K_MIN = 0.4
K_MAX = 4.0

# Hauteurs d'étage forfaitaires
DEFAULT_HEIGHT_PER_LEVEL = {
    "Habitation": 2.8,
    "Activités": 3.2,
    "Equipement collectif": 3.2,
    "Mixte": 3.2,
}

# Destinations à exclure
EXCLUDED_DESTINATIONS = {
    "Garage privé",
    "Garage",
    "Autre bât. < 20 m2",
    "Autre bât. 20m2 et plus",
    "Installation de climatisation",
    "Instal. tech. élec. SIG",
    "Réservoir",
    "Véranda",
    "Serre",
    "Poulailler",
    "Hangar",
    "Hangar agricole",
    "Dépôt",
    "Bâtiment électricité",
}

EXCLUDED_KEYWORDS = [
    "garage",
    "climatisation",
    "instal.",
    "installation",
    "réservoir",
    "veranda",
    "véranda",
    "serre",
    "poulailler",
    "hangar",
    "dépôt",
]

# Familles brutes autorisées
ALLOWED_CLASSES = {
    "Habitation",
    "Activités",
    "Equipement collectif",
    "Mixte: logements/activités ou équipement collectifs",
}

# Mapping famille brute -> famille simple
FAMILY_MAP = {
    "Habitation": "Habitation",
    "Activités": "Activités",
    "Equipement collectif": "Equipement collectif",
    "Mixte: logements/activités ou équipement collectifs": "Mixte",
}

# Règles de fusion / suppression finales
FINAL_TYPOLOGY_MAP = {
    "Maison individuelle": "Maison individuelle",
    "Maison individuelle grande": "Maison individuelle grande",
    "Maison 2 logements": "Maison 2 logements",
    "Immeuble collectif": "Immeuble collectif",
    "Petit collectif": "Immeuble collectif",
    "Résidentiel atypique": "Immeuble collectif",
    "Grand collectif": "Grand collectif",
    "Résidentiel collectif spécialisé": "Grand collectif",
    "Mixte": "Mixte",
    "Activités": "Activités",
    "Equipement collectif": "Equipement collectif",
    "Maison / petit résidentiel": None,  # suppression
}


# ============================================================
# OUTILS
# ============================================================

def to_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def normalize_text(s):
    if pd.isna(s):
        return ""
    return str(s).strip()


def is_excluded_destination(dest):
    d = normalize_text(dest).lower()
    if d in {x.lower() for x in EXCLUDED_DESTINATIONS}:
        return True
    return any(keyword in d for keyword in EXCLUDED_KEYWORDS)


def choose_height(row):
    """
    Hauteur retenue:
    - HAUTEUR si plausible
    - sinon niveaux * hauteur forfaitaire
    """
    levels = row.get("NIVEAUX_HORSOL", np.nan)
    raw_h = row.get("HAUTEUR", np.nan)
    family = row.get("famille_brute", "")

    if pd.isna(levels) or levels < 1:
        return np.nan

    default_h_per_level = DEFAULT_HEIGHT_PER_LEVEL.get(family, 3.0)
    fallback_h = levels * default_h_per_level

    if pd.notna(raw_h):
        if (1.8 * levels) <= raw_h <= (5.0 * levels):
            return raw_h

    return fallback_h


def classify_subtype(row):
    """
    Sous-typage plus fin pour les bâtiments d'habitation.
    """
    famille = row.get("famille_brute", "")
    dest = normalize_text(row.get("DESTINATION", ""))
    levels = row.get("NIVEAUX_HORSOL", np.nan)
    s_hs = row.get("SURFACE_TOTALE_HS", np.nan)

    if famille != "Habitation":
        return famille

    if pd.isna(levels):
        levels = 0
    if pd.isna(s_hs):
        s_hs = 0

    if dest == "Habitation un logement":
        if levels <= 2 and s_hs < 350:
            return "Maison individuelle"
        elif levels <= 3 and s_hs < 700:
            return "Maison individuelle grande"
        else:
            return "Résidentiel atypique"

    if dest == "Hab. deux logements":
        return "Maison 2 logements"

    if dest == "Hab plusieurs logements":
        if levels <= 2 and s_hs < 800:
            return "Petit collectif"
        elif levels <= 4 and s_hs < 2500:
            return "Immeuble collectif"
        else:
            return "Grand collectif"

    if dest in {"Résidence meublée", "EMS"}:
        return "Résidentiel collectif spécialisé"

    if levels <= 2 and s_hs < 500:
        return "Maison / petit résidentiel"
    elif levels <= 4:
        return "Immeuble collectif"
    else:
        return "Grand collectif"


def iqr_filter(group, col="k"):
    """
    Filtre des outliers par IQR au sein de chaque typologie détaillée.
    """
    if group[col].dropna().shape[0] < 5:
        return group

    q1 = group[col].quantile(0.25)
    q3 = group[col].quantile(0.75)
    iqr = q3 - q1

    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr

    return group[(group[col] >= low) & (group[col] <= high)]


# ============================================================
# 1) CHARGEMENT
# ============================================================

geom = pd.read_csv(CSV_GEOM, sep=SEP, dtype={"EGID": str})
surf = pd.read_csv(CSV_SURF, sep=SEP, dtype={"EGID": str})

geom.columns = [c.strip() for c in geom.columns]
surf.columns = [c.strip() for c in surf.columns]

geom = to_numeric(
    geom,
    ["NIVEAUX_HORSOL", "NIVEAUX_SSOL", "HAUTEUR", "SHAPE.AREA", "SHAPE.LEN"]
)
surf = to_numeric(
    surf,
    ["VOLUME", "SURFACE_TOTALE", "SURFACE_PARTAGE", "SURFACE_TOTALE_HS"]
)

# ============================================================
# 2) DÉDUPLICATION + FUSION
# ============================================================

geom = geom.sort_values(["EGID", "SHAPE.AREA"], ascending=[True, False])
geom = geom.drop_duplicates(subset="EGID", keep="first").copy()

surf = surf.sort_values(["EGID", "SURFACE_TOTALE_HS"], ascending=[True, False])
surf = surf.drop_duplicates(subset="EGID", keep="first").copy()

print(f"Lignes geom après déduplication: {len(geom)}")
print(f"Lignes surf après déduplication: {len(surf)}")

df = geom.merge(surf, on="EGID", how="inner", validate="one_to_one")

print(f"Nombre de lignes après fusion: {len(df)}")

# ============================================================
# 3) FILTRES DE BASE
# ============================================================

for c in ["DESTINATION", "NOMEN_CLASSE", "GENRE", "TYPE", "COMMUNE"]:
    if c in df.columns:
        df[c] = df[c].apply(normalize_text)

df = df[df["NOMEN_CLASSE"].isin(ALLOWED_CLASSES)].copy()
df = df[df["GENRE"].str.lower() != "sous-sol"].copy()
df = df[~df["DESTINATION"].apply(is_excluded_destination)].copy()

df = df[
    (df["NIVEAUX_HORSOL"] >= MIN_LEVELS) &
    (df["SHAPE.AREA"] >= MIN_AREA) &
    (df["SHAPE.LEN"] >= MIN_PERIM) &
    (df["SURFACE_TOTALE_HS"] >= MIN_SURFACE_HS)
].copy()

print(f"Nombre de lignes après filtres de base: {len(df)}")

# ============================================================
# 4) FAMILLE BRUTE + TYPOLOGIE DÉTAILLÉE
# ============================================================

df["famille_brute"] = df["NOMEN_CLASSE"].map(FAMILY_MAP)
df = df[df["famille_brute"].notna()].copy()

df["typologie_detaillee"] = df.apply(classify_subtype, axis=1)

# ============================================================
# 5) HAUTEUR RETENUE
# ============================================================

df["HAUTEUR_RETENUE"] = df.apply(choose_height, axis=1)
df = df[df["HAUTEUR_RETENUE"].notna()].copy()

# ============================================================
# 6) CALCUL Sdep ET k
# ============================================================

df["Sdep"] = df["SHAPE.LEN"] * df["HAUTEUR_RETENUE"] + 2 * df["SHAPE.AREA"]
df["k"] = df["Sdep"] / df["SURFACE_TOTALE_HS"]

df = df[(df["k"] >= K_MIN) & (df["k"] <= K_MAX)].copy()

print(f"Nombre de lignes après filtre global sur k: {len(df)}")

# ============================================================
# 7) FILTRE IQR PAR TYPOLOGIE DÉTAILLÉE
# ============================================================

groups = []
for typ, g in df.groupby("typologie_detaillee"):
    g2 = iqr_filter(g, col="k").copy()
    g2["typologie_detaillee"] = typ
    groups.append(g2)

df_clean = pd.concat(groups, ignore_index=True)

print(f"Nombre de lignes après filtre IQR: {len(df_clean)}")
print("Colonnes disponibles après filtre IQR :", df_clean.columns.tolist())

# ============================================================
# 8) APPLICATION DES FUSIONS FINALES
# ============================================================

df_clean["typologie_finale"] = df_clean["typologie_detaillee"].map(FINAL_TYPOLOGY_MAP)
df_clean = df_clean[df_clean["typologie_finale"].notna()].copy()

print(f"Nombre de lignes après fusion/suppression des catégories faibles: {len(df_clean)}")

# ============================================================
# 9) RÉSUMÉ TYPOLOGIE DÉTAILLÉE
# ============================================================

summary_detail = (
    df_clean.groupby("typologie_detaillee")
    .agg(
        n=("EGID", "count"),
        k_mediane=("k", "median"),
        k_moyenne=("k", "mean"),
        k_q25=("k", lambda s: s.quantile(0.25)),
        k_q75=("k", lambda s: s.quantile(0.75)),
        niveaux_median=("NIVEAUX_HORSOL", "median"),
        surface_hs_mediane=("SURFACE_TOTALE_HS", "median"),
        hauteur_mediane=("HAUTEUR_RETENUE", "median"),
    )
    .sort_values("k_mediane")
    .reset_index()
)

print("\n=== Résumé par typologie détaillée ===")
print(summary_detail)

# ============================================================
# 10) RÉSUMÉ TYPOLOGIE FINALE
# ============================================================

summary_final = (
    df_clean.groupby("typologie_finale")
    .agg(
        n=("EGID", "count"),
        k_mediane=("k", "median"),
        k_moyenne=("k", "mean"),
        k_q25=("k", lambda s: s.quantile(0.25)),
        k_q75=("k", lambda s: s.quantile(0.75)),
        niveaux_median=("NIVEAUX_HORSOL", "median"),
        surface_hs_mediane=("SURFACE_TOTALE_HS", "median"),
        hauteur_mediane=("HAUTEUR_RETENUE", "median"),
    )
    .sort_values("k_mediane")
    .reset_index()
)

print("\n=== Résumé final par typologie exploitable ===")
print(summary_final)

# ============================================================
# 11) EXPORTS
# ============================================================

df_clean.to_csv("geneve_batiments_nettoyes_k.csv", sep=";", index=False)
summary_detail.to_csv("geneve_resume_k_par_typologie_detaillee.csv", sep=";", index=False)
summary_final.to_csv("geneve_resume_k_par_typologie_finale.csv", sep=";", index=False)

print("\nFichiers exportés :")
print("- geneve_batiments_nettoyes_k.csv")
print("- geneve_resume_k_par_typologie_detaillee.csv")
print("- geneve_resume_k_par_typologie_finale.csv")

# ============================================================
# 12) DICTIONNAIRE FINAL PRÊT À COPIER
# ============================================================

k_typ_final = dict(zip(summary_final["typologie_finale"], summary_final["k_mediane"]))

print("\n=== Dictionnaire final des k médians ===")
print(k_typ_final)