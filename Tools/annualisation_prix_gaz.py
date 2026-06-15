import sys
from pathlib import Path
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: py Tools/nettoyage_prix_gaz.py <chemin_csv>")
    sys.exit(1)

input_file = Path(sys.argv[1])

df = pd.read_csv(
    input_file,
    sep=";",
    decimal=",",
    header=None,
    names=["year_float", "price_ht"],
    engine="python",
)

df["year_float"] = pd.to_numeric(df["year_float"], errors="coerce")
df["price_ht"] = pd.to_numeric(df["price_ht"], errors="coerce")
df = df.dropna(subset=["year_float", "price_ht"]).copy()

df["year"] = df["year_float"].astype(int)
df["price_ttc"] = df["price_ht"] * 1.081

df_annual = (
    df.groupby("year", as_index=False)[["price_ht", "price_ttc"]]
    .mean()
)

output_file = Path("Data/prix_gaz_ttc_annuel.csv")
df_annual.to_csv(output_file, index=False)

print("Fichier annuel créé :", output_file.resolve())
print(df_annual.head())
