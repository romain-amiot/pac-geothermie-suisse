from pathlib import Path
import re
import pandas as pd

# chemins
HTML_PATH = "Data/raw/midland_mazout_snippet.html"
OUT_CSV = "Data/processed/midland_mazout_prices.csv"

# lire le fichier html
html = Path(HTML_PATH).read_text(encoding="utf-8", errors="ignore")

# extraire les deux attributs
vals_match = re.search(r'data-preisverlauf-kurse="([^"]+)"', html)
dates_match = re.search(r'data-preisverlauf-categories="([^"]+)"', html)

if not vals_match or not dates_match:
    raise RuntimeError("Données non trouvées dans le fichier HTML.")

vals_str = vals_match.group(1)
dates_str = dates_match.group(1)

# transformer en listes
values = [float(v) for v in vals_str.split(",")]
dates = [d.strip() for d in dates_str.split(",")]

# sécurité
if len(values) != len(dates):
    raise RuntimeError("Nombre de dates différent du nombre de valeurs.")

# dataframe
df = pd.DataFrame({
    "date": dates,
    "price": values
})

# convertir dates
df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")

# trier
df = df.sort_values("date").reset_index(drop=True)

# créer dossier de sortie si besoin
Path("Data/processed").mkdir(parents=True, exist_ok=True)

# sauvegarder CSV
df.to_csv(OUT_CSV, index=False)

print(f"CSV créé : {OUT_CSV} ({len(df)} lignes)")