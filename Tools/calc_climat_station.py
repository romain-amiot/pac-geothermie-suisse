# Tools/calc_climat_station.py
# commande pour lancer le script :
# cd "C:\Travail Master\CodePAC"
# python Tools\calc_climat_station.py

from __future__ import annotations

import csv
import datetime as dt
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


# =========================================================
# PARAMÈTRES
# =========================================================

T_BASE_HEAT = 20.0   # °C pour HDD
T_BASE_COOL = 24.0   # °C pour clim

WINTER_MONTHS = {11, 12, 1, 2, 3}

# Nouveau défaut demandé : 0.5% (percentile froid)
DEFAULT_COLD_PERCENTILE = 0.5


# =========================================================
# STRUCTURE DE SORTIE
# =========================================================

@dataclass(frozen=True)
class StationClimateMetrics:
    station: str
    postcode: str

    hdd_mean: float

    t_min_abs: float
    t_design_percentile: float

    cooling_hours_mean: float
    cooling_degree_hours_mean: float
    cooling_mode: str

    # meta
    temp_cols_used: str            # ex: "HDD=tre200h0, DESIGN=tre200hn"
    data_granularity: str
    years_used: list[int]
    files_used: list[str]


# =========================================================
# OUTILS
# =========================================================

def parse_timestamp(ts: str) -> dt.datetime:
    return dt.datetime.strptime(ts, "%d.%m.%Y %H:%M")


def safe_float(x: str) -> float | None:
    if x is None:
        return None
    s = str(x).strip().replace(",", ".")
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def detect_temp_columns(fieldnames: list[str]) -> tuple[str, str]:
    """
    Retourne (temp_col_for_hdd_and_cool, temp_col_for_design_min).

    Règles :
    - Si horaire :
        - HDD/clim : tre200h0 (moyenne horaire) si dispo, sinon fallback
        - Design   : tre200hn (min horaire) si dispo, sinon fallback
    - Si journalier :
        - HDD/clim : tre200d0 si dispo
        - Design   : tre200dn (min journalier) si dispo, sinon fallback

    Fallback : première colonne qui commence par 'tre' ou contient 'temp'
    """
    names = fieldnames

    # candidats "propres"
    hourly_mean = "tre200h0" if "tre200h0" in names else None
    hourly_min = "tre200hn" if "tre200hn" in names else None

    daily_mean = "tre200d0" if "tre200d0" in names else None
    daily_min = "tre200dn" if "tre200dn" in names else None

    # fallback général : première colonne température trouvée
    fallback = None
    for n in names:
        low = n.lower()
        if low.startswith("tre") or "temp" in low:
            fallback = n
            break
    if fallback is None:
        raise ValueError("Colonne température introuvable (aucune colonne tre*/temp*).")

    # on ne sait pas encore si c'est horaire ou journalier, on choisit le meilleur match possible
    temp_hdd = hourly_mean or daily_mean or fallback
    temp_design = hourly_min or daily_min or fallback

    return temp_hdd, temp_design


def infer_granularity(times: list[dt.datetime]) -> str:
    if len(times) < 3:
        return "daily"

    deltas = [
        (b - a).total_seconds() / 3600
        for a, b in zip(times[:-1], times[1:])
        if (b - a).total_seconds() > 0
    ]
    if not deltas:
        return "daily"

    deltas_sorted = sorted(deltas)
    median = deltas_sorted[len(deltas_sorted) // 2]
    return "hourly" if median <= 2 else "daily"


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("Liste vide pour percentile().")

    values = sorted(values)
    k = (len(values) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (k - f) * (values[c] - values[f])


# =========================================================
# LECTURE MULTI-FICHIERS + DÉDUP
# =========================================================

def read_many_csv(
    csv_paths: list[Path],
) -> tuple[list[dt.datetime], list[float], list[float], str, str, list[str]]:
    """
    Lit plusieurs CSV MeteoSwiss et retourne :
      - times (triés)
      - temps_mean (pour HDD/clim)
      - temps_min  (pour T_design)
      - granularity ("hourly" / "daily") inférée sur la série complète
      - temp_cols_used_str (meta)
      - files_used (noms de fichiers)

    Déduplication :
      - Si un timestamp apparaît plusieurs fois, on garde la dernière occurrence lue.
    """
    by_ts: dict[dt.datetime, tuple[float, float]] = {}
    temp_col_hdd = None
    temp_col_design = None
    files_used: list[str] = []

    for path in csv_paths:
        files_used.append(path.name)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            if reader.fieldnames is None:
                continue

            col_hdd, col_design = detect_temp_columns(reader.fieldnames)
            # garder pour méta : on prend les colonnes du premier fichier
            if temp_col_hdd is None:
                temp_col_hdd = col_hdd
            if temp_col_design is None:
                temp_col_design = col_design

            for row in reader:
                ts_raw = row.get("reference_timestamp")
                if not ts_raw:
                    continue
                try:
                    ts = parse_timestamp(ts_raw)
                except Exception:
                    continue

                t_mean = safe_float(row.get(col_hdd))
                t_min = safe_float(row.get(col_design))

                # on exige au moins une valeur moyenne (sinon impossible HDD/clim)
                if t_mean is None:
                    continue
                # pour design, si pas de min dispo on fallback sur mean
                if t_min is None:
                    t_min = t_mean

                by_ts[ts] = (t_mean, t_min)

    if not by_ts:
        raise ValueError("Aucune donnée valide trouvée dans les fichiers (timestamp + température).")

    times_sorted = sorted(by_ts.keys())
    temps_mean = [by_ts[t][0] for t in times_sorted]
    temps_min = [by_ts[t][1] for t in times_sorted]

    granularity = infer_granularity(times_sorted)

    cols_str = f"HDD/CLIM={temp_col_hdd}, DESIGN_MIN={temp_col_design}"
    return times_sorted, temps_mean, temps_min, granularity, cols_str, files_used


def resolve_csv_list(data_dir: Path, user_input: str) -> list[Path]:
    """
    user_input peut être :
    - un nom unique : "Geneve_hourly_2000.csv"
    - une liste séparée par virgules : "Geneve_hourly_2000.csv,Geneve_hourly_2010.csv"
    - un préfixe : "Geneve_hourly_" (on prend tous les csv qui commencent par ce préfixe)
    - un pattern glob : "Geneve_hourly_*.csv"
    """
    s = user_input.strip()
    if not s:
        raise ValueError("Entrée fichier vide.")

    # Liste comma-separated
    if "," in s:
        paths = []
        for part in [p.strip() for p in s.split(",") if p.strip()]:
            pth = data_dir / part
            if not pth.exists():
                raise FileNotFoundError(f"Fichier introuvable : {pth}")
            paths.append(pth)
        return paths

    # Pattern glob
    if "*" in s or "?" in s or "[" in s:
        paths = sorted(data_dir.glob(s))
        if not paths:
            raise FileNotFoundError(f"Aucun fichier ne correspond au pattern : {s}")
        return paths

    # Préfixe (pas d'extension) => tous les csv qui commencent par le préfixe
    if not s.lower().endswith(".csv"):
        paths = sorted(data_dir.glob(f"{s}*.csv"))
        if not paths:
            raise FileNotFoundError(f"Aucun CSV trouvé avec le préfixe : {s}")
        return paths

    # Fichier unique
    pth = data_dir / s
    if not pth.exists():
        raise FileNotFoundError(f"Fichier introuvable : {pth}")
    return [pth]


# =========================================================
# CALCULS CLIMAT (série complète)
# =========================================================

def compute_metrics_from_many_csv(
    csv_paths: list[Path],
    station: str,
    postcode: str,
    t_base_heat: float,
    t_base_cool: float,
    cold_percentile: float,
) -> StationClimateMetrics:

    times, temps_mean, temps_min, granularity, cols_used, files_used = read_many_csv(csv_paths)

    # ---------------- HDD ----------------
    hdd_year = defaultdict(float)

    if granularity == "hourly":
        for i in range(len(times) - 1):
            dt_h = (times[i + 1] - times[i]).total_seconds() / 3600.0
            if dt_h <= 0:
                continue
            deg = max(t_base_heat - temps_mean[i], 0.0)
            # degrés-heures -> degrés-jours
            hdd_year[times[i].year] += (deg * dt_h) / 24.0
    else:
        for t, ts in zip(temps_mean, times):
            hdd_year[ts.year] += max(t_base_heat - t, 0.0)

    years = sorted(hdd_year)
    hdd_mean = sum(hdd_year[y] for y in years) / len(years)

    # ---------------- T_ext design (minima) ----------------
    # min absolu basé sur la série des minima (pas des moyennes)
    t_min_abs = min(temps_min)

    # Percentile froid sur les mois d’hiver, basé sur temps_min
    winter_mins = [t for t, ts in zip(temps_min, times) if ts.month in WINTER_MONTHS] or temps_min
    t_design = percentile(winter_mins, cold_percentile)

    # ---------------- CLIM (sur moyenne) ----------------
    cool_hours = defaultdict(float)
    cool_degree_hours = defaultdict(float)

    if granularity == "hourly":
        for i in range(len(times) - 1):
            dt_h = (times[i + 1] - times[i]).total_seconds() / 3600.0
            if dt_h <= 0:
                continue
            delta = temps_mean[i] - t_base_cool
            if delta > 0:
                year = times[i].year
                cool_hours[year] += dt_h
                cool_degree_hours[year] += delta * dt_h
        mode = "hourly"
    else:
        for t, ts in zip(temps_mean, times):
            delta = t - t_base_cool
            if delta > 0:
                year = ts.year
                cool_hours[year] += 24.0
                cool_degree_hours[year] += delta * 24.0
        mode = "daily_approx"

    years_cool = sorted(cool_hours) or years

    cooling_hours_mean = (
        sum(cool_hours[y] for y in years_cool) / len(years_cool)
        if cool_hours else 0.0
    )
    cooling_degree_hours_mean = (
        sum(cool_degree_hours[y] for y in years_cool) / len(years_cool)
        if cool_degree_hours else 0.0
    )

    return StationClimateMetrics(
        station=station,
        postcode=postcode,
        hdd_mean=hdd_mean,
        t_min_abs=t_min_abs,
        t_design_percentile=t_design,
        cooling_hours_mean=cooling_hours_mean,
        cooling_degree_hours_mean=cooling_degree_hours_mean,
        cooling_mode=mode,
        temp_cols_used=cols_used,
        data_granularity=granularity,
        years_used=years,
        files_used=files_used,
    )


# =========================================================
# CLI
# =========================================================

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "Data" / "Climat_Villes_Daily"

    print("\n=== Calcul climat station (multi-CSV) ===\n")
    print(f"Dossier CSV : {data_dir}\n")
    print("Entrée fichiers acceptée :")
    print(" - fichier unique : Geneve_hourly_2000.csv")
    print(" - liste : Geneve_hourly_2000.csv,Geneve_hourly_2010.csv,Geneve_hourly_2020.csv")
    print(" - préfixe : Geneve_hourly_   (prend tous les CSV commençant par ce préfixe)")
    print(" - pattern : Geneve_hourly_*.csv\n")

    raw = input(f"Dossier alternatif [{data_dir}] : ").strip()
    if raw:
        data_dir = Path(raw)

    if not data_dir.exists():
        print(f"Dossier introuvable : {data_dir}")
        return

    files_in = input("Nom(s) fichier(s) / préfixe / pattern : ").strip()
    try:
        csv_paths = resolve_csv_list(data_dir, files_in)
    except Exception as e:
        print(f"Erreur fichiers : {e}")
        if data_dir.exists():
            print("\nCSV disponibles dans le dossier :")
            for p in sorted(data_dir.glob("*.csv")):
                print(f" - {p.name}")
        return

    station = input("Nom station : ").strip() or "STATION"
    postcode = input("Code postal : ").strip() or "0000"

    t_heat = float(input(f"T base chauffage [{T_BASE_HEAT}] : ").strip() or T_BASE_HEAT)
    t_cool = float(input(f"T base clim [{T_BASE_COOL}] : ").strip() or T_BASE_COOL)

    # Nouveau : défaut 0.5
    p_raw = input(f"Percentile froid [%] (ex 0.5) [{DEFAULT_COLD_PERCENTILE}] : ").strip()
    p = float(p_raw) if p_raw else DEFAULT_COLD_PERCENTILE

    try:
        m = compute_metrics_from_many_csv(
            csv_paths=csv_paths,
            station=station,
            postcode=postcode,
            t_base_heat=t_heat,
            t_base_cool=t_cool,
            cold_percentile=p,
        )
    except Exception as e:
        print(f"Erreur : {e}")
        return

    print("\n=== RÉSULTATS ===\n")
    print(f"Fichiers utilisés            : {', '.join(m.files_used)}")
    print(f"Colonnes utilisées           : {m.temp_cols_used}")
    print(f"Granularité détectée         : {m.data_granularity}")
    print(f"Années utilisées (HDD)        : {', '.join(str(y) for y in m.years_used)}")
    print("")
    print(f"HDD moyen (base {t_heat:.1f}°C)             : {m.hdd_mean:.0f} °C·jours/an")
    print(f"T_ext min absolu (sur minima)               : {m.t_min_abs:.1f} °C")
    print(f"T_ext design (percentile {p:.2f}% hiver)    : {m.t_design_percentile:.1f} °C")
    print(f"Heures clim (base {t_cool:.1f}°C)            : {m.cooling_hours_mean:.0f} h/an ({m.cooling_mode})")
    print(f"Cooling degree hours (base {t_cool:.1f}°C)   : {m.cooling_degree_hours_mean:.0f} °C·h/an ({m.cooling_mode})")

    print("\n=== Ligne CSV enrichie ===")
    print("station;postcode;hdd;t_ext_design_c;cooling_hours;cooling_degree_hours")
    print(
        f"{m.station};{m.postcode};"
        f"{m.hdd_mean:.0f};{m.t_design_percentile:.1f};"
        f"{m.cooling_hours_mean:.0f};{m.cooling_degree_hours_mean:.0f}"
    )


if __name__ == "__main__":
    main()
