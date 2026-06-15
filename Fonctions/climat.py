from __future__ import annotations

import csv
import math
from pathlib import Path

# Racine du projet
ROOT_DIR = Path(__file__).resolve().parents[1]

# Base stations climatiques
DATA_FILE = ROOT_DIR / "data" / "climat_ch_stations.csv"

# Base codes postaux suisses nettoyée
POSTCODE_FILE = ROOT_DIR / "data" /"localites_postcodes" / "ch_postcodes_weighted.csv"


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    s = str(value).strip().replace(",", ".")
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _load_postcode_data() -> dict[str, dict[str, float | str]]:
    """
    Charge le fichier des codes postaux suisses agrégés.

    Colonnes attendues :
      - postcode
      - e
      - n
      - canton
      - locality
    """
    mapping: dict[str, dict[str, float | str]] = {}

    with open(POSTCODE_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError("CSV des codes postaux sans en-têtes.")

        required = {"postcode", "e", "n", "canton", "locality"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Colonnes manquantes dans {POSTCODE_FILE}: {sorted(missing)}"
            )

        for row in reader:
            pc = str(row.get("postcode", "")).strip().zfill(4)
            if not pc:
                continue

            e = _to_float(row.get("e"))
            n = _to_float(row.get("n"))
            if e is None or n is None:
                continue

            mapping[pc] = {
                "postcode": pc,
                "e": float(e),
                "n": float(n),
                "canton": str(row.get("canton", "")).strip(),
                "locality": str(row.get("locality", "")).strip(),
            }

    return mapping


def _load_station_data() -> list[dict[str, float | str]]:
    """
    Charge climat_ch_stations.csv.

    Colonnes attendues :
      - postcode
      - hdd

    Colonnes recommandées pour la recherche par proximité :
      - e
      - n

    Colonnes optionnelles :
      - station_name
      - t_ext_design_c
      - cooling_hours
      - cooling_degree_hours
    """
    stations: list[dict[str, float | str]] = []

    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError("CSV climat_ch_stations.csv sans en-têtes.")

        required = {"postcode", "hdd", "e", "n"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Colonnes manquantes dans {DATA_FILE}: {sorted(missing)}. "
                "Pour associer la station la plus proche, il faut au minimum "
                "postcode, e, n, hdd."
            )

        for row in reader:
            pc = str(row.get("postcode", "")).strip().zfill(4)
            if not pc:
                continue

            hdd = _to_float(row.get("hdd"))
            e = _to_float(row.get("e"))
            n = _to_float(row.get("n"))

            if hdd is None or e is None or n is None:
                continue

            stations.append(
                {
                    "postcode": pc,
                    "station_name": str(row.get("station_name", pc)).strip() or pc,
                    "e": float(e),
                    "n": float(n),
                    "hdd": float(hdd),
                    "t_ext_design_c": float(_to_float(row.get("t_ext_design_c")) or 0.0),
                    "cooling_hours": float(_to_float(row.get("cooling_hours")) or 0.0),
                    "cooling_degree_hours": float(_to_float(row.get("cooling_degree_hours")) or 0.0),
                }
            )

    if not stations:
        raise ValueError(f"Aucune station climatique exploitable trouvée dans {DATA_FILE}.")

    return stations


# Caches
_POSTCODE_DATA: dict[str, dict[str, float | str]] | None = None
_STATIONS: list[dict[str, float | str]] | None = None


def _get_postcodes() -> dict[str, dict[str, float | str]]:
    global _POSTCODE_DATA
    if _POSTCODE_DATA is None:
        _POSTCODE_DATA = _load_postcode_data()
    return _POSTCODE_DATA


def _get_stations() -> list[dict[str, float | str]]:
    global _STATIONS
    if _STATIONS is None:
        _STATIONS = _load_station_data()
    return _STATIONS


def _distance_m(e1: float, n1: float, e2: float, n2: float) -> float:
    """Distance plane en mètres dans le système suisse E/N."""
    return math.hypot(e2 - e1, n2 - n1)


def postcode_info(postcode: str) -> dict[str, float | str]:
    """
    Retourne les infos géographiques du code postal utilisateur.
    """
    postcodes = _get_postcodes()
    pc = str(postcode).strip().zfill(4)

    if pc not in postcodes:
        raise ValueError(
            f"Code postal suisse inconnu : {pc}. "
            f"Ajoute-le dans {POSTCODE_FILE}."
        )

    return postcodes[pc]


def nearest_station_for_postcode(postcode: str) -> dict[str, float | str]:
    """
    Associe un code postal suisse à la station climatique la plus proche.
    """
    pc_info = postcode_info(postcode)
    e_pc = float(pc_info["e"])
    n_pc = float(pc_info["n"])

    best_station = None
    best_distance = None

    for station in _get_stations():
        d = _distance_m(
            e_pc,
            n_pc,
            float(station["e"]),
            float(station["n"]),
        )
        if best_distance is None or d < best_distance:
            best_distance = d
            best_station = station

    if best_station is None:
        raise ValueError("Impossible de trouver une station climatique proche.")

    result = dict(best_station)
    result["distance_m"] = float(best_distance)
    result["input_postcode"] = str(postcode).strip().zfill(4)
    result["input_locality"] = str(pc_info["locality"])
    result["input_canton"] = str(pc_info["canton"])
    return result


def climate_for_postcode(postcode: str) -> dict[str, float | str]:
    """
    Retourne les données climatiques associées à la station la plus proche.
    """
    return nearest_station_for_postcode(postcode)


def hdd_for_postcode(postcode: str) -> float:
    return float(climate_for_postcode(postcode)["hdd"])


def t_ext_design_for_postcode(postcode: str) -> float:
    return float(climate_for_postcode(postcode)["t_ext_design_c"])


def cooling_hours_for_postcode(postcode: str) -> float:
    return float(climate_for_postcode(postcode)["cooling_hours"])


def cooling_degree_hours_for_postcode(postcode: str) -> float:
    return float(climate_for_postcode(postcode)["cooling_degree_hours"])