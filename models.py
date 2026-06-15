from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ProjectInputs:
    project_type: str
    canton: str
    postcode: str

    building_type_key: str
    ubat: float
    ventilation_r: float

    sdep_mode: str
    sdep_m2: Optional[float]
    vh_m3: Optional[float]
    shab_m2: Optional[float]
    niveaux: Optional[int]
    perimetre_m: Optional[float]
    hauteur_m: float
    toiture_exposee: bool
    plancher_expose: bool

    forme_generale: Optional[str]
    mitoyennete: Optional[str]
    longueur_m: Optional[float]
    largeur_m: Optional[float]

    current_energy: Optional[str]
    current_efficiency: Optional[float]

    want_cooling: bool
    surface_climatisee_m2: float
    cooling_mode: str
    vitrage_level: str
    solar_protection_level: str
    usage_level: str
    night_ventilation: bool
    has_existing_ac: bool
    eer_current_ac: Optional[float]