from __future__ import annotations

import argparse
from pathlib import Path
import requests


SPARQL_ENDPOINT = "https://lindas.admin.ch/query"  # alternatif: https://lindas-cached.cluster.ldbar.ch/query


QUERY_TEMPLATE = r"""
PREFIX cube:   <https://cube.link/>
PREFIX schema: <http://schema.org/>
PREFIX xsd:    <http://www.w3.org/2001/XMLSchema#>
PREFIX eldim:  <https://energy.ld.admin.ch/elcom/electricityprice/dimension/>

SELECT
  ?cantonCode
  ?municipalityName
  ?year
  ?product
  ?category
  ?energyname
  (xsd:decimal(?total)/100.0     AS ?total_chf_kwh)
  (xsd:decimal(?gridusage)/100.0 AS ?gridusage_chf_kwh)
  (xsd:decimal(?energy)/100.0    AS ?energy_chf_kwh)
  (xsd:decimal(?charge)/100.0    AS ?communityfees_chf_kwh)
  (xsd:decimal(?aidfee)/100.0    AS ?aidfee_chf_kwh)
  ?fixcosts_chf_year
  ?annualmeteringcost_chf_year
  ?meteringrate_rp_kwh
WHERE {
  GRAPH <https://lindas.admin.ch/elcom/electricityprice> {
    ?obs a cube:Observation ;
      eldim:municipality ?municipality ;
      eldim:period ?year ;
      eldim:product ?productIRI ;
      eldim:category ?categoryIRI ;
      eldim:total ?total ;
      eldim:gridusage ?gridusage ;
      eldim:energy ?energy ;
      eldim:charge ?charge ;
      eldim:aidfee ?aidfee .

    OPTIONAL { ?obs eldim:energyname ?energyname . }
    OPTIONAL { ?obs eldim:fixcosts ?fixcosts_chf_year . }
    OPTIONAL { ?obs eldim:annualmeteringcost ?annualmeteringcost_chf_year . }
    OPTIONAL { ?obs eldim:meteringrate ?meteringrate_rp_kwh . }
  }

  ?canton schema:containsPlace ?municipality ;
          schema:alternateName ?cantonCode .

  ?municipality schema:name ?municipalityName .
  ?productIRI  schema:name ?product .
  ?categoryIRI schema:name ?category .

  FILTER(BOUND(?energyname))
  FILTER(CONTAINS(LCASE(STR(?energyname)), "unternehmen"))

  VALUES ?cantonCode {{ "{canton}" }}
}
ORDER BY ?year ?municipalityName ?category ?product
"""


def run_sparql_to_csv(query: str, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Accept": "text/csv"}
    r = requests.get(
        SPARQL_ENDPOINT,
        params={"query": query},
        headers=headers,
        timeout=120,
    )
    r.raise_for_status()
    out_csv.write_bytes(r.content)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canton", required=True, help="Code canton (ex: VD, GE, ZH)")
    ap.add_argument("--out", default="Data/out/elcom_nonhousehold_prices.csv", help="Chemin CSV de sortie")
    args = ap.parse_args()

    query = QUERY_TEMPLATE.format(canton=args.canton.strip().upper())
    out_csv = Path(args.out)

    run_sparql_to_csv(query, out_csv)
    print(f"OK -> {out_csv.resolve()}")


if __name__ == "__main__":
    main()
