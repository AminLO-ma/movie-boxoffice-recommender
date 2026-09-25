"""Inventaire des champs d'un ensemble de documents JSON (un document = un film)."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from boxoffice import config

REPORTS = config.ROOT / "reports"
COUNTRY_CODE = re.compile(r"[A-Z]{2}")
DISTINCT_CAP = 50_000
N_EXAMPLES = 3
LEGEND = (
    "- **rempli_%** : part des films ayant au moins une valeur non vide (ni null, ni \"\", ni liste vide)",
    "- **zero_%** : part des valeurs numériques renseignées égales à 0 (valeur manquante déguisée ?)",
    "- **utile_%** : part des films ayant au moins une valeur non vide et non nulle",
    "- **val/film** : nombre moyen de valeurs par film renseigné (> 1 pour une liste)",
    "- **distinctes** : nombre de valeurs différentes (plafonné à 50 000)",
    "- `[]` marque une liste ; `{pays}` regroupe les clés qui sont des codes pays (FR, US…)",
)


def flatten(obj, prefix: str = "", out: dict | None = None) -> dict[str, list]:
    """Aplatit un document JSON en {chemin: [valeurs feuilles]}."""
    out = defaultdict(list) if out is None else out
    if isinstance(obj, dict):
        for key, value in obj.items():
            key = "{pays}" if COUNTRY_CODE.fullmatch(key) else key
            flatten(value, f"{prefix}.{key}" if prefix else key, out)
    elif isinstance(obj, list):
        for item in obj:
            flatten(item, f"{prefix}[]", out)
    else:
        out[prefix].append(obj)
    return out


def _is_zero(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0


def inventory(documents: list[dict], sections: Iterable[str] = ()) -> pd.DataFrame:
    sections = list(sections)
    acc = defaultdict(lambda: {"filled": 0, "useful": 0, "values": 0, "numeric": 0, "zeros": 0,
                               "types": set(), "distinct": set(), "examples": []})
    for doc in documents:
        for path, values in flatten(doc).items():
            s = acc[path]
            filled = [v for v in values if v is not None and v != ""]
            if not filled:
                continue
            s["filled"] += 1
            s["useful"] += any(not _is_zero(v) for v in filled)
            s["values"] += len(filled)
            for v in filled:
                s["types"].add(type(v).__name__)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    s["numeric"] += 1
                    s["zeros"] += v == 0
                if len(s["distinct"]) < DISTINCT_CAP:
                    s["distinct"].add(v)
                if len(s["examples"]) < N_EXAMPLES and v not in s["examples"] and not _is_zero(v):
                    s["examples"].append(v)

    containers = {p for p in acc for q in acc if q.startswith((p + ".", p + "["))}
    n_docs = len(documents)
    rows = []
    for path, s in acc.items():
        if not s["filled"] and path in containers:
            continue
        top = re.split(r"[.\[]", path, maxsplit=1)[0]
        rows.append({
            "section": top if top in sections else "details",
            "champ": path,
            "types": "/".join(sorted(s["types"])),
            "rempli_%": round(100 * s["filled"] / n_docs, 1),
            "zero_%": round(100 * s["zeros"] / s["numeric"], 1) if s["numeric"] else None,
            "utile_%": round(100 * s["useful"] / n_docs, 1),
            "val/film": round(s["values"] / s["filled"], 1) if s["filled"] else None,
            "distinctes": len(s["distinct"]),
            "exemples": " ; ".join(str(v)[:40] for v in s["examples"]),
        })
    order = {name: rank for rank, name in enumerate(["details", *sections])}
    df = pd.DataFrame(rows)
    return df.sort_values("section", key=lambda col: col.map(order), kind="stable").reset_index(drop=True)


def _md_cell(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def to_markdown(df: pd.DataFrame, title: str, n_docs: int) -> str:
    columns = [c for c in df.columns if c != "section"]
    lines = [f"# {title}", "", f"{n_docs} films analysés.", "", *LEGEND, ""]
    for section, group in df.groupby("section", sort=False):
        lines += [f"## {section} ({len(group)} champs)", "",
                  "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        for row in group[columns].itertuples(index=False):
            cells = [f"`{row[0]}`", *(_md_cell(v) for v in row[1:])]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventaire des champs d'un cache TMDB.")
    parser.add_argument("--cache", type=Path, default=config.TMDB_EXPLORE_CACHE)
    parser.add_argument("--name", default="tmdb_inventory", help="nom des fichiers produits dans reports/")
    args = parser.parse_args(argv)

    files = sorted(args.cache.glob("*.json"))
    if not files:
        parser.error(f"aucun JSON dans {args.cache}")
    documents = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    df = inventory(documents, sections=config.TMDB_APPEND_ALL)

    REPORTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS / f"{args.name}.csv", index=False, encoding="utf-8")
    md = to_markdown(df, f"Inventaire des champs TMDB ({args.cache.name})", len(documents))
    (REPORTS / f"{args.name}.md").write_text(md, encoding="utf-8")
    print(f"{len(df)} champs sur {len(documents)} films -> reports/{args.name}.csv / .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
