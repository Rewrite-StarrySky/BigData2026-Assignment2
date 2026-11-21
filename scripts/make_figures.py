"""Q1 figure: write the DOT file of the schema graph (8 tables, foreign keys as edges).

    python -m scripts.make_figures          # writes figures/schema_graph.dot
    node figures/render.mjs figures/schema_graph.dot      # DOT -> SVG (Graphviz WASM build; `dot -Tsvg` works too)
"""
import json
from pathlib import Path
from src.config import RESULTS_DIR
from src.graphs import schema_dot, FKS, TABLES

FIG = Path(__file__).resolve().parents[1] / "figures"


def main():
    FIG.mkdir(exist_ok=True)
    (FIG / "schema_graph.dot").write_text(schema_dot())
    json.dump(dict(n_tables=len(TABLES), n_foreign_keys=len(FKS)), open(RESULTS_DIR / "entity_graph.json", "w"), indent=1)
    print("schema:", len(TABLES), "tables,", len(FKS), "foreign keys")


if __name__ == "__main__":
    main()
