"""Command-line interface (argparse-based, no extra dependencies).

Run ``python main.py --help`` to see all commands. Each subcommand maps to a
method on :class:`uxeg.EvidenceGraphProject`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import EvidenceGraphProject
from .utils import console, error, info, success, warn


def _project(args) -> EvidenceGraphProject:
    return EvidenceGraphProject.from_config(args.config)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------
def cmd_init(args) -> int:
    project = _project(args)
    project.init()
    success(f"Initialized database at {project.config.database_path}")
    info("Next: python main.py ingest --input data/sample")
    return 0


def cmd_health(args) -> int:
    project = _project(args)
    ok = project.health_check()
    return 0 if ok else 1


def cmd_ingest(args) -> int:
    project = _project(args)
    try:
        project.store._ensure_initialized()
    except RuntimeError as exc:
        error(str(exc))
        return 1
    try:
        project.ingest(args.input)
    except FileNotFoundError as exc:
        error(str(exc))
        return 1
    return 0


def cmd_extract(args) -> int:
    project = _project(args)
    try:
        project.store._ensure_initialized()
    except RuntimeError as exc:
        error(str(exc))
        return 1
    project.extract()
    info("Next: review edges with 'python main.py export-review'")
    return 0


def cmd_embed(args) -> int:
    project = _project(args)
    project.embed()
    return 0


def cmd_export_review(args) -> int:
    project = _project(args)
    project.export_review(args.output)
    info("Open the CSV, edit review_status / weight / confidence, then run import-review.")
    return 0


def cmd_import_review(args) -> int:
    project = _project(args)
    try:
        project.import_review(args.input)
    except (FileNotFoundError, ValueError) as exc:
        error(str(exc))
        return 1
    return 0


def cmd_export_nodes_review(args) -> int:
    project = _project(args)
    project.export_nodes_review(args.output)
    return 0


def cmd_import_nodes_review(args) -> int:
    project = _project(args)
    try:
        project.import_nodes_review(args.input)
    except (FileNotFoundError, ValueError) as exc:
        error(str(exc))
        return 1
    return 0


def cmd_suggest_merges(args) -> int:
    project = _project(args)
    project.suggest_merges(args.output)
    info("Edit 'approve_merge' to 'yes' for merges you accept, then run import-merges.")
    return 0


def cmd_import_merges(args) -> int:
    project = _project(args)
    try:
        project.import_merges(args.input)
    except FileNotFoundError as exc:
        error(str(exc))
        return 1
    return 0


def cmd_ask(args) -> int:
    project = _project(args)
    try:
        project.store._ensure_initialized()
    except RuntimeError as exc:
        error(str(exc))
        return 1
    if project.store.count("nodes") == 0:
        warn("The graph has no nodes yet. Run ingest + extract first.")
    answer = project.ask(args.question)
    console.rule("[bold]Answer")
    console.print(answer)
    success("Answer saved under outputs/answers/")
    return 0


def cmd_export_graph(args) -> int:
    project = _project(args)
    path = project.export_graph(args.format)
    success(f"Exported graph to {path}")
    return 0


def cmd_export_tables(args) -> int:
    project = _project(args)
    paths = project.export_tables()
    for name, path in paths.items():
        success(f"Exported {name} to {path}")
    return 0


def cmd_export_summary(args) -> int:
    project = _project(args)
    path = project.export_summary()
    success(f"Wrote graph summary to {path}")
    return 0


def cmd_export_image(args) -> int:
    project = _project(args)
    path = project.export_image()
    if path:
        success(f"Wrote graph image to {path}")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uxeg",
        description="ux-evidence-graph: turn unstructured UX text into a reviewable evidence graph (local, LM Studio).",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Create the SQLite database and tables").set_defaults(func=cmd_init)
    sub.add_parser("health", help="Check LM Studio connectivity").set_defaults(func=cmd_health)

    p_ingest = sub.add_parser("ingest", help="Read and chunk documents from a folder")
    p_ingest.add_argument("--input", default="data/raw", help="Folder to read (default: data/raw)")
    p_ingest.set_defaults(func=cmd_ingest)

    sub.add_parser("extract", help="Extract nodes and edges from chunks via LM Studio").set_defaults(func=cmd_extract)
    sub.add_parser("embed", help="Create embeddings for all chunks").set_defaults(func=cmd_embed)

    p_er = sub.add_parser("export-review", help="Export edges to a review CSV")
    p_er.add_argument("--output", default="outputs/review/edges_for_review.csv")
    p_er.set_defaults(func=cmd_export_review)

    p_ir = sub.add_parser("import-review", help="Import an edited edge review CSV")
    p_ir.add_argument("--input", default="outputs/review/edges_for_review.csv")
    p_ir.set_defaults(func=cmd_import_review)

    p_enr = sub.add_parser("export-nodes-review", help="Export nodes to a review CSV")
    p_enr.add_argument("--output", default="outputs/review/nodes_for_review.csv")
    p_enr.set_defaults(func=cmd_export_nodes_review)

    p_inr = sub.add_parser("import-nodes-review", help="Import an edited node review CSV")
    p_inr.add_argument("--input", default="outputs/review/nodes_for_review.csv")
    p_inr.set_defaults(func=cmd_import_nodes_review)

    p_sm = sub.add_parser("suggest-merges", help="Suggest near-duplicate nodes to merge")
    p_sm.add_argument("--output", default="outputs/review/node_merge_suggestions.csv")
    p_sm.set_defaults(func=cmd_suggest_merges)

    p_im = sub.add_parser("import-merges", help="Apply approved node merges from a CSV")
    p_im.add_argument("--input", default="outputs/review/node_merge_suggestions.csv")
    p_im.set_defaults(func=cmd_import_merges)

    p_ask = sub.add_parser("ask", help="Ask a higher-level question over the graph")
    p_ask.add_argument("--question", required=True, help="The question to answer")
    p_ask.set_defaults(func=cmd_ask)

    p_eg = sub.add_parser("export-graph", help="Export the graph to json or graphml")
    p_eg.add_argument("--format", choices=["json", "graphml"], default="json")
    p_eg.set_defaults(func=cmd_export_graph)

    sub.add_parser("export-tables", help="Export nodes.csv, edges.csv, adjacency.csv").set_defaults(func=cmd_export_tables)
    sub.add_parser("export-summary", help="Write outputs/graphs/graph_summary.md").set_defaults(func=cmd_export_summary)
    sub.add_parser("export-image", help="Render a static PNG of the graph (optional)").set_defaults(func=cmd_export_image)

    return parser


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    project = None
    try:
        return args.func(args)
    except KeyboardInterrupt:
        warn("Interrupted by user.")
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level friendly handler
        error(f"Unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
