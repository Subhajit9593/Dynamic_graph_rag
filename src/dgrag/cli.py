"""Command-line entry point: `dgrag <command>` (installed via pyproject) or `python -m dgrag.cli`.

Phase 1 commands:
  download    fetch MultiHop-RAG into data/raw
  load-docs   store articles as documents in SQLite (idempotent; no chunking/extraction yet)
  stats       print SQLite / FAISS / graph statistics
  check       verify FAISS and graph views are consistent with SQLite (repairs FAISS if not)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter

from dgrag.config import Config, load_config, setup_logging

log = logging.getLogger("dgrag")


def _store(cfg: Config):
    from dgrag.store import SQLiteStore

    return SQLiteStore(cfg.resolve(cfg.paths.db_path))


def cmd_download(cfg: Config, args: argparse.Namespace) -> None:
    from dgrag.ingest.loader import download_multihoprag

    path = download_multihoprag(cfg.resolve(cfg.paths.raw_dir), cfg.dataset.hf_repo)
    print(f"Dataset ready at {path}")


def cmd_load_docs(cfg: Config, args: argparse.Namespace) -> None:
    from dgrag.ingest.loader import load_corpus

    raw = cfg.resolve(cfg.paths.raw_dir)
    articles = load_corpus(raw / cfg.dataset.corpus_file)
    if args.limit:
        articles = articles[: args.limit]
    t0 = time.perf_counter()
    inserted = skipped = 0
    with _store(cfg) as store:
        with store.transaction():
            for a in articles:
                _, new = store.add_document(
                    a.body, title=a.title, author=a.author, source=a.source, url=a.url,
                    category=a.category, published_at=a.published_at,
                )
                inserted += new
                skipped += not new
        dt = time.perf_counter() - t0
        print(f"Documents: {inserted} inserted, {skipped} already present ({dt:.2f}s)")
        print(f"Date range: {articles[0].published_at} -> {articles[-1].published_at}")
        print("By source  :", dict(Counter(a.source for a in articles).most_common(8)), "...")
        print("By category:", dict(Counter(a.category for a in articles).most_common()))


def cmd_chunk_embed(cfg: Config, args: argparse.Namespace) -> None:
    from dgrag.ingest.chunk_embed import chunk_and_embed

    with _store(cfg) as store:
        print(json.dumps(chunk_and_embed(cfg, store, limit=args.limit), indent=2))


def cmd_search(cfg: Config, args: argparse.Namespace) -> None:
    """Plain vector search over chunks (sanity check for embeddings; also the B1 baseline's retriever)."""
    from dgrag.embedder import Embedder
    from dgrag.store import FaissManager

    with _store(cfg) as store:
        fm = FaissManager(store, cfg.resolve(cfg.paths.faiss_dir), cfg.embedder.dim)
        fm.open()
        emb = Embedder(cfg.embedder.model_name, cfg.embedder.device, cfg.embedder.batch_size)
        q = emb.encode([args.query], is_query=True)
        hits = fm.chunks.search(q, args.k)[0]
        for rank, (cid, score) in enumerate(hits, 1):
            ch = store.get_chunks([cid])[0]
            doc = store.get_document(ch["doc_id"])
            print(f"\n#{rank} score={score:.3f} chunk={cid} | {doc['source']} {doc['published_at'][:10]} | "
                  f"{doc['title']}")
            print("   ", ch["text"][:300].replace("\n", " "), "...")


def cmd_extract(cfg: Config, args: argparse.Namespace) -> None:
    from dgrag.extract.extractor import run_extraction

    if args.parallel:
        cfg.llm.parallel = args.parallel
    if args.max_tokens:
        cfg.llm.max_tokens = args.max_tokens
        cfg.llm.ctx_per_slot = max(cfg.llm.ctx_per_slot, args.max_tokens * 2 + 1500)  # room for the 1.5x retry
    with _store(cfg) as store:
        if args.give_up_failed:
            n = store.give_up_failed_extractions()
            print(json.dumps({"chunks_given_up": n, "documents_completed": store.mark_documents_extracted()}))
            return
        res = run_extraction(cfg, store, limit=args.limit)
        print(json.dumps(res, indent=2))
        print(json.dumps({"extraction_totals": store.extraction_stats()}, indent=2))


def cmd_revalidate(cfg: Config, args: argparse.Namespace) -> None:
    from dgrag.extract.extractor import revalidate_all

    with _store(cfg) as store:
        print(json.dumps(revalidate_all(store), indent=2))


def cmd_show_extraction(cfg: Config, args: argparse.Namespace) -> None:
    """Print a chunk next to what the LLM extracted from it (for manual quality review)."""
    with _store(cfg) as store:
        ids = args.chunk_ids or [r[0] for r in store.conn.execute(
            "SELECT chunk_id FROM extractions WHERE ok = 1 ORDER BY RANDOM() LIMIT ?", (args.n,))]
        for cid in ids:
            x, ch = store.get_extraction(cid), store.get_chunks([cid])[0]
            doc = store.get_document(ch["doc_id"])
            print("=" * 100)
            print(f"chunk {cid} | {doc['source']} {doc['published_at'][:10]} | {doc['title']}")
            print("-" * 100)
            print(ch["text"][: args.chars])
            print("-" * 100)
            if x is None:
                print("(not extracted)")
                continue
            print(f"ENTITIES ({x['n_entities']}, dropped {x['n_dropped_entities']}):")
            for e in json.loads(x["entities_json"]):
                print(f"  [{e['type']:<8}] {e['name']}  - {e['desc']}")
            print(f"RELATIONS ({x['n_relations']}, dropped {x['n_dropped_relations']}):")
            for r in json.loads(x["relations_json"]):
                print(f"  {r['src']} --[{r['pred']}]--> {r['tgt']}   | {r['desc']}")


def cmd_build_graph(cfg: Config, args: argparse.Namespace) -> None:
    from dgrag.resolve.run import build_graph

    variant = args.variant or ("dynamic" if args.mode == "incremental" else "static")
    out = build_graph(cfg, variant, args.mode, use_judge=not args.no_judge, limit_docs=args.limit)
    print(json.dumps(out, indent=2))


def cmd_stream(cfg: Config, args: argparse.Namespace) -> None:
    from dgrag.resolve.run import stream_experiment

    out = stream_experiment(cfg, n_batches=args.batches, use_judge=not args.no_judge,
                            suffix="_nojudge" if args.no_judge else "")
    print(json.dumps({k: v for k, v in out.items() if k != "batches"}, indent=2))


def cmd_eval(cfg: Config, args: argparse.Namespace) -> None:
    from dgrag.eval.run_eval import evaluate

    kw = json.loads(args.retriever_kw) if args.retriever_kw else None
    for system in args.systems:
        m = evaluate(cfg, system, limit=args.limit, k_prompt=args.k,
                     tag=f"{system}{args.tag_suffix}" if args.tag_suffix else None, retriever_kw=kw,
                     include_facts=not args.no_facts)
        print(json.dumps(m, indent=2))


def cmd_ask(cfg: Config, args: argparse.Namespace) -> None:
    """Answer one question with one system, showing facts, evidence and the answer."""
    from dataclasses import replace

    from dgrag.generate.answer import ANSWER_SCHEMA, build_answer_messages, parse_answer
    from dgrag.eval.run_eval import retrieve_all
    from dgrag.ingest.loader import Question
    from dgrag.llm.server import LlamaServer

    q = Question(-1, args.question, "", "inference_query")
    ret = retrieve_all(cfg, args.system, [q])[-1]
    if ret is not None:
        ret = replace(ret, chunks=ret.chunks[: args.k])
        if ret.facts:
            print("FACTS:")
            for f in ret.facts[:10]:
                print(f"  {f.score:.2f}  {f.text}")
        print("EVIDENCE:")
        for i, c in enumerate(ret.chunks, 1):
            print(f"  [{i}] {c.score:.2f} {c.source} {c.published_at[:10]} | {c.title}")
    with LlamaServer.from_config(cfg, parallel=1, ctx_per_slot=8192) as srv:
        obj, res = srv.client().chat_json(build_answer_messages(args.question, ret), ANSWER_SCHEMA, max_tokens=256)
    ans, why = parse_answer(obj, res.content)
    print(f"\nREASONING: {why}\nANSWER: {ans}")


def cmd_add_article(cfg: Config, args: argparse.Namespace) -> None:
    """Add one new article (JSON file) to a graph variant without rebuilding anything."""
    from pathlib import Path

    from dgrag.embedder import Embedder
    from dgrag.ingest.pipeline import NewArticle, add_article
    from dgrag.llm.server import LlamaServer
    from dgrag.store import FaissManager, SQLiteStore

    data = json.loads(Path(args.file).read_text())
    art = NewArticle(data["title"], data["body"], data.get("source"), data.get("published_at"), data.get("url"),
                     data.get("category"))
    db, fdir = cfg.resolve(Path(f"data/db/{args.graph}.sqlite")), cfg.resolve(Path(f"data/faiss/{args.graph}"))
    if not db.exists():
        raise SystemExit(f"{db} not found (build it first, e.g. dgrag stream)")
    with SQLiteStore(db) as store:
        fm = FaissManager(store, fdir, cfg.embedder.dim)
        fm.open()
        emb = Embedder(cfg.embedder.model_name, cfg.embedder.device, cfg.embedder.batch_size)
        with LlamaServer.from_config(cfg, parallel=4, ctx_per_slot=4096) as srv:
            out = add_article(cfg, store, fm, emb, srv.client(), art)
        emb.close()
    print(json.dumps(out, indent=2, ensure_ascii=False))


def cmd_stats(cfg: Config, args: argparse.Namespace) -> None:
    from dgrag.store import FaissManager, GraphView

    with _store(cfg) as store:
        out = {"sqlite": store.stats()}
        fm = FaissManager(store, cfg.resolve(cfg.paths.faiss_dir), cfg.embedder.dim)
        for t in fm.TABLES:
            fm[t].load()
        out["faiss"] = {t: fm[t].ntotal for t in fm.TABLES}
        out["graph"] = GraphView.from_store(store).stats()
    print(json.dumps(out, indent=2, default=str))


def cmd_check(cfg: Config, args: argparse.Namespace) -> None:
    from dgrag.store import FaissManager, GraphView

    with _store(cfg) as store:
        fm = FaissManager(store, cfg.resolve(cfg.paths.faiss_dir), cfg.embedder.dim)
        report = fm.open()
        fm.save()
        ok_faiss = all(fm.is_consistent(t) for t in fm.TABLES)
        ok_graph = GraphView.from_store(store).matches_store(store)
    print(json.dumps({"faiss_reconcile": report, "faiss_consistent": ok_faiss, "graph_consistent": ok_graph},
                     indent=2))
    if not (ok_faiss and ok_graph):
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="dgrag", description="Dynamic Graph RAG")
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--variant", dest="db_variant", default=None,
                   help="operate on a graph copy (data/db/<variant>.sqlite), e.g. dynamic or static")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="download MultiHop-RAG").set_defaults(fn=cmd_download)
    ld = sub.add_parser("load-docs", help="store articles in SQLite")
    ld.add_argument("--limit", type=int, default=None, help="only the first N articles (chronological)")
    ld.set_defaults(fn=cmd_load_docs)
    ce = sub.add_parser("chunk-embed", help="chunk loaded documents and embed chunks (SQLite + FAISS)")
    ce.add_argument("--limit", type=int, default=None)
    ce.set_defaults(fn=cmd_chunk_embed)
    se = sub.add_parser("search", help="vector search over chunks")
    se.add_argument("query")
    se.add_argument("-k", type=int, default=5)
    se.set_defaults(fn=cmd_search)
    ex = sub.add_parser("extract", help="LLM entity/relation extraction for pending chunks (resumable)")
    ex.add_argument("--limit", type=int, default=None, help="only the next N pending chunks")
    ex.add_argument("--parallel", type=int, default=None, help="override llm.parallel slots")
    ex.add_argument("--give-up-failed", action="store_true",
                    help="mark chunks that still fail as done with no facts (their documents can proceed)")
    ex.add_argument("--max-tokens", type=int, default=None, help="override llm.max_tokens (e.g. to retry truncated)")
    ex.set_defaults(fn=cmd_extract)
    sub.add_parser("revalidate", help="re-apply cleaning rules to stored LLM output (no LLM)").set_defaults(
        fn=cmd_revalidate)
    sx = sub.add_parser("show-extraction", help="show chunks next to their extracted entities/relations")
    sx.add_argument("chunk_ids", type=int, nargs="*")
    sx.add_argument("-n", type=int, default=3, help="number of random chunks if no ids given")
    sx.add_argument("--chars", type=int, default=1200)
    sx.set_defaults(fn=cmd_show_extraction)
    bg = sub.add_parser("build-graph", help="build the knowledge graph from cached extractions")
    bg.add_argument("--mode", choices=["incremental", "global"], default="incremental")
    bg.add_argument("--variant", default=None, help="name of the graph copy (default: dynamic/static)")
    bg.add_argument("--limit", type=int, default=None, help="only the first N documents (chronological)")
    bg.add_argument("--no-judge", action="store_true", help="no LLM judge (uncertain cases are not merged)")
    bg.set_defaults(fn=cmd_build_graph)
    sx2 = sub.add_parser("stream", help="streaming experiment: incremental vs full rebuild, batch by batch")
    sx2.add_argument("--batches", type=int, default=10)
    sx2.add_argument("--no-judge", action="store_true")
    sx2.set_defaults(fn=cmd_stream)
    ev = sub.add_parser("eval", help="run MultiHop-RAG QA evaluation (resumable)")
    ev.add_argument("systems", nargs="+", choices=["none", "vector", "vector_rerank", "static", "dynamic"])
    ev.add_argument("--limit", type=int, default=None, help="stratified subset of N questions")
    ev.add_argument("-k", type=int, default=8, help="chunks in the prompt")
    ev.add_argument("--tag-suffix", default="", help="suffix for result file names (e.g. _dev)")
    ev.add_argument("--retriever-kw", default=None, help='JSON overrides for the retriever, e.g. \'{"w_source": 0}\'')
    ev.add_argument("--no-facts", action="store_true", help="do not put graph facts in the prompt")
    ev.set_defaults(fn=cmd_eval)
    ak = sub.add_parser("ask", help="answer one question")
    ak.add_argument("question")
    ak.add_argument("--system", default="dynamic", choices=["none", "vector", "vector_rerank", "static", "dynamic"])
    ak.add_argument("-k", type=int, default=8)
    ak.set_defaults(fn=cmd_ask)
    aa = sub.add_parser("add-article", help="add one new article (JSON file) to a graph, no rebuild")
    aa.add_argument("file", help='JSON with "title", "body", optional "source", "published_at", "url"')
    aa.add_argument("--graph", default="demo", help="graph variant to modify (default: demo copy)")
    aa.set_defaults(fn=cmd_add_article)
    sub.add_parser("stats", help="print statistics").set_defaults(fn=cmd_stats)
    sub.add_parser("check", help="verify derived views against SQLite").set_defaults(fn=cmd_check)

    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.db_variant and args.command not in ("build-graph", "stream"):
        from pathlib import Path

        cfg.paths.db_path = Path(f"data/db/{args.db_variant}.sqlite")
        cfg.paths.faiss_dir = Path(f"data/faiss/{args.db_variant}")
    setup_logging(cfg.logging.level)
    args.fn(cfg, args)


if __name__ == "__main__":
    main()
