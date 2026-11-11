"""Dynamic Graph RAG - demo UI.

Run:  source ~/ml-env/bin/activate && streamlit run app/streamlit_app.py
Tabs:
  Ask          question -> graph retrieval -> answer with citations, the facts used and the subgraph
  Add article  paste a new article -> chunk, embed, extract, merge into the existing graph (no rebuild)
  Graph        size, most connected entities, facts replaced over time

The demo works on its own copy of the dynamic graph (data/db/demo.sqlite), so the evaluated graphs stay untouched.
"""

from __future__ import annotations

import atexit
import datetime as dt
import sys
import time
from dataclasses import replace
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dgrag.config import load_config, setup_logging  # noqa: E402
from dgrag.generate.answer import ANSWER_SCHEMA, build_answer_messages, parse_answer  # noqa: E402
from dgrag.ingest.pipeline import NewArticle, add_article  # noqa: E402
from dgrag.retrieve.retrievers import GraphRetriever  # noqa: E402
from dgrag.store import FaissManager, SQLiteStore  # noqa: E402

DEMO = "demo"
st.set_page_config(page_title="Dynamic Graph RAG", page_icon="🕸️", layout="wide")


# ------------------------------------------------------------------ resources (created once per server)
@st.cache_resource
def get_cfg():
    cfg = load_config()
    setup_logging("WARNING")
    return cfg


def ensure_demo_db(cfg, reset: bool = False) -> None:
    import shutil
    import sqlite3

    src = cfg.resolve(Path("data/db/dynamic.sqlite"))
    dst = cfg.resolve(Path(f"data/db/{DEMO}.sqlite"))
    if reset or not dst.exists():
        if not src.exists():
            st.error("data/db/dynamic.sqlite not found. Build the graph first: `dgrag stream` or "
                     "`dgrag build-graph`.")
            st.stop()
        for p in (dst, dst.with_suffix(".sqlite-wal"), dst.with_suffix(".sqlite-shm")):
            p.unlink(missing_ok=True)
        shutil.rmtree(cfg.resolve(Path(f"data/faiss/{DEMO}")), ignore_errors=True)
        a, b = sqlite3.connect(str(src)), sqlite3.connect(str(dst))
        a.backup(b)
        a.close()
        b.close()


@st.cache_resource
def get_store():
    cfg = get_cfg()
    ensure_demo_db(cfg)
    store = SQLiteStore(cfg.resolve(Path(f"data/db/{DEMO}.sqlite")), check_same_thread=False)
    fm = FaissManager(store, cfg.resolve(Path(f"data/faiss/{DEMO}")), cfg.embedder.dim)
    fm.open()
    return store, fm


@st.cache_resource
def get_embedder():
    from dgrag.embedder import Embedder

    cfg = get_cfg()
    return Embedder(cfg.embedder.model_name, cfg.embedder.device, cfg.embedder.batch_size)


@st.cache_resource
def get_llm():
    from dgrag.llm.server import LlamaServer

    cfg = get_cfg()
    srv = LlamaServer.from_config(cfg, port=8095, parallel=4, ctx_per_slot=8192, kv_cache_type="q8_0")
    srv.start()
    atexit.register(srv.stop)
    return srv.client()


@st.cache_resource
def get_retriever(version: int):
    store, fm = get_store()
    return GraphRetriever(store, fm, k=20)


if "graph_version" not in st.session_state:
    st.session_state.graph_version = 0


# ------------------------------------------------------------------ helpers
def subgraph_html(store, facts) -> str:
    from pyvis.network import Network

    colors = {"PERSON": "#f4a261", "ORG": "#2a9d8f", "LOCATION": "#e9c46a", "EVENT": "#e76f51",
              "PRODUCT": "#8ab17d", "WORK": "#b56576", "OTHER": "#adb5bd"}
    net = Network(height="460px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")
    rel_ids = [f.relation_id for f in facts]
    if not rel_ids:
        return "<p>No graph facts for this question.</p>"
    rows = store.conn.execute(
        f"SELECT r.src_id, s.name, s.type, r.tgt_id, t.name, t.type, r.predicate FROM relations r "
        f"JOIN entities s ON s.id = r.src_id JOIN entities t ON t.id = r.tgt_id "
        f"WHERE r.id IN ({','.join('?' * len(rel_ids))})", rel_ids).fetchall()
    for sid, sn, stype, tid, tn, ttype, pred in rows:
        net.add_node(sid, label=sn, color=colors.get(stype, "#adb5bd"), title=stype)
        net.add_node(tid, label=tn, color=colors.get(ttype, "#adb5bd"), title=ttype)
        net.add_edge(sid, tid, label=pred, arrows="to")
    net.repulsion(node_distance=160, spring_length=160)
    return net.generate_html()


# ------------------------------------------------------------------ UI
cfg = get_cfg()
store, fm = get_store()
st.title("🕸️ Dynamic Graph RAG")
st.caption("Qwen2.5-7B (llama.cpp) · bge-base embeddings · FAISS · SQLite · MultiHop-RAG news (Sep–Dec 2023)")

with st.sidebar:
    s = store.stats()
    st.metric("Articles", s["documents"])
    st.metric("Entities (nodes)", f"{s['entities']:,}")
    st.metric("Relations (edges)", f"{s['relations_active'] + s['relations_conflicted']:,}")
    st.metric("Facts replaced over time", s["relations_superseded"])
    k = st.slider("Chunks given to the LLM", 4, 12, 8)
    if st.button("Reset demo graph to the evaluated version"):
        get_store.clear()
        get_retriever.clear()
        ensure_demo_db(cfg, reset=True)
        st.session_state.graph_version += 1
        st.rerun()

tab_ask, tab_add, tab_graph = st.tabs(["💬 Ask", "➕ Add article", "📊 Graph"])

with tab_ask:
    examples = [
        "Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and "
        "conspiracy charges, as reported by both The Verge and TechCrunch?",
        "Which company did Eddy Cue say had no valid alternative to Google search?",
        "Who owns X, formerly known as Twitter?",
    ]
    q = st.text_area("Question", examples[0], height=90)
    if st.button("Ask", type="primary"):
        emb, client = get_embedder(), get_llm()
        retriever = get_retriever(st.session_state.graph_version)
        t0 = time.perf_counter()
        ret = retriever.retrieve(q, emb.encode([q], is_query=True))
        ret = replace(ret, chunks=ret.chunks[:k])
        t_ret = time.perf_counter() - t0
        obj, res = client.chat_json(build_answer_messages(q, ret), ANSWER_SCHEMA, max_tokens=256)
        ans, why = parse_answer(obj, res.content)
        st.success(f"**Answer:** {ans}")
        st.caption(f"Reasoning: {why}")
        st.caption(f"retrieval {t_ret:.2f}s · generation {res.seconds:.2f}s · {res.prompt_tokens} prompt tokens · "
                   f"{ret.debug}")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Facts from the graph")
            st.dataframe([{"score": round(f.score, 2), "fact": f.text, "evidence chunks": len(f.chunk_ids)}
                          for f in ret.facts], width="stretch", hide_index=True)
            st.subheader("Subgraph")
            st.components.v1.html(subgraph_html(store, ret.facts[:12]), height=480)
        with c2:
            st.subheader("Evidence given to the LLM")
            for i, ch in enumerate(ret.chunks, 1):
                with st.expander(f"[{i}] {ch.source} · {ch.published_at[:10]} · {ch.title}"):
                    st.write(ch.text)

with tab_add:
    st.write("Paste a new article. It is chunked, embedded, read by the LLM and **merged into the existing "
             "graph**. Nothing else is rebuilt.")
    title = st.text_input("Title", "OpenAI board names new interim CEO")
    c1, c2 = st.columns(2)
    source = c1.text_input("Source", "Demo News")
    date = c2.date_input("Published", dt.date(2023, 12, 28))
    body = st.text_area("Article text", height=200, value=(
        "OpenAI's board announced on Thursday that Mira Murati will serve as interim CEO of OpenAI. "
        "Murati, who previously led the company's technology efforts, replaces Sam Altman. "
        "Microsoft, OpenAI's largest investor, said it remains committed to the partnership."))
    if st.button("Add to graph", type="primary"):
        with st.spinner("chunk → embed → LLM extract → merge into graph ..."):
            out = add_article(cfg, store, fm, get_embedder(), get_llm(),
                              NewArticle(title, body, source, f"{date.isoformat()}T12:00:00+00:00",
                                         url=f"demo://{int(time.time())}"))
        if not out.get("inserted"):
            st.warning(out["message"])
        else:
            st.session_state.graph_version += 1
            secs = out["seconds"]
            st.success(f"Added in {secs['total']}s (chunk+embed {secs['chunk_embed']}s · LLM {secs['llm_extract']}s"
                       f" · graph update {secs['graph_update']}s)")
            c1, c2 = st.columns(2)
            c1.markdown("**Linked to existing entities**")
            c1.write(out["entities_linked_to_existing"] or "—")
            c2.markdown("**New entities**")
            c2.write(out["entities_new"] or "—")
            st.markdown("**Relations from this article**")
            st.dataframe(out["relations"], width="stretch", hide_index=True)
            if out["superseded_old_facts"]:
                st.markdown("**Older facts now marked as replaced**")
                st.dataframe(out["superseded_old_facts"], width="stretch", hide_index=True)

with tab_graph:
    st.subheader("Most connected entities")
    hubs = store.conn.execute(
        "SELECT e.name, e.type, e.mention_count, (SELECT COUNT(*) FROM relations r WHERE (r.src_id = e.id OR "
        "r.tgt_id = e.id) AND r.status != 'superseded') AS degree FROM entities e ORDER BY degree DESC LIMIT 20"
    ).fetchall()
    st.dataframe([dict(name=h[0], type=h[1], mentions=h[2], degree=h[3]) for h in hubs], width="stretch",
                 hide_index=True)
    st.subheader("Facts replaced by newer articles")
    sup = store.conn.execute(
        "SELECT s.name, r.predicate, t.name, substr(r.valid_from,1,10), substr(r.valid_to,1,10), "
        "(SELECT s2.name || ' ' || n.predicate || ' ' || t2.name FROM relations n JOIN entities s2 ON s2.id = n.src_id "
        " JOIN entities t2 ON t2.id = n.tgt_id WHERE n.id = r.superseded_by) FROM relations r "
        "JOIN entities s ON s.id = r.src_id JOIN entities t ON t.id = r.tgt_id WHERE r.status = 'superseded' "
        "ORDER BY r.valid_to DESC LIMIT 50").fetchall()
    st.dataframe([dict(old_fact=f"{a} {b} {c}", valid_from=d, valid_to=e, replaced_by=f) for a, b, c, d, e, f in sup],
                 width="stretch", hide_index=True)
