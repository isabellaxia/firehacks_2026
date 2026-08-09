import asyncio
import json
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import agent
import db
import llm
import tools

HERE = os.path.dirname(__file__)
app = FastAPI(title="Nearmiss")
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/source")
async def source():
    """Where the reports came from. Shown on screen so nobody has to ask."""
    con = db.connect()
    m = {r["k"]: r["v"] for r in con.execute("SELECT k, v FROM meta")}
    if not m:
        m = {"source": "SYNTHETIC", "is_real": "0",
             "source_detail": "No provenance recorded. Re-run ingest.py."}
    n = con.execute("SELECT COUNT(*) c FROM reports").fetchone()["c"]
    span = con.execute("SELECT MIN(ym) a, MAX(ym) b FROM reports").fetchone()
    fields = con.execute("SELECT COUNT(DISTINCT airport) c FROM reports").fetchone()["c"]
    modes = con.execute("SELECT COUNT(*) c FROM clusters").fetchone()["c"]
    return {**m, "n_reports": n, "span": f"{span['a']}..{span['b']}",
            "fields": fields, "failure_modes": modes}


@app.get("/api/health")
async def health():
    """Confirms the database loaded and the configured models exist on Featherless."""
    con = db.connect()
    n = con.execute("SELECT COUNT(*) c FROM reports").fetchone()["c"]
    span = con.execute("SELECT MIN(ym) a, MAX(ym) b FROM reports").fetchone()
    wanted = [agent.AGENT_MODEL,
              os.environ.get("EXTRACT_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
              os.environ.get("VERIFY_MODEL", "mistralai/Mistral-Small-24B-Instruct-2501")]
    try:
        available = set(await llm.list_models())
        models = {m: (m in available) for m in wanted}
        catalog = len(available)
    except Exception as e:
        models, catalog = {m: f"check failed: {e}" for m in wanted}, None
    return {"reports": n, "span": [span["a"], span["b"]], "models": models,
            "featherless_catalog_size": catalog, "key_set": bool(llm.API_KEY)}


@app.get("/api/airports")
async def airports():
    con = db.connect()
    rows = con.execute(
        "SELECT a.icao, a.name, COUNT(r.id) n FROM airports a "
        "LEFT JOIN reports r ON r.airport = a.icao GROUP BY a.icao ORDER BY n DESC").fetchall()
    return [dict(r) for r in rows]


@app.get("/api/clusters")
async def clusters(airport: str = ""):
    return await tools.cluster_incidents(airport) if airport else \
        {"clusters": [dict(r) for r in db.connect().execute(
            "SELECT * FROM clusters ORDER BY n DESC")]}


@app.get("/api/ask")
async def ask(q: str):
    """SSE. Each tool call and result is pushed as it happens."""
    async def stream():
        queue: asyncio.Queue = asyncio.Queue()

        async def emit(kind, payload):
            await queue.put((kind, payload))

        async def work():
            try:
                await agent.run(q, emit)
            except Exception as e:
                await queue.put(("error", {"message": f"{type(e).__name__}: {e}"}))
            finally:
                await queue.put(("done", {}))

        task = asyncio.create_task(work())
        try:
            while True:
                kind, payload = await queue.get()
                yield f"event: {kind}\ndata: {json.dumps(payload)}\n\n"
                if kind == "done":
                    break
        finally:
            task.cancel()

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/watch")
async def watch(months: int = 24, recent_months: int = 6, max_p: float = 0.05):
    """Standing watch: every airport x failure mode, ranked. No model calls."""
    return await tools.scan_corpus(months, recent_months, max_p)


@app.get("/api/leadtime")
async def leadtime(cluster_id: str, airport: str, incident_ym: str):
    return await tools.lead_time(cluster_id, airport, incident_ym)


@app.get("/api/detail")
async def detail(cluster_id: str, airport: str):
    return await tools.get_cluster_detail(cluster_id, airport)


@app.get("/api/backtest")
async def backtest(cluster_id: str, airport: str, cutoff_ym: str):
    return await tools.backtest_asof(cluster_id, airport, cutoff_ym)

