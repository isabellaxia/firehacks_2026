"""The tools the agent can call.

House rule, and it is the whole safety argument of this project:

    Every number returned to the user is computed by Python from the database.
    Models read narratives, name patterns, and write prose. Models never produce
    a statistic, a count, a rate, or a date.

compute_trend / compare_baseline / backtest_asof contain no model calls at all.
"""
import json
import math
import os
from collections import Counter

import db
from llm import complete

MAX_ROWS = 400


# ---------------------------------------------------------------- statistics

def _binom_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p). Exact, no scipy."""
    if n == 0:
        return 1.0
    k = max(0, min(k, n))
    total = 0.0
    logp, logq = math.log(max(p, 1e-12)), math.log(max(1 - p, 1e-12))
    for i in range(k, n + 1):
        total += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                          + i * logp + (n - i) * logq)
    return min(1.0, total)


def _slope(counts):
    """Least-squares slope of monthly counts, in reports per month."""
    n = len(counts)
    if n < 3:
        return 0.0
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(counts) / n
    den = sum((x - mx) ** 2 for x in xs)
    return 0.0 if den == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, counts)) / den


def _month_range(end_ym: str, months: int):
    y, m = int(end_ym[:4]), int(end_ym[5:7])
    out = []
    for _ in range(months):
        out.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return list(reversed(out))


def _latest_ym(con) -> str:
    r = con.execute("SELECT MAX(ym) AS m FROM reports").fetchone()
    return r["m"] or "1970-01"


def _series(con, cluster_id, airport, window):
    rows = con.execute(
        "SELECT ym, COUNT(*) n FROM reports WHERE cluster_id=? AND airport=? "
        f"AND ym IN ({','.join('?' * len(window))}) GROUP BY ym",
        [cluster_id, airport, *window]).fetchall()
    counts = {r["ym"]: r["n"] for r in rows}
    return [counts.get(ym, 0) for ym in window]


def _trend_core(con, cluster_id, airport, end_ym, months, recent_months):
    window = _month_range(end_ym, months)
    counts = _series(con, cluster_id, airport, window)
    recent, baseline = counts[-recent_months:], counts[:-recent_months]
    n_rec, n_base = sum(recent), sum(baseline)
    rate_rec = n_rec / max(1, len(recent))
    rate_base = n_base / max(1, len(baseline))
    total, p0 = n_rec + n_base, len(recent) / months
    p_value = _binom_tail(n_rec, total, p0)
    return {
        "cluster_id": cluster_id,
        "airport": airport,
        "window": f"{window[0]}..{window[-1]}",
        "monthly_counts": dict(zip(window, counts)),
        "recent_months": len(recent),
        "recent_reports": n_rec,
        "recent_rate_per_month": round(rate_rec, 2),
        "baseline_rate_per_month": round(rate_base, 2),
        "rate_ratio": round(rate_rec / rate_base, 2) if rate_base > 0 else None,
        "slope_reports_per_month": round(_slope(counts), 3),
        "p_value": round(p_value, 5),
        "significant_at_0.01": p_value < 0.01,
        "note": "Counts and statistics computed in Python from the report database. "
                "No language model produced these numbers.",
    }


# ---------------------------------------------------------------- the tools

async def search_reports(query: str = "", airport: str = "", months: int = 24,
                         limit: int = 12) -> dict:
    """Full-text search over narratives, newest first."""
    con = db.connect()
    window = _month_range(_latest_ym(con), max(1, months))
    sql = ("SELECT r.report_id, r.ym, r.airport, r.aircraft, r.phase, r.cluster_id, "
           "substr(r.narrative,1,600) AS narrative FROM reports r ")
    params, where = [], [f"r.ym IN ({','.join('?' * len(window))})"]
    params += window
    if query.strip():
        sql += "JOIN reports_fts f ON f.rowid = r.id "
        where.append("reports_fts MATCH ?")
        params.append(" OR ".join(w for w in query.split() if len(w) > 2) or query)
    if airport.strip():
        where.append("r.airport = ?")
        params.append(airport.strip().upper())
    sql += "WHERE " + " AND ".join(where) + " ORDER BY r.ym DESC LIMIT ?"
    params.append(min(int(limit), 25))
    rows = [dict(r) for r in con.execute(sql, params)]
    return {"n": len(rows), "reports": rows}


async def cluster_incidents(airport: str, months: int = 24) -> dict:
    """What failure modes exist at this field, and how big is each."""
    con = db.connect()
    window = _month_range(_latest_ym(con), max(1, months))
    rows = con.execute(
        "SELECT r.cluster_id, c.label, c.keywords, COUNT(*) n FROM reports r "
        "LEFT JOIN clusters c ON c.cluster_id = r.cluster_id "
        f"WHERE r.airport=? AND r.ym IN ({','.join('?' * len(window))}) "
        "GROUP BY r.cluster_id ORDER BY n DESC",
        [airport.upper(), *window]).fetchall()
    return {"airport": airport.upper(), "window": f"{window[0]}..{window[-1]}",
            "clusters": [dict(r) for r in rows]}


async def compute_trend(cluster_id: str, airport: str, months: int = 24,
                        recent_months: int = 6) -> dict:
    """Is this failure mode getting more common here. Pure Python."""
    con = db.connect()
    return _trend_core(con, cluster_id, airport.upper(), _latest_ym(con),
                       max(6, int(months)), max(2, int(recent_months)))


async def compare_baseline(cluster_id: str, airport: str, months: int = 24,
                           recent_months: int = 6) -> dict:
    """Same cluster at every other field. Separates a local signal from a national one."""
    con = db.connect()
    end = _latest_ym(con)
    others = [r["airport"] for r in con.execute(
        "SELECT DISTINCT airport FROM reports WHERE airport <> ?", [airport.upper()])]
    peers = [_trend_core(con, cluster_id, a, end, months, recent_months) for a in others]
    ratios = [p["rate_ratio"] for p in peers if p["rate_ratio"] is not None]
    target = _trend_core(con, cluster_id, airport.upper(), end, months, recent_months)
    national = round(sum(ratios) / len(ratios), 2) if ratios else None
    return {
        "target": {"airport": target["airport"], "rate_ratio": target["rate_ratio"],
                   "p_value": target["p_value"]},
        "peer_median_rate_ratio": national,
        "peers": sorted([{"airport": p["airport"], "rate_ratio": p["rate_ratio"]} for p in peers],
                        key=lambda x: (x["rate_ratio"] is None, -(x["rate_ratio"] or 0))),
        "interpretation_hint": "If the target ratio is high and peers sit near 1.0, the pattern is "
                               "local. If every field rose together, it is a national or reporting "
                               "artifact, not an airport-specific risk.",
    }


async def get_airport_context(icao: str) -> dict:
    """Runway layout and known quirks, so the agent can reason about plausibility."""
    con = db.connect()
    r = con.execute("SELECT * FROM airports WHERE icao=?", [icao.upper()]).fetchone()
    if not r:
        return {"icao": icao.upper(), "found": False}
    total = con.execute("SELECT COUNT(*) c FROM reports WHERE airport=?", [icao.upper()]).fetchone()["c"]
    return {**dict(r), "found": True, "reports_on_file": total}


async def extract_causal_chain(report_id: str) -> dict:
    """Model reads one narrative and returns structured causation. No numbers."""
    con = db.connect()
    r = con.execute("SELECT report_id, narrative FROM reports WHERE report_id=?",
                    [report_id]).fetchone()
    if not r:
        return {"report_id": report_id, "found": False}
    out = await complete(
        os.environ.get("EXTRACT_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
        system="You extract causal structure from aviation safety reports. Reply with JSON only, "
               "no prose, no code fences. Keys: trigger, compounding_factors (array), "
               "what_caught_it, latent_condition. Each value is one short sentence. If the "
               "narrative does not say, use null. Never invent details.",
        user=r["narrative"][:4000], max_tokens=400)
    try:
        parsed = json.loads(out.strip().strip("`").removeprefix("json").strip())
    except Exception:
        parsed = {"raw": out[:600], "parse_error": True}
    return {"report_id": report_id, "found": True, "chain": parsed}


async def verify_finding(claim: str, report_ids: str = "") -> dict:
    """A different model family checks the claim against the actual narratives."""
    con = db.connect()
    ids = [s.strip() for s in report_ids.split(",") if s.strip()][:8]
    rows = con.execute(
        f"SELECT report_id, substr(narrative,1,700) t FROM reports WHERE report_id IN "
        f"({','.join('?' * len(ids))})", ids).fetchall() if ids else []
    evidence = "\n\n".join(f"[{r['report_id']}] {r['t']}" for r in rows) or "(no reports supplied)"
    out = await complete(
        os.environ.get("VERIFY_MODEL", "mistralai/Mistral-Small-24B-Instruct-2501"),
        system="You are an adversarial reviewer. Decide whether the CLAIM is supported by the "
               "EVIDENCE. Reply JSON only: {verdict: supported|partly|unsupported, reason: one "
               "sentence, unsupported_parts: array of strings}. Be strict. If the claim asserts a "
               "cause the reports do not state, say unsupported.",
        user=f"CLAIM:\n{claim}\n\nEVIDENCE:\n{evidence[:6000]}", max_tokens=350)
    try:
        parsed = json.loads(out.strip().strip("`").removeprefix("json").strip())
    except Exception:
        parsed = {"verdict": "unknown", "reason": out[:300]}
    return {"claim": claim, "checked_against": [r["report_id"] for r in rows], **parsed}


_scan_cache: dict = {}


async def scan_corpus(months: int = 24, recent_months: int = 6,
                      max_p: float = 0.05, min_reports: int = 6) -> dict:
    """Sweep every airport x cluster pair and rank what is rising. Pure Python.

    Nobody tells it where to look. This is the standing watch.
    """
    key = (months, recent_months, max_p, min_reports)
    if key in _scan_cache:
        return _scan_cache[key]

    con = db.connect()
    end = _latest_ym(con)
    labels = {r["cluster_id"]: r["label"] for r in con.execute("SELECT cluster_id, label FROM clusters")}
    names = {r["icao"]: r["name"] for r in con.execute("SELECT icao, name FROM airports")}
    pairs = con.execute(
        "SELECT airport, cluster_id, COUNT(*) n FROM reports GROUP BY airport, cluster_id "
        "HAVING n >= ?", [min_reports]).fetchall()

    findings = []
    for p in pairs:
        t = _trend_core(con, p["cluster_id"], p["airport"], end, months, recent_months)
        if t["p_value"] <= max_p and (t["rate_ratio"] or 0) > 1.0:
            findings.append({
                "airport": p["airport"], "airport_name": names.get(p["airport"], ""),
                "cluster_id": p["cluster_id"], "label": labels.get(p["cluster_id"], ""),
                "rate_ratio": t["rate_ratio"], "p_value": t["p_value"],
                "recent_reports": t["recent_reports"],
                "recent_rate_per_month": t["recent_rate_per_month"],
                "baseline_rate_per_month": t["baseline_rate_per_month"],
                "monthly_counts": t["monthly_counts"],
            })
    findings.sort(key=lambda f: (f["p_value"], -(f["rate_ratio"] or 0)))

    out = {"scanned_pairs": len(pairs), "window": f"through {end}",
           "flagged": len(findings), "findings": findings[:12],
           "note": "Every pair tested in Python. No language model involved in this scan."}
    _scan_cache[key] = out
    return out


async def lead_time(cluster_id: str, airport: str, incident_ym: str,
                    months: int = 18, recent_months: int = 6) -> dict:
    """Walk the cutoff backwards month by month: how early was this detectable?

    Answers the only question that matters about a prediction system.
    """
    con = db.connect()
    all_ym = [r["ym"] for r in con.execute("SELECT DISTINCT ym FROM reports ORDER BY ym")]
    candidates = [y for y in all_ym if y < incident_ym][-24:]

    trail, first_flag = [], None
    for cutoff in candidates:
        y, m = int(cutoff[:4]), int(cutoff[5:7])
        m -= 1
        if m == 0:
            y, m = y - 1, 12
        t = _trend_core(con, cluster_id, airport.upper(), f"{y:04d}-{m:02d}", months, recent_months)
        total = sum(t["monthly_counts"].values())
        # Guard against flagging on almost-empty early windows, where two reports in a
        # quiet month produce a huge ratio that means nothing.
        flagged = bool(t["p_value"] < 0.01 and (t["rate_ratio"] or 0) >= 1.8
                       and t["recent_reports"] >= 5 and total >= 12)
        trail.append({"cutoff_ym": cutoff, "rate_ratio": t["rate_ratio"],
                      "p_value": t["p_value"], "flagged": flagged})
        if flagged and first_flag is None:
            first_flag = cutoff

    months_early = None
    if first_flag:
        a = int(first_flag[:4]) * 12 + int(first_flag[5:7])
        b = int(incident_ym[:4]) * 12 + int(incident_ym[5:7])
        months_early = b - a

    return {"cluster_id": cluster_id, "airport": airport.upper(),
            "incident_ym": incident_ym, "first_flagged_ym": first_flag,
            "months_of_warning": months_early, "trail": trail,
            "note": "Each row uses only reports filed before that month. "
                    "No information from after the cutoff is visible to the test."}


async def get_cluster_detail(cluster_id: str, airport: str, limit: int = 6) -> dict:
    """The actual narratives behind a flagged pattern, newest first."""
    con = db.connect()
    rows = con.execute(
        "SELECT report_id, ym, aircraft, phase, substr(narrative,1,420) AS narrative "
        "FROM reports WHERE cluster_id=? AND airport=? ORDER BY ym DESC LIMIT ?",
        [cluster_id, airport.upper(), min(int(limit), 12)]).fetchall()
    c = con.execute("SELECT label, keywords FROM clusters WHERE cluster_id=?",
                    [cluster_id]).fetchone()
    return {"cluster_id": cluster_id, "airport": airport.upper(),
            "label": c["label"] if c else "", "keywords": c["keywords"] if c else "",
            "reports": [dict(r) for r in rows]}


async def backtest_asof(cluster_id: str, airport: str, cutoff_ym: str,
                        months: int = 18, recent_months: int = 6) -> dict:
    """Recompute the trend using ONLY data before cutoff_ym. This is the honesty check."""
    con = db.connect()
    y, m = int(cutoff_ym[:4]), int(cutoff_ym[5:7])
    m -= 1
    if m == 0:
        y, m = y - 1, 12
    end = f"{y:04d}-{m:02d}"
    res = _trend_core(con, cluster_id, airport.upper(), end, months, recent_months)
    res["cutoff_ym"] = cutoff_ym
    res["data_used_through"] = end
    res["would_have_flagged"] = bool(res["significant_at_0.01"] and (res["rate_ratio"] or 0) >= 1.8)
    return res


REGISTRY = {f.__name__: f for f in [
    search_reports, cluster_incidents, compute_trend, compare_baseline,
    get_airport_context, extract_causal_chain, verify_finding, backtest_asof,
    scan_corpus, lead_time, get_cluster_detail,
]}


def _schema(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}


S = lambda d: {"type": "string", "description": d}
I = lambda d: {"type": "integer", "description": d}

TOOL_SCHEMAS = [
    _schema("search_reports", "Full-text search of near-miss narratives. Use this first to see "
            "what crews are actually reporting.",
            {"query": S("keywords, e.g. 'parallel runway confusion'"),
             "airport": S("ICAO like KSFO, or empty for all"),
             "months": I("lookback window, default 24"), "limit": I("max 25")}, []),
    _schema("cluster_incidents", "List the failure-mode clusters present at an airport with "
            "report counts. Use this to find candidate patterns before measuring any of them.",
            {"airport": S("ICAO"), "months": I("default 24")}, ["airport"]),
    _schema("compute_trend", "Measure whether one cluster is rising at one airport. Returns "
            "monthly counts, rate ratio and a p-value, all computed in Python.",
            {"cluster_id": S("like C03"), "airport": S("ICAO"), "months": I("default 24"),
             "recent_months": I("default 6")}, ["cluster_id", "airport"]),
    _schema("compare_baseline", "Compare the same cluster's trend at every other airport, to "
            "tell a local risk from a national reporting shift. Always call this before "
            "concluding a pattern is airport-specific.",
            {"cluster_id": S("like C03"), "airport": S("ICAO"), "months": I("default 24"),
             "recent_months": I("default 6")}, ["cluster_id", "airport"]),
    _schema("get_airport_context", "Runway layout, annual operations and known quirks for an "
            "airport. Use it to judge whether a pattern is physically plausible.",
            {"icao": S("ICAO")}, ["icao"]),
    _schema("extract_causal_chain", "Read one report and return its trigger, compounding "
            "factors, what caught it, and the latent condition.",
            {"report_id": S("report id from search_reports")}, ["report_id"]),
    _schema("verify_finding", "Have an independent model check your conclusion against the "
            "source narratives. Call this on your final finding before you answer.",
            {"claim": S("your one-sentence finding"),
             "report_ids": S("comma-separated ids that should support it")}, ["claim"]),
    _schema("scan_corpus", "Sweep every airport and every failure mode at once and rank what is "
            "rising. Use this when the question is open-ended, e.g. 'what is most concerning right "
            "now' with no airport named.",
            {"months": I("default 24"), "recent_months": I("default 6"),
             "max_p": {"type": "number", "description": "significance cutoff, default 0.05"}}, []),
    _schema("lead_time", "Walk the cutoff month backwards to find the earliest month this pattern "
            "would have been flagged, and how many months of warning that gives before a known "
            "incident.",
            {"cluster_id": S("like C06"), "airport": S("ICAO"),
             "incident_ym": S("YYYY-MM of the incident")}, ["cluster_id", "airport", "incident_ym"]),
    _schema("get_cluster_detail", "Pull the actual narratives behind a flagged pattern at one "
            "airport, newest first.",
            {"cluster_id": S("like C06"), "airport": S("ICAO"), "limit": I("max 12")},
            ["cluster_id", "airport"]),
    _schema("backtest_asof", "Recompute a trend using only data before a cutoff month, to test "
            "whether the pattern was detectable in advance.",
            {"cluster_id": S("like C03"), "airport": S("ICAO"),
             "cutoff_ym": S("YYYY-MM, the month to pretend is 'now'"),
             "months": I("default 18"), "recent_months": I("default 6")},
            ["cluster_id", "airport", "cutoff_ym"]),
]

