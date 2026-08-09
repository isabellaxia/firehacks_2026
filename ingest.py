"""Offline ingestion. Run this on YOUR laptop, commit the resulting .db, deploy that.

Never run this on Render — clustering needs more RAM than the free tier has.

    python make_seed.py                       # synthetic corpus (works immediately)
    python ingest.py --csv data/seed_reports.csv
    python ingest.py --csv asrs_export.csv --asrs   # real NASA ASRS export

Real ASRS data: https://asrs.arc.nasa.gov/search/database.html
Export as CSV. The --asrs flag maps their column names onto ours; ASRS changes
their export headers occasionally, so check COLUMN_GUESSES if it fails.
"""
import argparse
import csv
import re
import sys
from collections import Counter, defaultdict

import db

# ASRS exports are messy and their headers move. These are candidates, tried in order.
COLUMN_GUESSES = {
    "narrative": ["Narrative", "Report 1 Narrative", "Narrative_1", "Report 1_Narrative"],
    "ym": ["Date", "Time / Date", "Time_Date"],
    "airport": ["Locale Reference", "Place_Locale Reference", "Locale Reference.ATC Facility"],
    "aircraft": ["Make Model Name", "Aircraft_Make Model Name", "Aircraft 1_Make Model Name"],
    "phase": ["Flight Phase", "Aircraft_Flight Phase", "Aircraft 1_Flight Phase"],
}

AIRPORT_FACTS = {
    "KSFO": ("San Francisco Intl", "28L/28R (750ft sep), 01L/01R", 460000,
             "Closely spaced parallels. Simultaneous approaches to 28L/28R are common in VMC "
             "and the designators differ by one character."),
    "KLAX": ("Los Angeles Intl", "24L/24R, 25L/25R, 06L/06R, 07L/07R", 700000,
             "Two parallel pairs on each side of the terminal core. High volume of runway "
             "crossings between the inboard and outboard runways."),
    "KSEA": ("Seattle-Tacoma Intl", "16L/16C/16R, 34L/34C/34R", 440000,
             "Triple parallels. Center runway assignment changes frequently."),
    "KPDX": ("Portland Intl", "10L/10R, 28L/28R, 03/21", 200000, "Lower volume field."),
    "KBOS": ("Boston Logan Intl", "04L/04R, 09/27, 15R/33L, 22L/22R", 400000,
             "Converging runway operations. Short taxi distances between the terminal and 09/27 "
             "leave little cooling roll on quick turns."),
    "KORD": ("Chicago O'Hare Intl", "09L/09R/09C, 10L/10R/10C, 27L/27R/27C, 28L/28R/28C", 730000,
             "Six east-west parallels. Frequency congestion is chronic."),
    "KDEN": ("Denver Intl", "16L/16R, 17L/17R, 34L/34R, 35L/35R, 07/25", 620000,
             "Widely spaced runways, long taxi distances, frequent winter deicing."),
    "KATL": ("Hartsfield-Jackson Atlanta Intl", "08L/08R, 09L/09R, 26L/26R, 27L/27R", 780000,
             "Five parallels, highest movement count in the US. Triple simultaneous approaches."),
}

STOP = set("""the a an and or of to in on at is was were be been being for with as by from that this
it we our us i they he she his her not no but if then than so such into over under out up down
were are had has have do did does can could would should will may might must about after before
during while when where which who whom whose there their them these those other another each any
all both few more most some no nor only own same too very just also aircraft flight crew captain
first officer report reported filed feet ft ftl knots kt kts approximately approx due per via
""".split())


def norm_ym(raw: str) -> str:
    """ASRS dates look like '202405'. Ours look like '2024-05'."""
    s = re.sub(r"[^0-9]", "", str(raw))
    if len(s) >= 6:
        return f"{s[:4]}-{s[4:6]}"
    return ""


def norm_airport(raw: str) -> str:
    """'SFO.Airport' / 'SFO' / 'KSFO' -> 'KSFO'."""
    s = str(raw).upper()
    m = re.search(r"\b([A-Z]{3,4})\b", s)
    if not m:
        return ""
    code = m.group(1)
    if len(code) == 3:
        code = "K" + code
    return code


def pick(header, candidates):
    for c in candidates:
        if c in header:
            return c
    low = {h.lower(): h for h in header}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def read_rows(path: str, asrs: bool):
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        rows = list(csv.reader(f))
    if not asrs:
        header, body = rows[0], rows[1:]
        idx = {h: i for i, h in enumerate(header)}
        for i, r in enumerate(body):
            yield {
                "report_id": r[idx["report_id"]] if "report_id" in idx else f"R{i:06d}",
                "ym": r[idx["ym"]],
                "airport": r[idx["airport"]].upper(),
                "aircraft": r[idx.get("aircraft", 0)] if "aircraft" in idx else "",
                "phase": r[idx["phase"]] if "phase" in idx else "",
                "narrative": r[idx["narrative"]],
            }
        return
    # ASRS exports carry two header rows (section, then field).
    h0, h1, body = rows[0], rows[1], rows[2:]
    header = [f"{a}_{b}".strip("_") if a else b for a, b in zip(h0, h1)]
    cols = {k: pick(header, v) for k, v in COLUMN_GUESSES.items()}
    if not cols["narrative"]:
        sys.exit(f"Could not find a narrative column. Headers seen:\n{header}")
    idx = {h: i for i, h in enumerate(header)}
    for i, r in enumerate(body):
        def g(key):
            c = cols.get(key)
            return r[idx[c]] if c and idx[c] < len(r) else ""
        text = g("narrative")
        if len(text) < 120:
            continue
        yield {
            "report_id": f"ASRS{i:06d}",
            "ym": norm_ym(g("ym")),
            "airport": norm_airport(g("airport")),
            "aircraft": g("aircraft")[:40],
            "phase": g("phase")[:40],
            "narrative": text,
        }


def cluster(narratives, k):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3,
                          max_df=0.45, stop_words="english", sublinear_tf=True)
    X = vec.fit_transform(narratives)
    km = MiniBatchKMeans(n_clusters=k, random_state=17, n_init=8, batch_size=1024)
    labels = km.fit_predict(X)
    terms = vec.get_feature_names_out()
    keywords = {}
    for c in range(k):
        order = km.cluster_centers_[c].argsort()[::-1][:8]
        keywords[c] = [terms[i] for i in order]
    return labels, keywords


def label_from_keywords(words):
    """Readable cluster name without calling a model — ingestion stays offline and free.

    You can optionally re-label clusters with an LLM later; the id never changes.
    """
    picked = [w for w in words if w not in STOP][:3]
    return " / ".join(picked) if picked else "unlabeled"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--asrs", action="store_true", help="input is a raw NASA ASRS export")
    ap.add_argument("--clusters", type=int, default=14)
    args = ap.parse_args()

    rows = [r for r in read_rows(args.csv, args.asrs) if r["ym"] and r["airport"] and r["narrative"]]
    if not rows:
        sys.exit("No usable rows.")
    print(f"read {len(rows)} reports")

    k = min(args.clusters, max(2, len(rows) // 50))
    labels, keywords = cluster([r["narrative"] for r in rows], k)

    counts = Counter(labels)
    con = db.connect(write=True)
    db.init(con)
    con.executescript("DELETE FROM reports; DELETE FROM clusters; DELETE FROM airports; "
                      "DELETE FROM reports_fts;")

    for r, lab in zip(rows, labels):
        r["cluster_id"] = f"C{int(lab):02d}"
    con.executemany(
        "INSERT OR IGNORE INTO reports (report_id, ym, airport, aircraft, phase, narrative, cluster_id) "
        "VALUES (:report_id, :ym, :airport, :aircraft, :phase, :narrative, :cluster_id)", rows)

    con.executemany(
        "INSERT OR REPLACE INTO clusters (cluster_id, label, keywords, n) VALUES (?,?,?,?)",
        [(f"C{c:02d}", label_from_keywords(keywords[c]), ", ".join(keywords[c]), counts[c])
         for c in range(k)])

    seen = defaultdict(int)
    for r in rows:
        seen[r["airport"]] += 1
    con.executemany(
        "INSERT OR REPLACE INTO airports (icao, name, runways, ops_year, notes) VALUES (?,?,?,?,?)",
        [(a, *AIRPORT_FACTS.get(a, (a, "unknown", 0, "No context on file for this field.")))
         for a in seen])

    con.execute("INSERT INTO reports_fts(rowid, narrative) SELECT id, narrative FROM reports")
    con.commit()

    print(f"clusters ({k}):")
    for c in range(k):
        print(f"  C{c:02d}  n={counts[c]:5d}  {label_from_keywords(keywords[c])}")
    print(f"airports: {sorted(seen)}")
    print(f"wrote {db.DB_PATH}")


if __name__ == "__main__":
    main()

