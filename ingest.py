"""Offline ingestion. Run on YOUR laptop, commit the resulting .db, deploy that.

REAL DATA (what you want for judging):

    1. Go to https://asrs.arc.nasa.gov/search/database.html
    2. Run a search. Set a date range and leave the rest broad.
    3. Download the results as CSV.
    4. python ingest.py --csv ASRS_DBOnline.csv --asrs

SYNTHETIC FALLBACK (works with no download, clearly labelled as fake in the UI):

    python make_seed.py
    python ingest.py --csv data/seed_reports.csv

The database records which one you used, and the web page says so on screen.
Never run this on Render — clustering needs more RAM than the free tier has.
"""
import argparse
import csv
import re
import sys
from collections import Counter
from datetime import datetime, timezone

import db

csv.field_size_limit(10_000_000)  # ASRS narratives are long

# ASRS renames export headers periodically, so we try names first and then fall back
# to picking the column that actually contains long prose.
NAME_HINTS = {
    "narrative": ["narrative", "report 1", "report 2", "callback", "synopsis"],
    "ym": ["date", "time"],
    "airport": ["locale reference", "place", "location", "facility"],
    "aircraft": ["make model", "aircraft"],
    "phase": ["flight phase", "phase"],
}

# Optional colour for well-known fields. Anything not listed still works; the app
# just reports "no context on file", which is honest.
AIRPORT_FACTS = {
    "KSFO": ("San Francisco Intl", "28L/28R (750ft sep), 01L/01R", 460000,
             "Closely spaced parallels; simultaneous approaches to 28L/28R are routine in VMC."),
    "KLAX": ("Los Angeles Intl", "24L/24R, 25L/25R, 06L/06R, 07L/07R", 700000,
             "Two parallel pairs; high volume of runway crossings between inboard and outboard."),
    "KSEA": ("Seattle-Tacoma Intl", "16L/16C/16R, 34L/34C/34R", 440000,
             "Triple parallels; centre runway assignment changes frequently."),
    "KBOS": ("Boston Logan Intl", "04L/04R, 09/27, 15R/33L, 22L/22R", 400000,
             "Converging runway operations and short taxi distances on quick turns."),
    "KORD": ("Chicago O'Hare Intl", "09L/C/R, 10L/C/R, 27L/C/R, 28L/C/R", 730000,
             "Six east-west parallels; frequency congestion is chronic."),
    "KDEN": ("Denver Intl", "16L/16R, 17L/17R, 34L/34R, 35L/35R, 07/25", 620000,
             "Widely spaced runways, long taxi distances, frequent winter deicing."),
    "KATL": ("Hartsfield-Jackson Atlanta Intl", "08L/08R, 09L/09R, 26L/26R, 27L/27R", 780000,
             "Five parallels, highest movement count in the US."),
    "KJFK": ("John F. Kennedy Intl", "04L/04R, 13L/13R, 22L/22R, 31L/31R", 460000,
             "Intersecting runways and complex taxi routes."),
    "KEWR": ("Newark Liberty Intl", "04L/04R, 11/29, 22L/22R", 440000,
             "Closely spaced parallels inside congested New York airspace."),
    "KLGA": ("LaGuardia", "04/22, 13/31", 370000, "Short intersecting runways, tight airspace."),
    "KPHX": ("Phoenix Sky Harbor", "07L/07R, 08, 25L/25R, 26", 440000, "Triple parallel operations."),
    "KLAS": ("Harry Reid Intl", "01L/01R, 08L/08R, 19L/19R, 26L/26R", 540000,
             "Intersecting parallel pairs."),
    "KDFW": ("Dallas/Fort Worth Intl", "17L/C/R, 18L/R, 35L/C/R, 36L/R", 720000,
             "Seven runways, high volume."),
    "KMIA": ("Miami Intl", "08L/08R, 09, 26L/26R, 27", 420000, "Heavy international widebody mix."),
    "KPDX": ("Portland Intl", "10L/10R, 28L/28R, 03/21", 200000, "Lower volume field."),
}

STOP = set("""aircraft flight crew captain first officer report reported filed feet ft knots kt kts
approximately approx due per via said told stated advised acft rwy twy atc zzz zzz1 zzz2 air
carrier pilot would could asked
""".split())

JARGON = {"ATC", "IFR", "VFR", "TCAS", "GPWS", "PIC", "FAA", "AGL", "MSL", "RWY", "TWY",
          "SID", "STAR", "ILS", "VMC", "IMC", "CRM", "FMS", "APU", "EFB", "NAS", "NOTAM",
          "TRACON", "FBO", "PAX", "QRH", "MEL", "TFR", "CFR", "AOA"}


def norm_ym(raw: str) -> str:
    s = re.sub(r"[^0-9]", "", str(raw))
    if len(s) >= 6 and 1980 <= int(s[:4]) <= 2100 and 1 <= int(s[4:6]) <= 12:
        return f"{s[:4]}-{s[4:6]}"
    m = re.search(r"(19|20)(\d\d)[-/](\d\d?)", str(raw))
    if m:
        return f"{m.group(1)}{m.group(2)}-{int(m.group(3)):02d}"
    return ""


def norm_airport(raw: str, narrative: str = "") -> str:
    """ASRS anonymises many locations as ZZZ. Prefer a real code; fall back to the narrative."""
    for src in (str(raw), narrative[:400]):
        for m in re.finditer(r"\b([A-Z]{3,4})\b", src.upper()):
            code = m.group(1)
            if code.startswith("ZZ") or code in JARGON:
                continue
            return code if len(code) == 4 else "K" + code
    return ""


def load(path, asrs):
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        rows = list(csv.reader(f))
    if not rows:
        sys.exit("empty file")

    if not asrs:
        header, body = rows[0], rows[1:]
        ix = {h: i for i, h in enumerate(header)}
        for i, r in enumerate(body):
            yield {"report_id": r[ix["report_id"]] if "report_id" in ix else f"R{i:06d}",
                   "ym": r[ix["ym"]], "airport": r[ix["airport"]].upper(),
                   "aircraft": r[ix["aircraft"]] if "aircraft" in ix else "",
                   "phase": r[ix["phase"]] if "phase" in ix else "",
                   "narrative": r[ix["narrative"]]}
        return

    # ASRS ships one or two header rows. Detect which.
    two = len(rows) > 2 and sum(1 for c in rows[1] if c.strip()) > sum(
        1 for c in rows[0] if c.strip()) * 0.6
    if two:
        header, body = [f"{a} {b}".strip() for a, b in zip(rows[0], rows[1])], rows[2:]
    else:
        header, body = rows[0], rows[1:]
    low = [h.lower() for h in header]

    def find(key, multi=False):
        hits = [i for i, h in enumerate(low) if any(x in h for x in NAME_HINTS[key])]
        return hits if multi else (hits[0] if hits else None)

    narr_cols = find("narrative", multi=True)
    if not narr_cols:  # last resort: whichever column holds the longest prose
        sample = body[:200]
        avg = [sum(len(r[i]) for r in sample if i < len(r)) / max(1, len(sample))
               for i in range(len(header))]
        narr_cols = [max(range(len(avg)), key=lambda i: avg[i])]
        print(f"    (no named narrative column; using '{header[narr_cols[0]]}')")
    ym_c, apt_c, ac_c, ph_c = find("ym"), find("airport"), find("aircraft"), find("phase")
    print(f"    narrative column(s): {[header[i] for i in narr_cols]}")
    print(f"    date={header[ym_c] if ym_c is not None else '-'}  "
          f"place={header[apt_c] if apt_c is not None else '-'}")

    for i, r in enumerate(body):
        def g(c):
            return r[c] if c is not None and c < len(r) else ""
        text = " ".join(dict.fromkeys(g(c).strip() for c in narr_cols if g(c).strip()))
        if len(text) < 200:
            continue
        yield {"report_id": f"ASRS{i:06d}", "ym": norm_ym(g(ym_c)),
               "airport": norm_airport(g(apt_c), text),
               "aircraft": g(ac_c)[:40], "phase": g(ph_c)[:40], "narrative": text}


def cluster(narratives, k):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=3, max_df=0.4,
                          stop_words="english", sublinear_tf=True)
    X = vec.fit_transform(narratives)
    km = MiniBatchKMeans(n_clusters=k, random_state=17, n_init=10, batch_size=2048)
    labels = km.fit_predict(X)
    terms = vec.get_feature_names_out()
    kw = {c: [terms[i] for i in km.cluster_centers_[c].argsort()[::-1][:10]] for c in range(k)}
    return labels, kw


def label_of(words):
    picked = [w for w in words if w not in STOP and not w.startswith("zzz")][:3]
    return " / ".join(picked) if picked else "unlabeled"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--asrs", action="store_true", help="input is a raw NASA ASRS export")
    ap.add_argument("--clusters", type=int, default=14)
    ap.add_argument("--min-airport", type=int, default=25,
                    help="drop fields with fewer than this many reports")
    args = ap.parse_args()

    raw = list(load(args.csv, args.asrs))
    rows = [r for r in raw if r["ym"] and r["airport"] and len(r["narrative"]) > 200]
    print(f"read {len(raw)} rows, {len(rows)} usable")
    if not rows:
        sys.exit("Nothing usable. Check the export includes narratives, a date and a place.")

    counts = Counter(r["airport"] for r in rows)
    keep = {a for a, n in counts.items() if n >= args.min_airport}
    if not keep:
        keep = {a for a, _ in counts.most_common(8)}
        print(f"    (no field reached --min-airport; keeping the top {len(keep)})")
    rows = [r for r in rows if r["airport"] in keep]
    print(f"kept {len(rows)} reports across {len(keep)} fields")

    span = (min(r["ym"] for r in rows), max(r["ym"] for r in rows))
    k = max(2, min(args.clusters, len(rows) // 50))
    print(f"clustering into {k} failure modes...")
    labels, kw = cluster([r["narrative"] for r in rows], k)
    csize = Counter(labels)
    for r, l in zip(rows, labels):
        r["cluster_id"] = f"C{int(l):02d}"

    con = db.connect(write=True)
    db.init(con)
    con.executescript("DELETE FROM reports; DELETE FROM clusters; DELETE FROM airports; "
                      "DELETE FROM reports_fts; DELETE FROM meta;")
    con.executemany(
        "INSERT OR IGNORE INTO reports (report_id, ym, airport, aircraft, phase, narrative, "
        "cluster_id) VALUES (:report_id,:ym,:airport,:aircraft,:phase,:narrative,:cluster_id)", rows)
    con.executemany("INSERT OR REPLACE INTO clusters VALUES (?,?,?,?)",
                    [(f"C{c:02d}", label_of(kw[c]), ", ".join(kw[c]), csize[c]) for c in range(k)])
    con.executemany("INSERT OR REPLACE INTO airports VALUES (?,?,?,?,?)",
                    [(a, *AIRPORT_FACTS.get(a, (a, "layout not on file", 0,
                                                "No airport context on file for this field.")))
                     for a in sorted(keep)])
    con.executemany("INSERT OR REPLACE INTO meta VALUES (?,?)", [
        ("source", "NASA ASRS" if args.asrs else "SYNTHETIC"),
        ("source_detail",
         "NASA Aviation Safety Reporting System, public export" if args.asrs else
         "Generated by make_seed.py to validate the pipeline. These are not real reports."),
        ("source_url", "https://asrs.arc.nasa.gov/search/database.html" if args.asrs else ""),
        ("is_real", "1" if args.asrs else "0"),
        ("ingested_at", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
        ("n_reports", str(len(rows))), ("span", f"{span[0]}..{span[1]}"),
        ("csv_file", args.csv.split("/")[-1]),
    ])
    con.execute("INSERT INTO reports_fts(rowid, narrative) SELECT id, narrative FROM reports")
    con.commit()

    print(f"\nfailure modes ({k}):")
    for c in range(k):
        print(f"  C{c:02d}  n={csize[c]:6d}  {label_of(kw[c])}")
    print(f"\nfields: {sorted(keep)}")
    print(f"span:   {span[0]} .. {span[1]}")
    print(f"source: {'NASA ASRS (REAL)' if args.asrs else 'SYNTHETIC'}")
    print(f"wrote   {db.DB_PATH}")


if __name__ == "__main__":
    main()

