"""Generate a synthetic ASRS-style corpus so the app runs before you have real data.

The narratives are FAKE. They imitate the register of NASA ASRS reports and contain
two deliberately planted escalations so the trend math and the backtest have
something true to find:

  1. KSFO / parallel-approach runway confusion  -> ramps up over the last 9 months
  2. KBOS / hot-brakes on short taxi turnarounds -> ramped up before 2024-05,
     which the backtest treats as a "known incident" month.

Replace with real data via `python ingest.py --asrs path/to/asrs_export.csv`.
"""
import csv
import random
from datetime import date

random.seed(1789)

AIRPORTS = ["KSFO", "KLAX", "KSEA", "KPDX", "KBOS", "KORD", "KDEN", "KATL"]
AIRCRAFT = ["A320", "B737-800", "E175", "CRJ-900", "A321neo", "B757-200", "B787-9"]

# (theme_key, phase, [sentence fragments to recombine])
THEMES = {
    "parallel_runway_confusion": (
        "approach",
        [
            "On simultaneous parallel approach we were cleared to the left runway but the "
            "sequencing put a similar type on the adjacent centerline at nearly the same range.",
            "Approach control issued a side-step late; both crews read back the same runway "
            "designator and neither controller caught the duplicate readback.",
            "Visual acquisition of the parallel traffic was delayed by haze. We questioned our "
            "runway assignment inside the final approach fix.",
            "The runway designators differ by one character and under high workload the "
            "difference was not salient on the readback.",
            "We initiated a go-around when it became apparent the aircraft on the parallel was "
            "drifting toward our extended centerline.",
        ],
    ),
    "hot_brakes_short_turn": (
        "taxi",
        [
            "Scheduled turn time did not allow brake cooling after a heavy landing. Brake "
            "temperature indications were still elevated at pushback.",
            "Ramp pressure to depart on time meant the cooling schedule was compressed. We "
            "observed rising brake temps during taxi out.",
            "Maintenance was consulted regarding elevated brake temperature but the turn "
            "continued. Fumes were reported in the cabin during taxi.",
            "A short taxi distance meant no opportunity for cooling roll. Temps exceeded the "
            "advisory threshold on the takeoff roll.",
            "We elected to return to gate after brake temperature indications continued to climb.",
        ],
    ),
    "taxiway_holdshort_confusion": (
        "taxi",
        [
            "Complex taxi route with multiple hold short instructions in a single transmission. "
            "We stopped short of the wrong marking.",
            "Construction had relocated the hold short line and the painted markings were "
            "inconsistent with the airport diagram we had loaded.",
            "Low visibility taxi with a controller handoff mid-route. The clearance was long and "
            "we did not read back the second hold short.",
            "Signage at the intersection was partially obscured. We requested progressive taxi.",
        ],
    ),
    "altitude_deviation_descent": (
        "descent",
        [
            "Descent clearance was issued with a crossing restriction that conflicted with the "
            "loaded arrival. The automation captured the wrong altitude.",
            "We received an amended crossing altitude during a frequency change and the "
            "restriction was not entered before top of descent.",
            "A late runway change reloaded the arrival and dropped the speed and altitude "
            "constraints without an obvious annunciation.",
            "Traffic alert during the descent required a level off. We were above the profile.",
        ],
    ),
    "wake_turbulence_encounter": (
        "approach",
        [
            "Encountered a significant roll on final behind a heavy that had been sequenced with "
            "reduced separation under the wake recategorization rules.",
            "Wake encounter at approximately 800 feet resulted in an uncommanded bank. We "
            "disconnected the autopilot and stabilized.",
            "Reported wake turbulence on short final. Preceding traffic was a wide body on a "
            "converging approach path.",
        ],
    ),
    "unstable_approach_continued": (
        "approach",
        [
            "Approach was not stabilized by the company criteria but the crew continued due to "
            "sequencing pressure and a short final vector.",
            "Slam dunk descent from ATC left us high and fast. Speed brakes were used late.",
            "We were kept high by approach and continued past the stabilization height before "
            "electing to go around.",
        ],
    ),
    "gpws_terrain_alert": (
        "approach",
        [
            "Received a terrain caution during a circling maneuver at night with limited visual "
            "references.",
            "A GPWS alert annunciated during the descent on the visual. We executed the escape "
            "maneuver.",
            "Terrain awareness alert triggered by an early descent below the step down fix.",
        ],
    ),
    "deice_holdover_expiry": (
        "predeparture",
        [
            "Holdover time expired while holding for departure. We returned for a second "
            "application after inspecting the wing.",
            "Deice queue was longer than forecast and holdover was marginal at the time of "
            "takeoff clearance.",
            "Contamination check after extended taxi showed residual on the leading edge.",
        ],
    ),
    "frequency_congestion_missed_call": (
        "climbout",
        [
            "Frequency congestion prevented an initial call for several minutes after handoff. We "
            "leveled at the last assigned altitude.",
            "Blocked transmissions on departure meant a climb clearance was missed. Controller "
            "queried our altitude.",
            "Two aircraft with similar call signs on frequency led to a clearance being taken by "
            "the wrong crew.",
        ],
    ),
    "fatigue_reduced_rest": (
        "cruise",
        [
            "Reduced rest overnight with an early report time. Both pilots reported degraded "
            "alertness during the descent phase.",
            "Multiple time zone changes across the pairing. Errors in the FMS entry were caught "
            "by the other pilot.",
            "Fatigue was a factor. The duty period ran to the regulatory limit after a delay.",
        ],
    ),
}

MONTHS = []
y, m = 2022, 1
while (y, m) <= (2025, 12):
    MONTHS.append(f"{y:04d}-{m:02d}")
    m += 1
    if m == 13:
        y, m = y + 1, 1


def base_rate(theme: str, airport: str) -> float:
    """Boring background rate, in reports per month."""
    r = {"KORD": 2.2, "KATL": 2.0, "KLAX": 1.9, "KDEN": 1.6, "KSFO": 1.5,
         "KSEA": 1.2, "KBOS": 1.1, "KPDX": 0.7}[airport]
    if theme == "deice_holdover_expiry" and airport in ("KLAX", "KSFO"):
        r *= 0.15  # they don't deice much
    if theme == "hot_brakes_short_turn" and airport == "KBOS":
        r *= 2.4   # short taxi distances, quick turns
    if theme == "parallel_runway_confusion":
        # needs closely spaced parallels, so it concentrates at a few fields
        r *= 1.7 if airport in ("KSFO", "KSEA", "KATL", "KLAX") else 0.18
    return r * 0.55


def planted_multiplier(theme: str, airport: str, ym: str) -> float:
    """The two signals a good system should find."""
    if theme == "parallel_runway_confusion" and airport == "KSFO":
        # ramp begins 2025-04, reaching ~4x by 2025-12
        if ym >= "2025-04":
            step = MONTHS.index(ym) - MONTHS.index("2025-04")
            return 1.0 + 0.46 * step
    if theme == "hot_brakes_short_turn" and airport == "KBOS":
        # ramp 2023-11 -> 2024-04, "incident" 2024-05, then a fix drops it back
        if "2023-11" <= ym <= "2024-05":
            step = MONTHS.index(ym) - MONTHS.index("2023-11")
            return 1.0 + 0.60 * step
        if ym > "2024-05":
            return 0.6
    return 1.0


def poisson(lam: float) -> int:
    """Knuth. Fine for the small lambdas here."""
    import math
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        k += 1
        p *= random.random()
        if p <= L:
            return k - 1


def narrative(theme: str, airport: str, ac: str) -> str:
    frags = THEMES[theme][1]
    body = " ".join(random.sample(frags, k=min(len(frags), random.choice([2, 2, 3]))))
    tail = random.choice([
        "No injuries. Suggest reviewing the procedure with crews.",
        "Recommend a review of the phraseology used in this situation.",
        "Filed to highlight a latent condition that will recur if unaddressed.",
        "Crew resource management was effective in catching the error.",
        "This is the second time this month our crew has seen this at this field.",
    ])
    return f"Aircraft {ac}. {body} {tail}"


def main(path: str = "data/seed_reports.csv") -> None:
    rows, n = [], 0
    for ym in MONTHS:
        for airport in AIRPORTS:
            for theme, (phase, _) in THEMES.items():
                lam = base_rate(theme, airport) * planted_multiplier(theme, airport, ym)
                for _ in range(poisson(lam)):
                    n += 1
                    ac = random.choice(AIRCRAFT)
                    rows.append({
                        "report_id": f"SYN{n:06d}",
                        "ym": ym,
                        "airport": airport,
                        "aircraft": ac,
                        "phase": phase,
                        "narrative": narrative(theme, airport, ac),
                    })
    random.shuffle(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} synthetic reports -> {path}")
    print(f"span {MONTHS[0]}..{MONTHS[-1]}  generated {date.today()}")


if __name__ == "__main__":
    main()

