"""AI-Powered Exercise Route Planner — ONE FILE. Backend and frontend together.

Runs with the start command the service already has:

    uvicorn app:app --host 0.0.0.0 --port $PORT


    uvicorn main:app --reload

Environment
    FEATHERLESS_API_KEY   required   https://featherless.ai/account/api-keys
    ORS_API_KEY           required   https://openrouteservice.org/dev/#/signup
    OPENAQ_API_KEY        optional   https://explore.openaq.org  (falls back to Open-Meteo)

Everything else is keyless: OpenStreetMap tiles, Open-Meteo weather and air quality.

House rule: the language model reads the request and writes the summary. Every number —
distance, elevation, air quality, every score — is computed here in Python from API
responses. The model never produces a metric.
"""
import concurrent.futures as cf
import json
import math
import os
import re
from typing import Any

import time
from datetime import datetime, timezone

import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))

FEATHERLESS_KEY = os.environ.get("FEATHERLESS_API_KEY", "")
ORS_KEY = os.environ.get("ORS_API_KEY", "")
OPENAQ_KEY = os.environ.get("OPENAQ_API_KEY", "")
GOOGLE_MAPS_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")

PARSE_MODEL = os.environ.get("PARSE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
WRITE_MODEL = os.environ.get("WRITE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

ai = OpenAI(base_url="https://api.featherless.ai/v1",
            api_key=FEATHERLESS_KEY or "missing-key", timeout=45.0)

ORS_URL = "https://api.openrouteservice.org/v2/directions/{profile}/geojson"
ORS_PROFILE = {"run": "foot-walking", "walk": "foot-walking", "hike": "foot-hiking",
               "cycle": "cycling-regular"}

app = FastAPI(title="Exercise Route Planner")


# ────────────────────────────────────────────────────────────── scoring model
#
# Score(r) = Σ weight_i · S_i , weights sum to 1.0 and depend on the activity.
#
# Nine criteria. The first four are the original brief; the rest come from data
# OpenRouteService and Open-Meteo already return and that materially change which
# route a person actually wants.

WEIGHTS = {
    #            dist  air  green noise safe  elev  surf  simple weather
    "run":   dict(distance=.26, air=.16, green=.10, noise=.06, safety=.12,
                  elevation=.12, surface=.08, simplicity=.06, weather=.04),
    "walk":  dict(distance=.24, air=.14, green=.16, noise=.08, safety=.14,
                  elevation=.06, surface=.06, simplicity=.04, weather=.08),
    "hike":  dict(distance=.24, air=.10, green=.20, noise=.08, safety=.10,
                  elevation=.14, surface=.06, simplicity=.02, weather=.06),
    "cycle": dict(distance=.26, air=.14, green=.06, noise=.06, safety=.20,
                  elevation=.12, surface=.10, simplicity=.02, weather=.04),
}

# Free-text preferences the model can detect, and which criteria each one boosts.
# "as safe as possible" boosts safety; "flat" boosts elevation matching at a low
# ideal; "quiet" boosts noise; and so on. Boosted weights are renormalised so the
# total still sums to 1.0 — the emphasis changes the ranking, not the scale.
EMPHASIS = {
    "safety":    {"safety": 3.4, "simplicity": 1.4},
    "green":     {"green": 3.2, "noise": 1.4},
    "quiet":     {"noise": 3.2, "safety": 1.4},
    "clean_air": {"air": 3.2, "green": 1.3},
    "flat":      {"elevation": 3.0},
    "hilly":     {"elevation": 3.0},
    "smooth":    {"surface": 3.0, "simplicity": 1.5},
    "simple":    {"simplicity": 3.2},
    "scenic":    {"green": 2.6, "noise": 1.8, "safety": 1.3},
    "exact":     {"distance": 2.4},
}
EMPHASIS_LABEL = {
    "safety": "as safe as possible", "green": "green and leafy",
    "quiet": "quiet, away from traffic", "clean_air": "cleanest air",
    "flat": "as flat as possible", "hilly": "give me hills",
    "smooth": "smooth underfoot", "simple": "few turns",
    "scenic": "scenic", "exact": "exact distance",
}
# "flat" and "hilly" also move the target climb rate, not just its weight.
CLIMB_OVERRIDE = {"flat": 2.0, "hilly": 60.0}


def apply_emphasis(weights: dict, emphasis: list[str]) -> dict:
    """Boost the criteria the user asked for, then renormalise to 1.0."""
    w = dict(weights)
    for e in emphasis:
        for k, mult in EMPHASIS.get(e, {}).items():
            w[k] = w[k] * mult
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}


# ── every slider expressed as a pull toward or away from green ───────────────
#
# The nine criteria are not independent in the real world. Parks are quiet, have
# cleaner air, and keep you off roads; but their paths are unpaved, they wind, and
# they are often on the hilly edge of town. So each criterion carries a signed
# correlation with greenery, and moving ANY slider shifts where the route is aimed.
#
# Positive means "wanting more of this pulls the route toward green".
# Negative means "wanting more of this pulls it toward streets".
GREEN_CORRELATION = {
    "green":      1.00,   # by definition
    "noise":      0.80,   # parks are quiet; arterials are not
    "air":        0.70,   # vegetation and distance from traffic both help
    "safety":     0.60,   # footpaths and park trails beat road shoulders
    "elevation":  0.25,   # green land is often the hilly, unbuilt edge of town
    "surface":   -0.50,   # wanting smooth tarmac pulls you onto streets
    "simplicity":-0.35,   # park paths wind; the grid is simpler to follow
    "distance":  -0.15,   # hitting an exact distance is easier on a street grid
    "weather":    0.00,   # measured at the start; unrelated to the route
}

GREEN_EXPLAIN = {
    "green":      "aims the loops straight at the nearest green land",
    "noise":      "pushes toward parks and away from traffic",
    "air":        "pushes toward vegetation and away from busy roads",
    "safety":     "prefers footpaths and park trails over road shoulders",
    "elevation":  "leans toward the unbuilt, hillier edge of town",
    "surface":    "pulls toward paved streets and away from gravel paths",
    "simplicity": "pulls toward the street grid, which has fewer turns",
    "distance":   "pulls toward streets, where hitting an exact length is easier",
    "weather":    "measured at your start point, so it does not steer the route",
}


def bias_from_weights(weights: dict, base: dict) -> float:
    """One number, -1 to +1: how hard the whole weight set pulls toward green.

    Every slider feeds in. Raising quiet or safety pulls toward parks; raising
    surface or flow pulls toward streets. This is what aims the candidate loops.
    """
    pull = 0.0
    for k, corr in GREEN_CORRELATION.items():
        delta = (weights.get(k, 0.0) - base.get(k, 0.0))
        pull += delta * corr
    return max(-1.0, min(1.0, pull * 4.0))


# Preferred climb, metres per kilometre. A runner wants gentle rolling; a hiker
# came for the hill; a cyclist on a road bike does not.
IDEAL_CLIMB = {"run": 12.0, "walk": 8.0, "hike": 45.0, "cycle": 10.0}

# ORS surface codes → how pleasant that is underfoot, 0..1.
SURFACE_Q = {1: .80, 2: .95, 3: .90, 4: .85, 5: .70, 6: .60, 7: .55, 8: .45,
             9: .50, 10: .40, 11: .55, 12: .35, 13: .30, 14: .25, 15: .40,
             16: .65, 17: .30, 18: .20, 20: .50}
# ORS waytype codes: 1 state road, 2 road, 3 street, 4 path, 5 track, 6 cycleway,
# 7 footway, 8 steps, 9 ferry, 10 construction.
WAYTYPE_SAFETY = {0: .55, 1: .20, 2: .40, 3: .55, 4: .90, 5: .85, 6: .88,
                  7: .92, 8: .60, 9: .30, 10: .15}


def s_distance(actual_m: float, target_m: float) -> float:
    """Original brief: exponential penalty on relative distance error."""
    if target_m <= 0:
        return 0.0
    return 100.0 * math.exp(-abs((actual_m - target_m) / target_m))


def s_air(aqi: float) -> float:
    """Original brief: linear from clean to unhealthy, floored at zero."""
    return max(0.0, 100.0 * (1.0 - (aqi / 200.0)))


def _extra_fraction(extras: dict, name: str, weight_map: dict, total_m: float) -> float | None:
    """ORS extra_info arrives as [from_idx, to_idx, value] spans along the geometry.

    We fold it into one 0..100 score weighted by how much of the route each value
    covers. Returns None when ORS did not supply that extra for this profile.
    """
    block = (extras or {}).get(name)
    if not block or not block.get("values"):
        return None
    num = den = 0.0
    for a, b, val in block["values"]:
        span = max(1, b - a)
        num += weight_map.get(int(val), 0.5) * span
        den += span
    return 100.0 * (num / den) if den else None


# How green a road type usually is to travel along. Paths and tracks run through
# parks and woods; state roads do not. ORS returns waytypes for every route, so
# this always discriminates — unlike the park lookup, which can come back empty
# when Overpass throttles the request.
WAYTYPE_GREEN = {0: .35, 1: .04, 2: .10, 3: .22, 4: .88, 5: .92, 6: .55,
                 7: .48, 8: .70, 9: .40, 10: .10}
# Unpaved usually means a park path or a trail rather than a street.
SURFACE_GREEN = {1: .18, 2: .06, 3: .08, 4: .12, 5: .30, 6: .45, 7: .55, 8: .78,
                 9: .62, 10: .85, 11: .40, 12: .88, 13: .90, 14: .92, 15: .95,
                 16: .60, 17: .80, 18: .70, 20: .50}


def s_green(extras: dict, osm_fraction: float | None = None) -> tuple[float, bool]:
    """Greenery, measured three ways, never assumed if anything real is available.

    1. ORS's own green index, when this deployment ships one.
    2. The share of the route on paths, tracks and unpaved surfaces — always
       available, so this is the one that guarantees routes differ from each other.
    3. The fraction running past OpenStreetMap parks and woods, blended in as a
       bonus when the Overpass lookup succeeded.

    The 85.0 baseline from the brief only appears if all three are unavailable,
    and it is flagged as estimated so the interface can say so.
    """
    signals, weights = [], []

    block = (extras or {}).get("green")
    if block and block.get("values"):
        num = den = 0.0
        for a, b, val in block["values"]:
            span = max(1, b - a)
            num += (int(val) / 10.0) * span
            den += span
        if den:
            signals.append(num / den)
            weights.append(0.45)

    # Path type and surface always vary between routes, so they are what keeps this
    # criterion able to tell candidates apart. ORS's own green index reads the same
    # on every street in a suburban grid, which is why blending matters.
    wt = _extra_fraction(extras, "waytype", WAYTYPE_GREEN, 0)
    if wt is not None:
        signals.append(min(1.0, (wt / 100.0) * 1.5))
        weights.append(0.35)
    sf = _extra_fraction(extras, "surface", SURFACE_GREEN, 0)
    if sf is not None:
        signals.append(min(1.0, (sf / 100.0) * 1.5))
        weights.append(0.20)

    if osm_fraction is not None:
        signals.append(min(1.0, osm_fraction * 1.4))
        weights.append(0.35)

    if not signals:
        return 85.0, False
    total = sum(weights)
    return 100.0 * sum(v * w for v, w in zip(signals, weights)) / total, True


# How loud each road class is to walk beside. Traffic volume is the dominant
# source of ambient noise, and road class is the best free proxy for it.
WAYTYPE_QUIET = {0: .55, 1: .10, 2: .28, 3: .48, 4: .92, 5: .88, 6: .82,
                 7: .78, 8: .85, 9: .60, 10: .20}


def s_noise(extras: dict) -> tuple[float, bool]:
    """Quiet score. Uses ORS's noise index when present, road class otherwise.

    ORS only ships the noise index on some deployments, and falling back to a
    constant made every route score identically — which is why the slider did
    nothing. Road class is always returned, so this always discriminates.
    """
    signals, weights = [], []
    block = (extras or {}).get("noise")
    if block and block.get("values"):
        num = den = 0.0
        for a, b, val in block["values"]:
            span = max(1, b - a)
            num += (1.0 - int(val) / 10.0) * span
            den += span
        if den:
            signals.append(num / den)
            weights.append(0.4)
    v = _extra_fraction(extras, "waytype", WAYTYPE_QUIET, 0)
    if v is not None:
        signals.append(v / 100.0)
        weights.append(0.6)
    if not signals:
        return 70.0, False
    return 100.0 * sum(a * b for a, b in zip(signals, weights)) / sum(weights), True


def s_safety(extras: dict, steps: int, km: float) -> tuple[float, bool]:
    """Separated paths beat streets beat state roads; frequent junctions cost."""
    base = _extra_fraction(extras, "waytype", WAYTYPE_SAFETY, 0)
    real = base is not None
    if base is None:
        base = 90.0                 # brief's baseline
    junction_rate = steps / max(0.4, km)          # manoeuvres per km
    penalty = min(25.0, max(0.0, (junction_rate - 4.0) * 3.0))
    return max(0.0, base - penalty), real


def s_surface(extras: dict) -> tuple[float, bool]:
    v = _extra_fraction(extras, "surface", SURFACE_Q, 0)
    return (v, True) if v is not None else (75.0, False)


def s_elevation(gain_m: float, km: float, activity: str, ideal: float | None = None) -> float:
    """Distance from the ideal climb rate, both directions.

    The ideal comes from the activity unless the user asked for flat or hilly,
    in which case CLIMB_OVERRIDE moves it.
    """
    if km <= 0:
        return 50.0
    rate = gain_m / km
    ideal = ideal or IDEAL_CLIMB.get(activity, 12.0)
    return 100.0 * math.exp(-abs(rate - ideal) / (max(2.0, ideal) * 1.6))


def s_simplicity(steps: int, km: float) -> float:
    """Fewer turns per km means you can hold a rhythm instead of navigating."""
    if km <= 0:
        return 50.0
    turns = steps / km
    return max(0.0, 100.0 * math.exp(-max(0.0, turns - 3.0) / 7.0))


def s_weather(w: dict, activity: str) -> float:
    """Apparent temperature, rain, wind and UV, from Open-Meteo."""
    if not w:
        return 70.0
    t = w.get("apparent_temperature")
    score = 100.0
    if t is not None:
        ideal = 12.0 if activity in ("run", "cycle") else 18.0
        score -= min(55.0, abs(t - ideal) * 3.2)
    score -= min(30.0, (w.get("precipitation") or 0) * 22.0)
    score -= min(18.0, max(0.0, (w.get("wind_speed_10m") or 0) - 18.0) * 1.1)
    score -= min(15.0, max(0.0, (w.get("uv_index") or 0) - 6.0) * 3.0)
    return max(0.0, score)


# ────────────────────────────────────────────────────────────── external data

def fetch_air(lat: float, lon: float) -> dict:
    """Ground sensors from OpenAQ when a key is present; Open-Meteo otherwise."""
    if OPENAQ_KEY:
        try:
            r = requests.get("https://api.openaq.org/v3/locations",
                             params={"coordinates": f"{lat},{lon}", "radius": 25000,
                                     "parameters_id": 2, "limit": 20},
                             headers={"X-API-Key": OPENAQ_KEY}, timeout=12)
            r.raise_for_status()
            best = None
            for loc in r.json().get("results", []):
                for s in loc.get("sensors", []) or []:
                    v = (s.get("latest") or {}).get("value")
                    if v is not None:
                        best = (float(v), loc.get("name", "sensor"))
                        break
                if best:
                    break
            if best:
                pm, name = best
                return {"pm25": round(pm, 1), "aqi": round(pm25_to_aqi(pm)),
                        "source": f"OpenAQ · {name}", "measured": True}
        except Exception:
            pass
    try:
        r = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality",
                         params={"latitude": lat, "longitude": lon,
                                 "current": "pm2_5,us_aqi", "timezone": "auto"},
                         timeout=12)
        r.raise_for_status()
        c = r.json().get("current", {})
        pm = c.get("pm2_5")
        aqi = c.get("us_aqi") or (pm25_to_aqi(pm) if pm is not None else 50)
        return {"pm25": pm, "aqi": round(aqi), "source": "Open-Meteo air quality",
                "measured": True}
    except Exception as e:
        return {"pm25": None, "aqi": 50, "source": f"unavailable ({e})", "measured": False}


def pm25_to_aqi(pm: float) -> float:
    """US EPA piecewise conversion."""
    bands = [(0, 12, 0, 50), (12, 35.4, 51, 100), (35.4, 55.4, 101, 150),
             (55.4, 150.4, 151, 200), (150.4, 250.4, 201, 300), (250.4, 500, 301, 500)]
    for lo, hi, alo, ahi in bands:
        if pm <= hi:
            return alo + (ahi - alo) * (pm - lo) / (hi - lo)
    return 500.0


# ── real greenery, measured from OpenStreetMap ───────────────────────────────
# ORS only returns its green/noise indices on some deployments. Relying on them
# meant every route scored the same constant, so the greenery slider did nothing.
# Instead we pull the actual parks, woods and water near the start from Overpass
# once per plan, index them in a coarse grid, and measure what fraction of each
# route runs within GREEN_RADIUS of one.

OVERPASS_HOSTS = ["https://overpass.kumi.systems/api/interpreter",
                  "https://overpass-api.de/api/interpreter",
                  "https://overpass.private.coffee/api/interpreter"]
GREEN_RADIUS = 90.0          # metres; a park across the street still counts
_green_cache: dict = {}


def fetch_green(lat: float, lon: float, radius_m: float) -> list:
    """Centres of parks, woods, grass, water and tree rows near the start."""
    key = (round(lat, 3), round(lon, 3), round(radius_m / 500))
    if key in _green_cache:
        return _green_cache[key]
    r = int(min(8000, max(1200, radius_m)))
    q = f"""[out:json][timeout:13];
(
  way["leisure"~"park|garden|nature_reserve|recreation_ground|golf_course|common|pitch"](around:{r},{lat},{lon});
  way["landuse"~"forest|grass|meadow|village_green|recreation|allotments|orchard|vineyard|farmland|cemetery|greenfield"](around:{r},{lat},{lon});
  way["natural"~"wood|water|scrub|grassland|heath|wetland|tree_row"](around:{r},{lat},{lon});
  way["amenity"="grave_yard"](around:{r},{lat},{lon});
  relation["leisure"~"park|nature_reserve"](around:{r},{lat},{lon});
  relation["landuse"~"forest|grass|meadow"](around:{r},{lat},{lon});
  relation["natural"~"wood|water"](around:{r},{lat},{lon});
);
out center 400;"""
    pts = []
    for host in OVERPASS_HOSTS:
        try:
            resp = requests.get(host, params={"data": q}, timeout=14,
                                headers={"User-Agent": "route-planner/1.0 (hackathon)"})
            if resp.status_code != 200:
                continue
            for el in resp.json().get("elements", []):
                c = el.get("center") or el
                if c.get("lat") and c.get("lon"):
                    pts.append((c["lat"], c["lon"]))
            if pts:
                break
        except Exception:
            continue
    _green_cache[key] = pts
    return pts


def green_hotspots(lat, lon, pts, k=3):
    """Where the green actually is: the densest clusters, as bearing and distance.

    Used to aim candidate loops at parks when the runner wants greenery, and away
    from them when they do not.
    """
    if not pts:
        return []
    cells = {}
    for pla, plo in pts:
        key = (round(pla, 2), round(plo, 2))
        cells.setdefault(key, []).append((pla, plo))
    ranked = sorted(cells.items(), key=lambda kv: -len(kv[1]))[:k]
    out = []
    for _, group in ranked:
        cla = sum(x[0] for x in group) / len(group)
        clo = sum(x[1] for x in group) / len(group)
        dy = (cla - lat) * 111320.0
        dx = (clo - lon) * 111320.0 * math.cos(math.radians(lat))
        dist = math.hypot(dx, dy)
        bearing = (math.degrees(math.atan2(dx, dy)) + 360) % 360
        out.append({"lat": cla, "lon": clo, "bearing": bearing,
                    "distance_m": round(dist), "features": len(group)})
    return out


def _grid(pts, cell_deg=0.004):
    g = {}
    for la, lo in pts:
        g.setdefault((int(la / cell_deg), int(lo / cell_deg)), []).append((la, lo))
    return g, cell_deg


def green_fraction(coords: list, green_pts: list) -> float | None:
    """Fraction of the route running close to a green feature. 0..1, or None."""
    if not green_pts or not coords:
        return None
    grid, cell = _grid(green_pts)
    sample = coords[::max(1, len(coords) // 160)]
    near = 0
    for c in sample:
        lo, la = c[0], c[1]
        gi, gj = int(la / cell), int(lo / cell)
        hit = False
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for (pla, plo) in grid.get((gi + di, gj + dj), ()):
                    dy = (pla - la) * 111320.0
                    dx = (plo - lo) * 111320.0 * math.cos(math.radians(la))
                    if dy * dy + dx * dx <= GREEN_RADIUS * GREEN_RADIUS * 4:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                break
        near += 1 if hit else 0
    return near / len(sample)


def fetch_weather(lat: float, lon: float) -> dict:
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast",
                         params={"latitude": lat, "longitude": lon,
                                 "current": "temperature_2m,apparent_temperature,"
                                            "precipitation,wind_speed_10m,uv_index",
                                 "daily": "sunset", "timezone": "auto"}, timeout=12)
        r.raise_for_status()
        j = r.json()
        cur = j.get("current", {})
        cur["sunset"] = (j.get("daily", {}).get("sunset") or [None])[0]
        return cur
    except Exception:
        return {}


def offset(lat: float, lon: float, bearing_deg: float, metres: float):
    """Move a point along a compass bearing. Plain great-circle maths."""
    R = 6371000.0
    br = math.radians(bearing_deg)
    d = metres / R
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(br))
    l2 = l1 + math.atan2(math.sin(br) * math.sin(d) * math.cos(p1),
                         math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), math.degrees(l2)


def fetch_routes(lat: float, lon: float, target_m: float, activity: str,
                 emphasis: list[str] | None = None, n: int = 14,
                 hotspots: list | None = None, green_bias: float = 0.0) -> list[dict]:
    """Candidate loops that actually come back the length you asked for.

    Placing our own via-points let us aim a loop in a chosen direction, but it gave
    no control over length: a point two kilometres away with no footpath to it makes
    the router detour for tens of kilometres, which is how a five mile request came
    back as fifty.

    So we use OpenRouteService's own round_trip instead, which takes the target
    length directly and honours it. Variety comes from seeding it many different
    ways rather than from steering it, and the greenery ladder is then built by
    measuring the results and sorting them.
    """
    if not ORS_KEY:
        raise RuntimeError("ORS_API_KEY is not set")
    emphasis = emphasis or []
    profile = ORS_PROFILE.get(activity, "foot-walking")

    avoid = []
    if "safety" in emphasis or "smooth" in emphasis:
        avoid = ["steps", "ferries"]
    elif "flat" in emphasis:
        avoid = ["steps"]

    # (seed, waypoint count) pairs. Different seeds send the loop off in different
    # directions; different point counts change how much it wanders.
    specs = [(1, 3), (7, 4), (13, 5), (23, 3), (31, 4), (41, 5), (53, 6), (67, 3),
             (79, 4), (97, 5), (103, 4), (127, 5), (149, 3), (167, 6)][:n]

    def one(spec):
        seed, points = spec
        body = {
            "coordinates": [[lon, lat]],
            "elevation": True,
            "instructions": True,
            "extra_info": ["surface", "waytype", "steepness", "green", "noise"],
            "options": {"round_trip": {"length": int(target_m), "points": points,
                                       "seed": seed}},
        }
        if avoid:
            body["options"]["avoid_features"] = avoid
        try:
            resp = requests.post(ORS_URL.format(profile=profile), json=body,
                                 headers={"Authorization": ORS_KEY,
                                          "Content-Type": "application/json"}, timeout=14)
            if resp.status_code != 200:
                return {"error": f"ORS {resp.status_code}: {resp.text[:160]}"}
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def aimed(spec):
        """A loop deliberately pointed at green land.

        round_trip cannot be steered, so the greenest option it offers is whatever
        its seeds happen to find. When there is green nearby we also send explicit
        triangles at it, sized from the target length so they still come back the
        right distance: a triangle with two via-points 72 degrees apart has a
        perimeter of roughly four times the radius once streets are accounted for.
        """
        bearing, scale = spec
        rr = (target_m / 4.0) * scale
        p1 = offset(lat, lon, bearing, rr)
        p2 = offset(lat, lon, bearing + 72, rr)
        body = {
            "coordinates": [[lon, lat], [p1[1], p1[0]], [p2[1], p2[0]], [lon, lat]],
            "elevation": True, "instructions": True,
            "extra_info": ["surface", "waytype", "steepness", "green", "noise"],
        }
        if avoid:
            body["options"] = {"avoid_features": avoid}
        try:
            resp = requests.post(ORS_URL.format(profile=profile), json=body,
                                 headers={"Authorization": ORS_KEY,
                                          "Content-Type": "application/json"}, timeout=14)
            if resp.status_code != 200:
                return {"error": f"ORS {resp.status_code}"}
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    aims = []
    if hotspots:
        for h in hotspots[:3]:
            b = h["bearing"]
            aims += [(b, 1.0), ((b + 30) % 360, 0.95), ((b - 30) % 360, 1.05)]
        aims = aims[:5]

    with cf.ThreadPoolExecutor(max_workers=18) as pool:
        rt = pool.map(one, specs)
        am = pool.map(aimed, aims) if aims else []
        return list(rt) + list(am)


# ────────────────────────────────────────────────────────────── model calls

_parse_cache: dict = {}


def parse_request(prompt: str, skip_model: bool = False) -> dict:
    """Free text → target distance in metres and activity. Model picks, Python validates.

    Cached, and skippable: a re-plan sends the same prompt with new weights, and
    re-asking the model the same question just adds latency.
    """
    key = prompt.strip().lower()
    if key in _parse_cache:
        return _parse_cache[key]
    fallback = _regex_parse(prompt)
    if skip_model or not FEATHERLESS_KEY:
        return fallback
    try:
        r = ai.chat.completions.create(
            model=PARSE_MODEL, max_tokens=180, temperature=0.0,
            messages=[
                {"role": "system", "content":
                 "Extract the exercise request. Reply with JSON only, no prose, no code "
                 'fences: {"distance_value": <number>, "unit": "mi|km|m", '
                 '"activity": "run|walk|hike|cycle", '
                 '"emphasis": [<zero or more of: safety, green, quiet, clean_air, flat, '
                 'hilly, smooth, simple, scenic, exact>], '
                 '"notes": "<what they asked for in their own words, or null>"}. '
                 "Map their wording onto the emphasis list: 'as safe as possible' -> "
                 "safety; 'avoid traffic', 'quiet' -> quiet; 'parks', 'trees', 'nature' "
                 "-> green; 'flat', 'no hills' -> flat; 'hilly', 'hill training' -> "
                 "hilly; 'clean air', 'asthma' -> clean_air; 'even ground', 'paved' -> "
                 "smooth; 'no turns', 'simple' -> simple; 'pretty', 'scenic' -> scenic; "
                 "'exactly' -> exact. Return an empty list if they stated no preference. "
                 'If no distance is stated use 5 and unit "km". Never invent a location.'},
                {"role": "user", "content": prompt[:600]}])
        txt = (r.choices[0].message.content or "").strip().strip("`")
        txt = re.sub(r"^json", "", txt, flags=re.I).strip()
        a, b = txt.find("{"), txt.rfind("}")
        import json
        got = json.loads(txt[a:b + 1])
        val = float(got.get("distance_value") or 5)
        unit = str(got.get("unit", "km")).lower()
        metres = val * {"mi": 1609.34, "km": 1000.0, "m": 1.0}.get(unit, 1000.0)
        act = str(got.get("activity", "run")).lower()
        if act not in WEIGHTS:
            act = "run"
        emph = [e for e in (got.get("emphasis") or []) if e in EMPHASIS][:4]
        if not emph:
            emph = fallback["emphasis"]        # keyword safety net
        out = {"target_m": max(400.0, min(50000.0, metres)), "activity": act,
               "notes": got.get("notes"), "parsed_by": PARSE_MODEL,
               "emphasis": emph, "stated": f"{val} {unit}"}
        _parse_cache[key] = out
        return out
    except Exception:
        return fallback


def _regex_parse(prompt: str) -> dict:
    p = (prompt or "").lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(miles?|mi|kilometers?|kilometres?|km|m\b)", p)
    val, unit = (float(m.group(1)), m.group(2)) if m else (5.0, "km")
    metres = val * (1609.34 if unit.startswith(("mi", "mile")) else
                    1.0 if unit.strip() == "m" else 1000.0)
    act = ("hike" if "hik" in p else "cycle" if any(x in p for x in ("cycl", "bike", "ride"))
           else "walk" if "walk" in p else "run")
    # Keyword safety net, so a preference is never silently dropped even if the
    # model is unavailable or returns nothing useful.
    kw = {
        "safety": ("safe", "safest", "safety", "dangerous", "traffic-free", "sidewalk"),
        "quiet": ("quiet", "peaceful", "away from traffic", "no cars", "calm"),
        "green": ("green", "park", "trees", "nature", "leafy", "trail", "woods"),
        "clean_air": ("clean air", "air quality", "pollution", "asthma", "smog"),
        "flat": ("flat", "no hills", "nothing steep", "level"),
        "hilly": ("hilly", "hills", "climb", "elevation", "steep"),
        "smooth": ("smooth", "paved", "even ground", "no gravel", "pavement"),
        "simple": ("simple", "few turns", "no turns", "straightforward", "easy to follow"),
        "scenic": ("scenic", "pretty", "beautiful", "views", "nice"),
        "exact": ("exactly", "precisely", "exact"),
    }
    emph = [k for k, words in kw.items() if any(x in p for x in words)][:4]
    return {"target_m": max(400.0, min(50000.0, metres)), "activity": act,
            "notes": None, "parsed_by": "keyword fallback", "emphasis": emph,
            "stated": f"{val} {unit}"}


def write_summary(best: dict, ctx: dict) -> str:
    facts = (
        f"Activity: {ctx['activity']}. Asked for {ctx['stated']} "
        f"({round(ctx['target_m'])} m).\n"
        f"Chosen loop: {best['distance_mi']} miles ({round(best['distance_m'])} m), "
        f"score {best['score']} out of 100.\n"
        f"Elevation gain {best['elevation_gain_m']} m over the loop.\n"
        f"Air quality index {ctx['air']['aqi']} from {ctx['air']['source']}.\n"
        f"Sub-scores — distance {best['scores']['distance']}, air {best['scores']['air']}, "
        f"greenery {best['scores']['green']}, quiet {best['scores']['noise']}, "
        f"safety {best['scores']['safety']}, elevation {best['scores']['elevation']}, "
        f"surface {best['scores']['surface']}, simplicity {best['scores']['simplicity']}, "
        f"weather {best['scores']['weather']}.\n"
        f"Estimated time {best['estimated_minutes']} minutes. "
        f"{best['turns']} turns. Conditions: "
        f"{ctx['weather'].get('apparent_temperature')}C feels-like, "
        f"{ctx['weather'].get('precipitation')} mm rain.")
    if not FEATHERLESS_KEY:
        return (f"A {best['distance_mi']} mile loop scoring {best['score']}/100. "
                f"Set FEATHERLESS_API_KEY for a written summary.")
    try:
        r = ai.chat.completions.create(
            model=WRITE_MODEL, max_tokens=220, temperature=0.4,
            messages=[
                {"role": "system", "content":
                 "You brief someone about to head out for exercise. Two or three short "
                 "sentences, warm and practical. Use ONLY the numbers given — never invent "
                 "or adjust a figure, a street name or a place name. Say what is good about "
                 "this route and name the one thing that is worst about it."},
                {"role": "user", "content": facts}])
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return (f"A {best['distance_mi']} mile loop scoring {best['score']}/100. "
                f"(Summary unavailable: {e})")


# ────────────────────────────────────────────────────────────── scoring a route

def score_route(feature_collection: dict, target_m: float, activity: str,
                air: dict, weather: dict, idx: int, weights: dict | None = None,
                climb_ideal: float | None = None, green_pts: list | None = None) -> dict | None:
    feats = feature_collection.get("features") or []
    if not feats:
        return None
    f = feats[0]
    props = f.get("properties", {})
    summary = props.get("summary", {})
    dist_m = float(summary.get("distance") or 0)
    if dist_m <= 0:
        return None
    km = dist_m / 1000.0
    ascent = float(props.get("ascent") or 0)
    steps = sum(len(seg.get("steps", [])) for seg in props.get("segments", []))
    extras = props.get("extras", {})

    coords = f.get("geometry", {}).get("coordinates", [])
    eles = [c[2] for c in coords if len(c) > 2]
    loopq = loop_quality(coords)
    green, green_real = s_green(extras, green_fraction(coords, green_pts or []))
    noise, noise_real = s_noise(extras)
    safety, safety_real = s_safety(extras, steps, km)
    surface, surface_real = s_surface(extras)

    sc = {
        "distance": s_distance(dist_m, target_m),
        "air": s_air(air["aqi"]),
        "green": green,
        "noise": noise,
        "safety": safety,
        "elevation": s_elevation(ascent, km, activity, climb_ideal),
        "surface": surface,
        "simplicity": s_simplicity(steps, km),
        "weather": s_weather(weather, activity),
    }
    w = weights or WEIGHTS.get(activity, WEIGHTS["run"])
    total = sum(w[k] * sc[k] for k in w)

    pace = {"run": 6.2, "walk": 12.5, "hike": 15.0, "cycle": 3.2}.get(activity, 6.2)
    return {
        "id": idx,
        "score": round(total, 1),
        "distance_m": round(dist_m),
        "distance_mi": round(dist_m / 1609.34, 2),
        "distance_km": round(km, 2),
        "elevation_gain_m": round(ascent),
        "climb_per_km": round(ascent / km, 1) if km else 0,
        "turns": steps,
        "estimated_minutes": round(km * pace),
        "aqi": air["aqi"],
        "scores": {k: round(v, 1) for k, v in sc.items()},
        "weights": w,
        "estimated": {"green": not green_real, "noise": not noise_real,
                      "safety": not safety_real, "surface": not surface_real},
        "loop": loopq,
        "extras": {k: v.get("values", []) for k, v in (extras or {}).items()},
        "elevation": {"points": [round(e, 1) for e in eles[::max(1, len(eles)//120)]],
                      "min": round(min(eles), 1) if eles else None,
                      "max": round(max(eles), 1) if eles else None,
                      "descent": round(float(props.get("descent") or 0))},
        "geojson": f,
    }


def loop_quality(coords: list) -> dict:
    """How much this route is a real loop rather than an out-and-back with a spur.

    Two measures. Overlap is the share of the route that doubles back over ground
    it already covered — high on a spur, near zero on a clean loop. Roundness is
    the enclosed area against the perimeter squared, which peaks for a circle and
    collapses toward zero for a there-and-back line.
    """
    if len(coords) < 8:
        return {"overlap": 1.0, "roundness": 0.0, "is_loop": False}

    pts = sample_points(coords, 120)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(pts[0][1]))
    xy = [((c[0] - pts[0][0]) * m_per_deg_lon, (c[1] - pts[0][1]) * m_per_deg_lat)
          for c in pts]

    near = 0
    for i, (x1, y1) in enumerate(xy):
        for j in range(i + 4, len(xy)):
            x2, y2 = xy[j]
            if (x1 - x2) ** 2 + (y1 - y2) ** 2 < 900:      # within 30 m
                near += 1
                break
    overlap = near / len(xy)

    area = abs(sum(xy[i][0] * xy[i - 1][1] - xy[i - 1][0] * xy[i][1]
                   for i in range(len(xy)))) / 2.0
    per = sum(math.dist(xy[i], xy[i - 1]) for i in range(1, len(xy)))
    roundness = (4 * math.pi * area / (per * per)) if per > 0 else 0.0

    return {"overlap": round(overlap, 3), "roundness": round(roundness, 3),
            "is_loop": overlap < 0.34 and roundness > 0.10}


def sample_points(coords: list, k: int) -> list:
    """Evenly spaced points along the loop, always including the start and end."""
    if len(coords) <= k:
        return coords
    step = (len(coords) - 1) / (k - 1)
    return [coords[min(len(coords) - 1, round(i * step))] for i in range(k)]


def graphhopper_url(coords: list, activity: str) -> str:
    """Open the exact loop on GraphHopper Maps.

    Google's directions URL caps at nine waypoints and re-routes between them, so a
    loop comes back approximate. GraphHopper accepts many more points, which
    reproduces the route as planned. Good for checking it on a laptop before you go.
    """
    if not coords:
        return ""
    prof = {"cycle": "bike", "hike": "hike"}.get(activity, "foot")
    pts = sample_points(coords, 14)
    q = "&".join(f"point={c[1]:.5f}%2C{c[0]:.5f}" for c in pts)
    return f"https://graphhopper.com/maps/?{q}&profile={prof}&layer=OpenStreetMap"


def google_maps_url(coords: list, activity: str) -> str:
    """Open the loop in Google Maps with as many pins as Google will take.

    The documented ?api=1 form caps at nine waypoints, which turns a loop into a
    rough approximation. The path form — /maps/dir/lat,lng/lat,lng/... — accepts
    far more stops, so the line Google draws follows the planned route closely.
    The trailing data parameter sets the travel mode: 3e2 walking, 3e1 cycling.
    """
    if not coords or len(coords) < 2:
        return ""
    picked = sample_points(coords, 22)
    # close the loop explicitly so Google returns to the start
    if picked[-1] != coords[0]:
        picked.append(coords[0])
    legs = "/".join(f"{c[1]:.5f},{c[0]:.5f}" for c in picked)
    mode = "!3e1" if activity == "cycle" else "!3e2"
    mid = picked[len(picked) // 2]
    return (f"https://www.google.com/maps/dir/{legs}"
            f"/@{mid[1]:.5f},{mid[0]:.5f},15z/data=!3m1!4b1!4m2!4m1{mode}")


# ──────────────────────────────────────────────── shared, saved routes
#
# Everyone who opens the site sees the same saved list, which is what makes it a
# shared board rather than a private one. Kept in memory with a file backup so a
# restart does not wipe it; a real deployment would use a database.

SAVED_PATH = os.environ.get("SAVED_PATH", "/tmp/greenroute_saved.json")
_saved: list = []
_saved_lock = __import__("threading").Lock()


def _load_saved():
    global _saved
    try:
        with open(SAVED_PATH, "r", encoding="utf-8") as f:
            _saved = json.load(f)[:300]
    except Exception:
        _saved = []


def _persist():
    try:
        with open(SAVED_PATH, "w", encoding="utf-8") as f:
            json.dump(_saved[:300], f)
    except Exception:
        pass


_load_saved()


# ────────────────────────────────────────────────────────────── API

class PlanRequest(BaseModel):
    lat: float
    lon: float
    prompt: str = "I want to run 5 km."
    # -1 aims the loops away from green, +1 aims them straight at it.
    green_bias: float | None = None
    # the client can send its whole weight set instead, and the server derives
    # the bias from it using GREEN_CORRELATION
    weights: dict | None = None


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/api/health")
def health() -> dict[str, Any]:
    out = {"featherless_key": bool(FEATHERLESS_KEY), "ors_key": bool(ORS_KEY),
           "openaq_key": bool(OPENAQ_KEY), "google_maps_key": bool(GOOGLE_MAPS_KEY),
           "map_engine": "google" if GOOGLE_MAPS_KEY else "leaflet", "models": {}}
    if FEATHERLESS_KEY:
        try:
            ids = {m.id for m in ai.models.list().data}
            out["models"] = {m: (m in ids) for m in {PARSE_MODEL, WRITE_MODEL}}
            out["featherless_catalog_size"] = len(ids)
        except Exception as e:
            out["models"] = {"error": str(e)[:200]}
    return out


@app.get("/api/config")
def config():
    """What the frontend needs to decide which map engine to load."""
    return {"google_maps_key": GOOGLE_MAPS_KEY,
            "map_engine": "google" if GOOGLE_MAPS_KEY else "leaflet"}


@app.get("/api/criteria")
def criteria():
    """What the scorer measures and where each number comes from. Shown in the UI."""
    return {
        "weights": WEIGHTS,
        "green_correlation": GREEN_CORRELATION,
        "green_explain": GREEN_EXPLAIN,
        "criteria": [
            {"key": "distance", "label": "Distance match",
             "source": "OpenRouteService loop length vs your target"},
            {"key": "air", "label": "Air quality",
             "source": "OpenAQ ground sensors, or Open-Meteo air quality"},
            {"key": "green", "label": "Greenery",
             "source": "OpenRouteService green index along the route"},
            {"key": "noise", "label": "Quiet",
             "source": "OpenRouteService noise index along the route"},
            {"key": "safety", "label": "Path safety",
             "source": "Way types (footway/path vs road) and junction density"},
            {"key": "elevation", "label": "Climb",
             "source": "SRTM elevation from OpenRouteService, scored against the "
                       "ideal climb rate for this activity"},
            {"key": "surface", "label": "Surface",
             "source": "OpenStreetMap surface tags along the route"},
            {"key": "simplicity", "label": "Flow",
             "source": "Turns per kilometre from the routing instructions"},
            {"key": "weather", "label": "Conditions",
             "source": "Open-Meteo feels-like temperature, rain, wind and UV"},
        ],
    }


@app.get("/api/diagnose")
def diagnose(lat: float = 37.6624, lon: float = -121.8747, prompt: str = "5 km run"):
    """Which data sources answered, and how far apart the candidates are on each
    criterion. If a slider does nothing, its spread here will be near zero."""
    r = _plan(PlanRequest(lat=lat, lon=lon, prompt=prompt))
    if "error" in r:
        return r
    return {
        "candidates": len(r["routes"]),
        "spread_per_criterion": r.get("spread"),
        "normalised_spread": {k: round(max(x["rel"][k] for x in r["routes"])
                                       - min(x["rel"][k] for x in r["routes"]), 1)
                              for k in r["routes"][0].get("rel", {})},
        "dead_sliders": [k for k, v in (r.get("spread") or {}).items()
                         if v < 0.5 and k not in ("air", "weather")],
        "green_features_found": r.get("green_features"),
        "air_source": r["air"]["source"],
        "estimated_flags": r["routes"][0]["estimated"],
        "extras_returned_by_ors": sorted((r["routes"][0].get("extras") or {}).keys()),
        "warnings": r.get("warnings"),
    }


@app.get("/api/saved")
def list_saved(room: str = ""):
    """Routes saved to one room. A room is just a short shared code."""
    code = (room or "").strip().upper()[:8]
    rs = [r for r in _saved if r.get("room", "") == code]
    return {"room": code,
            "routes": [{k: v for k, v in r.items() if k != "geojson"} for r in rs]}


@app.get("/api/saved/{route_id}")
def get_saved(route_id: str):
    for r in _saved:
        if r["id"] == route_id:
            return r
    return {"error": "No route with that id."}


@app.post("/api/saved")
def save_route(payload: dict):
    p = payload or {}
    if not p.get("geojson"):
        return {"error": "Nothing to save."}
    entry = {
        "id": f"r{int(time.time() * 1000) % 10_000_000}{len(_saved)}",
        "room": (p.get("room") or "").strip().upper()[:8],
        "name": (p.get("name") or "Untitled loop")[:60],
        "by": (p.get("by") or "anonymous")[:24],
        "distance_mi": p.get("distance_mi"),
        "minutes": p.get("minutes"),
        "elevation_gain_m": p.get("elevation_gain_m"),
        "green": p.get("green"),
        "start": p.get("start"),
        "google_maps_url": p.get("google_maps_url"),
        "graphhopper_url": p.get("graphhopper_url"),
        "note": (p.get("note") or "")[:200],
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "geojson": p["geojson"],
    }
    with _saved_lock:
        _saved.insert(0, entry)
        del _saved[300:]
        _persist()
    return {"ok": True, "id": entry["id"], "count": len(_saved)}


@app.patch("/api/saved/{route_id}")
def edit_saved(route_id: str, payload: dict):
    """Rename a saved route, change who it is by, or add a note."""
    p = payload or {}
    with _saved_lock:
        for r in _saved:
            if r["id"] == route_id:
                if p.get("name") is not None:
                    r["name"] = str(p["name"])[:60] or r["name"]
                if p.get("by") is not None:
                    r["by"] = str(p["by"])[:24] or r["by"]
                if p.get("note") is not None:
                    r["note"] = str(p["note"])[:200]
                _persist()
                return {"ok": True, "route": {k: v for k, v in r.items()
                                              if k != "geojson"}}
    return {"error": "No route with that id."}


@app.delete("/api/saved/{route_id}")
def delete_saved(route_id: str):
    with _saved_lock:
        before = len(_saved)
        _saved[:] = [r for r in _saved if r["id"] != route_id]
        _persist()
    return {"ok": True, "removed": before - len(_saved)}


@app.post("/api/plan-route")
def plan_route(req: PlanRequest):
    """Always returns JSON. A crash here used to surface as an HTML error page,
    which the browser then failed to parse — an unhelpful error for a real one."""
    try:
        return _plan(req)
    except Exception as e:
        return {"error": f"Planning failed: {type(e).__name__}: {e}",
                "detail": ["The server hit an unexpected error. Try a shorter distance "
                           "or a different start point."]}


def _plan(req: PlanRequest):
    if not (-90 <= req.lat <= 90 and -180 <= req.lon <= 180):
        return {"error": "Those coordinates are not on Earth. Check latitude and longitude."}

    parsed = parse_request(req.prompt,
                           skip_model=req.green_bias is not None or req.weights is not None)
    emphasis = parsed.get("emphasis") or []
    base_w = WEIGHTS.get(parsed["activity"], WEIGHTS["run"])
    weights = apply_emphasis(base_w, emphasis)
    climb_ideal = next((CLIMB_OVERRIDE[e] for e in emphasis if e in CLIMB_OVERRIDE), None)

    # Only the green-bias path needs the park lookup, and that lookup is the slowest
    # thing here. Skip it unless it will actually be used, so an ordinary plan never
    # waits on Overpass. When it is needed, a cached result usually makes it instant.
    need_green = True      # the ladder needs to know where the green is
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        f_air = pool.submit(fetch_air, req.lat, req.lon)
        f_wx = pool.submit(fetch_weather, req.lat, req.lon)
        f_gr = pool.submit(fetch_green, req.lat, req.lon,
                           parsed["target_m"] * 0.8) if need_green else None
        air = f_air.result()
        weather = f_wx.result()
        green_pts = []
        if f_gr is not None:
            try:
                green_pts = f_gr.result(timeout=16)
            except Exception:
                green_pts = []

    hotspots = green_hotspots(req.lat, req.lon, green_pts)

    # Where the greenery bias comes from: an explicit slider value if the client
    # sent one, otherwise the preference the model read from the request text.
    if req.weights:
        bias = bias_from_weights(req.weights, base_w)
    elif req.green_bias is not None:
        bias = max(-1.0, min(1.0, req.green_bias))
    elif "green" in emphasis or "scenic" in emphasis:
        bias = 1.0
    else:
        bias = 0.0

    try:
        raw = fetch_routes(req.lat, req.lon, parsed["target_m"], parsed["activity"],
                           emphasis, 14, hotspots, bias)
    except RuntimeError as e:
        return {"error": str(e)}

    routes, errors = [], []
    for i, fc in enumerate(raw):
        if "error" in fc:
            errors.append(fc["error"])
            continue
        s = score_route(fc, parsed["target_m"], parsed["activity"], air, weather, i + 1,
                        weights, climb_ideal, green_pts)
        if s:
            cs = s["geojson"].get("geometry", {}).get("coordinates", [])
            s["google_maps_url"] = google_maps_url(cs, parsed["activity"])
            s["graphhopper_url"] = graphhopper_url(cs, parsed["activity"])
            routes.append(s)
    if not routes:
        return {"error": "OpenRouteService returned no usable loops here. Try a different "
                         "starting point or a shorter distance.",
                "detail": errors[:2]}

    # A slider can only change the outcome for a criterion whose scores differ
    # between routes. Surface that plainly instead of letting a knob do nothing.
    spread = {}
    if len(routes) > 1:
        for k in routes[0]["scores"]:
            vals = [r["scores"][k] for r in routes]
            spread[k] = round(max(vals) - min(vals), 1)

    # Normalise each criterion across the candidate set, then re-rank on that.
    # Absolute scores are kept for display; ranking uses the normalised values so
    # a criterion with a small absolute range still counts for its full weight.
    if len(routes) > 1:
        for k in routes[0]["scores"]:
            vals = [r["scores"][k] for r in routes]
            lo, hi = min(vals), max(vals)
            rng = hi - lo
            for r in routes:
                r.setdefault("rel", {})
                r["rel"][k] = round(50.0 if rng < 0.5
                                    else 100.0 * (r["scores"][k] - lo) / rng, 1)
        for r in routes:
            r["score"] = round(sum(weights[k] * r["rel"][k] for k in weights), 1)
    else:
        for r in routes:
            r["rel"] = dict(r["scores"])
    # Air and weather are measured once at the start point, so they are the same for
    # every candidate by definition. That is not a broken slider, and the interface
    # should say so rather than implying the knob is faulty.
    # Drop loops that cover the same ground, and loops that missed the requested
    # distance badly. A five mile request must not return a fifteen mile loop.
    def signature(r):
        cs = r["geojson"]["geometry"]["coordinates"]
        lats = [c[1] for c in cs]
        lons = [c[0] for c in cs]
        return (round(sum(lats) / len(lats), 3), round(sum(lons) / len(lons), 3),
                round(max(lats) - min(lats), 3), round(max(lons) - min(lons), 3))

    for r in routes:
        r["distance_error_pct"] = round(
            100.0 * (r["distance_m"] - parsed["target_m"]) / parsed["target_m"])

    seen, unique = set(), []
    for r in routes:
        sig = signature(r)
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(r)

    # Keep real loops. An out-and-back with a spur on the end is not what anyone
    # pictures when they ask for a loop, however close its length is.
    loops = [r for r in unique if r["loop"]["is_loop"]]
    if len(loops) < 3:
        loops = sorted(unique, key=lambda r: r["loop"]["overlap"])[:6]

    # Hard distance gate. Ask for five miles, get five miles.
    for tol in (22, 35, 55):
        on_target = [r for r in loops if abs(r["distance_error_pct"]) <= tol]
        if len(on_target) >= 3:
            break
    else:
        on_target = sorted(loops, key=lambda r: abs(r["distance_error_pct"]))[:4]

    # Six rungs spanning the measured greenery range, so the ladder covers the whole
    # spectrum rather than clustering at one end.
    on_target.sort(key=lambda r: r["scores"]["green"])
    if len(on_target) > 6:
        step = (len(on_target) - 1) / 5.0
        picked, seen_i = [], set()
        for i in range(6):
            j = round(i * step)
            if j not in seen_i:
                seen_i.add(j)
                picked.append(on_target[j])
        routes = picked
    else:
        routes = on_target

    spread = {}
    if len(routes) > 1:
        for k in routes[0]["scores"]:
            vals = [r["scores"][k] for r in routes]
            spread[k] = round(max(vals) - min(vals), 1)

    if len(routes) > 1:
        for k in routes[0]["scores"]:
            vals = [r["scores"][k] for r in routes]
            lo, hi = min(vals), max(vals)
            rng = hi - lo
            for r in routes:
                r.setdefault("rel", {})
                r["rel"][k] = round(50.0 if rng < 0.5
                                    else 100.0 * (r["scores"][k] - lo) / rng, 1)
        for r in routes:
            r["score"] = round(sum(weights[k] * r["rel"][k] for k in weights), 1)
    else:
        for r in routes:
            r["rel"] = dict(r["scores"])

    # Order the list as a ladder: least green first, greenest last. Whatever the
    # candidates turned out to be, the list always ascends in greenery, so scrubbing
    # through it always shows the route getting greener.
    routes.sort(key=lambda r: r["scores"]["green"])
    n_levels = max(1, len(routes) - 1)
    for i, r in enumerate(routes):
        r["green_level"] = i
        r["green_level_pct"] = round(100 * i / n_levels)
        r["rank"] = i + 1

    uniform_by_design = ["air", "weather"]

    # Will they still be out after dark? Real times, computed here.
    daylight = None
    sunset = (weather or {}).get("sunset")
    if sunset:
        try:
            from datetime import datetime as _dt, timedelta as _td
            ss = _dt.fromisoformat(sunset)
            now = _dt.now(ss.tzinfo) if ss.tzinfo else _dt.now()
            mins_left = (ss - now).total_seconds() / 60.0
            need = routes[0]["estimated_minutes"]
            daylight = {"sunset": ss.strftime("%-I:%M %p"),
                        "minutes_of_light": round(mins_left),
                        "minutes_needed": need,
                        "finishes_in_dark": mins_left < need,
                        "dark_minutes": max(0, round(need - mins_left))}
        except Exception:
            daylight = None

    ctx = {**parsed, "air": air, "weather": weather}
    summary = write_summary(routes[0], ctx)

    return {
        "request": {"target_m": round(parsed["target_m"]),
                    "target_mi": round(parsed["target_m"] / 1609.34, 2),
                    "activity": parsed["activity"], "stated": parsed["stated"],
                    "notes": parsed.get("notes"), "parsed_by": parsed["parsed_by"],
                    "emphasis": emphasis,
                    "emphasis_labels": [EMPHASIS_LABEL.get(e, e) for e in emphasis]},
        "weights": {"base": base_w, "applied": {k: round(v, 3) for k, v in weights.items()},
                    "changed": sorted({k for e in emphasis for k in EMPHASIS.get(e, {})})},
        "air": air,
        "weather": weather,
        "daylight": daylight,
        "spread": spread,
        "green_bias": bias,
        "hotspots": hotspots,
        "uniform_by_design": uniform_by_design,
        "green_features": len(green_pts),
        "summary": summary,
        "routes": routes,
        "warnings": errors[:2],
    }


# ─────────────────────────────────────────────── frontend, served from "/"

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Green Route</title>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --paper:#FFFFFF; --wash:#F4F9F5; --line:#DCE9DF; --ink:#132318; --soft:#5E7565;
  --leaf:#2E7D4F; --leaf-dk:#1E5B39; --leaf-lt:#E6F3EA; --stone:#8FA396;
  --d:"Fraunces",Georgia,serif; --b:"Inter",system-ui,sans-serif;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--paper);color:var(--ink);font-family:var(--b)}
#app{display:grid;grid-template-columns:380px 1fr;height:100vh}
aside{border-right:1px solid var(--line);display:flex;flex-direction:column;overflow:hidden}
#map{height:100vh;background:var(--wash)}

.head{padding:22px 24px 16px;border-bottom:1px solid var(--line)}
h1{font-family:var(--d);font-weight:700;font-size:27px;margin:0;letter-spacing:-.02em}
h1 em{font-style:normal;color:var(--leaf)}
.tag{font-size:12px;color:var(--soft);margin-top:3px}

.ask{padding:18px 24px;border-bottom:1px solid var(--line)}
textarea{width:100%;height:66px;resize:none;border:1px solid var(--line);border-radius:8px;
  padding:11px 12px;font-family:var(--b);font-size:14px;color:var(--ink);background:var(--wash)}
textarea:focus{outline:2px solid var(--leaf);outline-offset:-1px;background:#fff}
.go{width:100%;margin-top:10px;background:var(--leaf);color:#fff;border:0;border-radius:8px;
  padding:12px;font-size:14px;font-weight:600;cursor:pointer}
.go:hover{background:var(--leaf-dk)}
.go:disabled{opacity:.55;cursor:not-allowed}
.tiny{font-size:11.5px;color:var(--stone);margin:9px 0 0;text-align:center}
.tiny a{color:var(--leaf);cursor:pointer;text-decoration:underline}

.body{flex:1;overflow-y:auto;padding:18px 24px 24px}

.dialbox{background:var(--leaf-lt);border-radius:10px;padding:15px 16px;margin-bottom:16px}
.dialbox h3{font-family:var(--d);font-weight:700;font-size:16px;margin:0 0 3px}
.dialbox p{font-size:12px;color:var(--soft);margin:0 0 12px}
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:6px;
  background:linear-gradient(90deg,#CBD8CE,var(--leaf));border-radius:4px;outline:none}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:24px;height:24px;
  border-radius:50%;background:#fff;border:3px solid var(--leaf);cursor:pointer;
  box-shadow:0 1px 4px rgba(0,0,0,.18)}
input[type=range]::-moz-range-thumb{width:22px;height:22px;border-radius:50%;background:#fff;
  border:3px solid var(--leaf);cursor:pointer}
.ends{display:flex;justify-content:space-between;font-size:11px;color:var(--soft);margin-top:6px}
.rung{text-align:center;font-family:var(--d);font-weight:700;font-size:15px;margin-top:8px}
.rung b{color:var(--leaf)}

.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:9px;margin-bottom:16px}
.stat{background:var(--wash);border:1px solid var(--line);border-radius:9px;padding:11px 8px;
  text-align:center}
.stat b{display:block;font-family:var(--d);font-weight:700;font-size:20px;line-height:1.1}
.stat span{font-size:10px;color:var(--soft);text-transform:uppercase;letter-spacing:.07em}

.sum{background:var(--wash);border-left:3px solid var(--leaf);border-radius:0 8px 8px 0;
  padding:12px 14px;font-size:13.5px;line-height:1.6;color:#243A2B;margin-bottom:16px}

.mini{display:grid;grid-template-columns:repeat(6,1fr);gap:5px;margin-bottom:16px}
.mini button{border:1px solid var(--line);background:#fff;border-radius:7px;padding:8px 2px;
  cursor:pointer;font-family:var(--b);font-size:11px;color:var(--soft)}
.mini button b{display:block;font-family:var(--d);font-weight:700;font-size:15px;color:var(--ink)}
.mini button[aria-pressed="true"]{border-color:var(--leaf);background:var(--leaf-lt)}
.mini button[aria-pressed="true"] b{color:var(--leaf-dk)}

.nav{display:block;text-align:center;background:var(--leaf);color:#fff;text-decoration:none;
  border-radius:8px;padding:11px;font-size:13.5px;font-weight:600;margin-bottom:7px}
.nav:hover{background:var(--leaf-dk)}
.nav.ghost{background:#fff;color:var(--leaf);border:1px solid var(--line)}
.nav.ghost:hover{border-color:var(--leaf);background:var(--leaf-lt)}

details{margin-top:16px;border-top:1px solid var(--line);padding-top:12px}
summary{font-size:12.5px;color:var(--soft);cursor:pointer}
.crit{display:grid;grid-template-columns:96px 1fr 30px;gap:8px;align-items:center;
  font-size:11.5px;color:var(--soft);margin-top:7px}
.crit .tr{height:6px;background:#EDF3EE;border-radius:3px;overflow:hidden}
.crit .tr i{display:block;height:100%;background:var(--leaf)}
.crit .v{font-variant-numeric:tabular-nums;color:var(--ink);text-align:right}
.crit input[type=range]{height:5px}
nav.tabs{display:flex;border-bottom:1px solid var(--line)}
nav.tabs button{flex:1;background:#fff;border:0;border-bottom:3px solid transparent;
  padding:12px;font-family:var(--b);font-size:13px;font-weight:600;color:var(--soft);
  cursor:pointer}
nav.tabs button[aria-selected="true"]{color:var(--leaf-dk);border-bottom-color:var(--leaf);
  background:var(--leaf-lt)}
.pane{display:none}.pane.on{display:block}

.mix{background:#fff;border:1px solid var(--line);border-radius:10px;padding:13px 15px;
  margin-bottom:16px}
.mix h4{font-family:var(--d);font-weight:700;font-size:14px;margin:0 0 2px}
.mix p{font-size:11.5px;color:var(--soft);margin:0 0 11px;line-height:1.5}
.mx{display:grid;grid-template-columns:74px 1fr;gap:9px;align-items:center;
  font-size:12px;color:var(--soft);margin-bottom:7px}

.card{border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:9px;
  cursor:pointer;background:#fff}
.card:hover{border-color:var(--leaf)}
.card.on{border-color:var(--leaf);background:var(--leaf-lt)}
.card h4{font-family:var(--d);font-weight:700;font-size:15px;margin:0 0 3px}
.card .meta{font-size:11.5px;color:var(--soft)}
.card .rm{float:right;color:var(--stone);font-size:11px;text-decoration:underline}
.roombar{background:var(--leaf-lt);border-radius:10px;padding:12px 14px;margin-bottom:14px}
.roombar .saverow{margin-bottom:4px}
.roombar input{text-transform:uppercase;letter-spacing:.12em;font-weight:600}
.editbox{display:none;margin-top:11px;padding-top:11px;border-top:1px solid var(--line)}
.card.on .editbox{display:block}
.saverow{display:grid;grid-template-columns:1fr auto;gap:7px;margin-bottom:7px}
.saverow input{border:1px solid var(--line);border-radius:8px;padding:9px 10px;
  font-family:var(--b);font-size:13px}
.err{background:#FDECEA;border-left:3px solid #C0392B;color:#7A2318;padding:11px 13px;
  border-radius:0 8px 8px 0;font-size:13px;line-height:1.5}
.empty{color:var(--stone);font-size:13px;line-height:1.6}
.spin{display:inline-block;width:13px;height:13px;border:2px solid rgba(255,255,255,.35);
  border-top-color:#fff;border-radius:50%;animation:sp .7s linear infinite;
  vertical-align:-2px;margin-right:7px}
@keyframes sp{to{transform:rotate(360deg)}}
@media(prefers-reduced-motion:reduce){.spin{animation:none}}
@media(max-width:860px){#app{grid-template-columns:1fr;height:auto}#map{height:56vh}
  aside{border-right:0;border-bottom:1px solid var(--line)}}
</style>
</head>
<body>
<div id="app">
  <aside>
    <div class="head">
      <h1>Green<em>Route</em></h1>
      <div class="tag">Running loops, your way</div>
    </div>

    <nav class="tabs">
      <button data-tab="plan" aria-selected="true">Plan</button>
      <button data-tab="saved" aria-selected="false">Saved routes</button>
    </nav>

    <div class="ask pane on" id="ask">
      <textarea id="prompt" placeholder="I want to run 5 miles">I want to run 5 miles</textarea>
      <button class="go" id="go">Find routes</button>
      <p class="tiny"><a id="locate">use my location</a> &nbsp;·&nbsp; or click the map</p>
    </div>

    <div class="body">
      <div class="pane on" id="pane-plan">
        <div id="out">
          <p class="empty">Type a distance, pick a start on the map, and press Find routes.</p>
        </div>
      </div>
      <div class="pane" id="pane-saved">
        <div class="roombar">
          <div class="saverow">
            <input id="roomcode" placeholder="Room code" maxlength="8">
            <button class="go" id="joinroom" style="width:auto;padding:9px 14px">Open</button>
          </div>
          <p class="tiny" id="roomnote">Enter a code to open a room, or
            <a id="newroom">make a new one</a>.</p>
        </div>
        <p class="empty" id="savedempty">Nothing saved in this room yet.</p>
        <div id="savedlist"></div>
      </div>
    </div>
  </aside>
  <div id="map"></div>
</div>

<script>
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const HOME = [37.6624, -121.8747];
let MAP = null, layers = [], startMarker = null, DATA = null, SEL = 0, LABELS = {},
    CORR = {}, USER_W = null, ROOM = '';

/* Room code lives in the URL, so sharing the link shares the room. */
function readRoom() {
  const mm = location.hash.match(/room=([A-Z0-9]{1,8})/i);
  return mm ? mm[1].toUpperCase() : '';
}
function setRoom(code) {
  ROOM = (code || '').toUpperCase().slice(0, 8);
  location.hash = ROOM ? 'room=' + ROOM : '';
  const inp = document.getElementById('roomcode');
  if (inp) inp.value = ROOM;
  const note = document.getElementById('roomnote');
  if (note) note.innerHTML = ROOM
    ? `Room <b>${ROOM}</b> — share this page's link to invite others.`
    : 'Enter a code to open a room, or <a id="newroom">make a new one</a>.';
  const nr = document.getElementById('newroom');
  if (nr) nr.addEventListener('click', makeRoom);
}
function makeRoom() {
  const A = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
  let c = '';
  for (let i = 0; i < 6; i++) c += A[Math.floor(Math.random() * A.length)];
  setRoom(c);
  loadSaved();
}

/* Older saved routes may predate the stored link, so rebuild it from the
   geometry rather than hiding the button. */
function gmapsFor(geo) {
  const cs = (geo && geo.geometry && geo.geometry.coordinates) || [];
  if (cs.length < 2) return '';
  const k = Math.min(22, cs.length);
  const step = (cs.length - 1) / (k - 1);
  const pts = [];
  for (let i = 0; i < k; i++) pts.push(cs[Math.min(cs.length - 1, Math.round(i * step))]);
  if (pts[pts.length - 1] !== cs[0]) pts.push(cs[0]);
  const legs = pts.map(c => `${c[1].toFixed(5)},${c[0].toFixed(5)}`).join('/');
  return `https://www.google.com/maps/dir/${legs}/data=!3m1!4b1!4m2!4m1!3e2`;
}

/* ── map ─────────────────────────────────────────────────────────────────── */
(function () {
  const css = document.createElement('link');
  css.rel = 'stylesheet';
  css.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
  document.head.appendChild(css);
  const js = document.createElement('script');
  js.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
  js.onload = initMap;
  document.head.appendChild(js);
})();

let m = null;
function initMap() {
  m = L.map('map', {zoomControl: true}).setView(HOME, 14);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    {maxZoom: 19, attribution: '&copy; OpenStreetMap'}).addTo(m);
  m.on('click', e => setStart(e.latlng.lat, e.latlng.lng));
  setStart(HOME[0], HOME[1]);
}
let LAT = HOME[0], LON = HOME[1];
function setStart(lat, lon) {
  LAT = lat; LON = lon;
  if (startMarker) m.removeLayer(startMarker);
  startMarker = L.circleMarker([lat, lon], {radius: 8, color: '#2E7D4F', weight: 3,
    fillColor: '#fff', fillOpacity: 1}).addTo(m).bindTooltip('Start');
}
function drawRoutes(sel) {
  layers.forEach(l => m.removeLayer(l));
  layers = [];
  if (!DATA) return;
  DATA.routes.forEach((r, i) => {
    if (i === sel) return;
    layers.push(L.polyline(r.geojson.geometry.coordinates.map(c => [c[1], c[0]]),
      {color: '#9DBFA8', weight: 2.5, opacity: .5}).addTo(m).on('click', () => pick(i)));
  });
  const r = DATA.routes[sel];
  const main = L.polyline(r.geojson.geometry.coordinates.map(c => [c[1], c[0]]),
    {color: '#2E7D4F', weight: 6, opacity: 1, lineJoin: 'round'}).addTo(m);
  layers.push(main);
  m.fitBounds(main.getBounds(), {padding: [40, 40]});
}

/* ── data ────────────────────────────────────────────────────────────────── */
fetch('/api/criteria').then(r => r.json()).then(c => {
  LABELS = Object.fromEntries(c.criteria.map(x => [x.key, x.label]));
  CORR = c.green_correlation || {};
}).catch(() => {});

$('#locate').addEventListener('click', () => {
  if (!navigator.geolocation) return;
  navigator.geolocation.getCurrentPosition(p => {
    setStart(p.coords.latitude, p.coords.longitude);
    m.setView([p.coords.latitude, p.coords.longitude], 15);
  }, () => alert('Could not get your location.'), {timeout: 10000});
});

setRoom(readRoom());
$('#joinroom').addEventListener('click', () => {
  setRoom(document.getElementById('roomcode').value);
  loadSaved();
});
$('#go').addEventListener('click', plan);
async function plan() {
  const b = $('#go');
  b.disabled = true;
  b.innerHTML = '<span class="spin"></span>Finding routes';
  $('#out').innerHTML = '<p class="empty">Planning…</p>';
  try {
    const res = await fetch('/api/plan-route', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({lat: LAT, lon: LON, prompt: $('#prompt').value})});
    const d = await res.json();
    if (d.error) $('#out').innerHTML = `<div class="err">${esc(d.error)}</div>`;
    else { DATA = d; USER_W = null; SEL = Math.min(3, d.routes.length - 1); render(); }
  } catch (e) {
    $('#out').innerHTML = `<div class="err">${esc(e.message)}</div>`;
  } finally { b.disabled = false; b.textContent = 'Find routes'; }
}

function pick(i) { SEL = i; render(); }

/* Slider mix. Each control carries a signed pull; the combined value selects
   which of the loaded loops to display. */
const MIX = [
  {k: 'quiet',  label: 'Quiet',      w:  1.0, def: 50},
  {k: 'air',    label: 'Clean air',  w:  0.9, def: 50},
  {k: 'safety', label: 'Safety',     w:  0.8, def: 50},
  {k: 'smooth', label: 'Smooth',     w: -0.9, def: 50},
  {k: 'simple', label: 'Few turns',  w: -0.7, def: 50},
];
let MIXV = Object.fromEntries(MIX.map(x => [x.k, x.def]));

function mixToRung() {
  const n = DATA.routes.length;
  let pull = 0, tot = 0;
  for (const s of MIX) {
    pull += ((MIXV[s.k] - 50) / 50) * s.w;
    tot += Math.abs(s.w);
  }
  const t = Math.max(-1, Math.min(1, pull / (tot * 0.55)));   // -1 .. +1
  return Math.max(0, Math.min(n - 1, Math.round(((t + 1) / 2) * (n - 1))));
}

document.querySelectorAll('nav.tabs button').forEach(b =>
  b.addEventListener('click', () => {
    document.querySelectorAll('nav.tabs button').forEach(x =>
      x.setAttribute('aria-selected', x === b));
    const t = b.dataset.tab;
    document.getElementById('pane-plan').classList.toggle('on', t === 'plan');
    document.getElementById('pane-saved').classList.toggle('on', t === 'saved');
    document.getElementById('ask').style.display = t === 'plan' ? '' : 'none';
    if (t === 'saved') loadSaved();
  }));

async function loadSaved() {
  const box = document.getElementById('savedlist');
  const empty = document.getElementById('savedempty');
  try {
    const d = await (await fetch('/api/saved?room=' + encodeURIComponent(ROOM))).json();
    const rs = d.routes || [];
    empty.style.display = rs.length ? 'none' : '';
    box.innerHTML = rs.map(r => `
      <div class="card" data-id="${esc(r.id)}">
        <a class="rm" data-del="${esc(r.id)}">remove</a>
        <h4>${esc(r.name)}</h4>
        <div class="meta">${r.distance_mi} mi · ${r.minutes} min · greenery
          ${Math.round(r.green ?? 0)}/100 · by ${esc(r.by)} · ${esc(r.saved_at)}</div>
        ${r.note ? `<div class="meta" style="margin-top:4px;font-style:italic">${
          esc(r.note)}</div>` : ''}
        <div class="editbox" id="ed-${esc(r.id)}">
          <div class="saverow">
            <input data-f="name" value="${esc(r.name)}" placeholder="Name">
            <input data-f="by" value="${esc(r.by)}" placeholder="You" style="width:88px">
          </div>
          <input data-f="note" value="${esc(r.note || '')}" placeholder="Add a note"
            style="width:100%;border:1px solid var(--line);border-radius:8px;
            padding:9px 10px;font-family:var(--b);font-size:13px;margin-bottom:7px">
          <button class="go" data-save="${esc(r.id)}">Save changes</button>
          <a class="nav" style="margin-top:7px" data-gmap="${esc(r.id)}"
            target="_blank" rel="noopener"
            href="${esc(r.google_maps_url || '#')}">Navigate in Google Maps</a>
        </div>
      </div>`).join('');

    box.querySelectorAll('.card').forEach(c =>
      c.addEventListener('click', async e => {
        if (e.target.closest('a') || e.target.closest('button') ||
            e.target.tagName === 'INPUT') return;
        const full = await (await fetch('/api/saved/' + c.dataset.id)).json();
        if (full.geojson) showSaved(full);
        box.querySelectorAll('.card').forEach(x => x.classList.toggle('on', x === c));
      }));

    box.querySelectorAll('[data-del]').forEach(a =>
      a.addEventListener('click', async e => {
        e.stopPropagation();
        await fetch('/api/saved/' + a.dataset.del, {method: 'DELETE'});
        loadSaved();
      }));

    // fill in any missing map link from the stored geometry, on demand
    box.querySelectorAll('[data-gmap]').forEach(a =>
      a.addEventListener('click', async e => {
        e.stopPropagation();
        if (a.getAttribute('href') !== '#') return;
        e.preventDefault();
        const full = await (await fetch('/api/saved/' + a.dataset.gmap)).json();
        const url = full.google_maps_url || gmapsFor(full.geojson);
        if (url) window.open(url, '_blank', 'noopener');
      }));

    box.querySelectorAll('[data-save]').forEach(b =>
      b.addEventListener('click', async e => {
        e.stopPropagation();
        const card = b.closest('.card');
        const get = f => card.querySelector(`[data-f="${f}"]`).value;
        b.disabled = true;
        b.textContent = 'Saving';
        await fetch('/api/saved/' + b.dataset.save, {method: 'PATCH',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({name: get('name'), by: get('by'), note: get('note')})});
        loadSaved();
      }));
  } catch (e) { box.innerHTML = `<div class="err">${esc(e.message)}</div>`; }
}

function showSaved(r) {
  layers.forEach(l => m.removeLayer(l));
  layers = [];
  const line = L.polyline(r.geojson.geometry.coordinates.map(c => [c[1], c[0]]),
    {color: '#2E7D4F', weight: 6, opacity: 1, lineJoin: 'round'}).addTo(m);
  layers.push(line);
  m.fitBounds(line.getBounds(), {padding: [40, 40]});
}

async function saveCurrent() {
  const r = DATA.routes[SEL];
  const name = (document.getElementById('savename').value || '').trim()
    || `${r.distance_mi} mi loop`;
  const by = (document.getElementById('saveby').value || '').trim() || 'anonymous';
  const btn = document.getElementById('savebtn');
  btn.disabled = true;
  btn.textContent = 'Saving';
  try {
    await fetch('/api/saved', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name, by, distance_mi: r.distance_mi,
        minutes: r.estimated_minutes, elevation_gain_m: r.elevation_gain_m,
        green: r.scores.green,
        google_maps_url: r.google_maps_url || gmapsFor(r.geojson),
        graphhopper_url: r.graphhopper_url, room: ROOM,
        start: [LAT, LON], geojson: r.geojson})});
    btn.textContent = 'Saved to the public board';
  } catch (e) { btn.textContent = 'Could not save'; }
  finally { setTimeout(() => { btn.disabled = false; btn.textContent = 'Save this route'; }, 1600); }
}

function combinedPull() {
  if (!USER_W || !DATA) return null;
  const base = DATA.weights.base;
  let p = 0;
  for (const k in CORR) p += ((USER_W[k] ?? 0) - (base[k] ?? 0)) * CORR[k];
  return Math.max(-1, Math.min(1, p * 4));
}

function render() {
  const d = DATA, n = d.routes.length, r = d.routes[SEL];
  const pct = n > 1 ? Math.round((SEL / (n - 1)) * 100) : 100;
  $('#out').innerHTML = `
    <div class="dialbox">
      <h3>Greenery</h3>
      <input type="range" min="0" max="${n - 1}" value="${SEL}" id="dial">
      <div class="ends"><span>built up</span><span>leafiest</span></div>
      <div class="rung">${SEL + 1} of ${n} &nbsp;·&nbsp; greenery
        <b>${Math.round(r.scores.green)}</b></div>
    </div>

    <div class="mix">
      <h4>What matters to you</h4>
      ${MIX.map(x => `<div class="mx"><span>${x.label}</span>
        <input type="range" min="0" max="100" value="${MIXV[x.k]}" data-mix="${x.k}">
        </div>`).join('')}
    </div>

    <div class="stats">
      <div class="stat"><b>${r.distance_mi}</b><span>miles</span></div>
      <div class="stat"><b>${r.estimated_minutes}</b><span>minutes</span></div>
      <div class="stat"><b>${r.elevation_gain_m}</b><span>m climb</span></div>
    </div>

    <div class="mini">${d.routes.map((x, i) =>
      `<button data-i="${i}" aria-pressed="${i === SEL}">
        <b>${Math.round(x.scores.green)}</b>${x.distance_mi}mi</button>`).join('')}</div>

    <div class="sum">${esc(d.summary)}</div>

    ${r.google_maps_url ? `<a class="nav" href="${esc(r.google_maps_url)}"
      target="_blank" rel="noopener">Navigate in Google Maps</a>` : ''}
    ${r.graphhopper_url ? `<a class="nav ghost" href="${esc(r.graphhopper_url)}"
      target="_blank" rel="noopener">Open the exact loop</a>` : ''}
    <a class="nav ghost" id="gpx">Download GPX</a>

    <div class="saverow" style="margin-top:14px">
      <input id="savename" placeholder="Name this loop">
      <input id="saveby" placeholder="You" style="width:88px">
    </div>
    <button class="go" id="savebtn">Save this route</button>
    <p class="tiny">Saved to your room.</p>

    <details>
      <summary>What went into this score</summary>
      <div id="crits"></div>
    </details>`;

  $('#dial').addEventListener('input', e => { SEL = +e.target.value; render(); });
  document.querySelectorAll('[data-mix]').forEach(inp =>
    inp.addEventListener('input', () => {
      MIXV[inp.dataset.mix] = +inp.value;
      SEL = mixToRung();
      render();
    }));
  $('#savebtn').addEventListener('click', saveCurrent);
  document.querySelectorAll('.mini button').forEach(b =>
    b.addEventListener('click', () => pick(+b.dataset.i)));
  $('#gpx').addEventListener('click', () => toGPX(r));
  $('#crits').innerHTML = Object.keys(r.scores).map(k =>
    `<div class="crit"><span>${esc(LABELS[k] || k)}</span>
      <span class="tr"><i style="width:${Math.max(2, r.scores[k])}%"></i></span>
      <span class="v">${Math.round(r.scores[k])}</span></div>`).join('') +
    `<p class="tiny" style="text-align:left;margin-top:10px">All figures computed from
      routing and sensor data.</p>`;

  drawRoutes(SEL);
}

function toGPX(r) {
  const pts = r.geojson.geometry.coordinates.map(c =>
    `<trkpt lat="${c[1].toFixed(6)}" lon="${c[0].toFixed(6)}">${
      c.length > 2 ? `<ele>${c[2].toFixed(1)}</ele>` : ''}</trkpt>`).join('');
  const gpx = `<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="GreenRoute" xmlns="http://www.topografix.com/GPX/1/1">
<trk><name>${r.distance_mi} mi loop</name><trkseg>${pts}</trkseg></trk></gpx>`;
  const url = URL.createObjectURL(new Blob([gpx], {type: 'application/gpx+xml'}));
  const a = document.createElement('a');
  a.href = url;
  a.download = `greenroute-${r.distance_mi}mi.gpx`;
  a.click();
  URL.revokeObjectURL(url);
}
</script>
</body>
</html>"""

