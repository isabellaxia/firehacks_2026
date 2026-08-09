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
import math
import os
import re
from typing import Any

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
    q = f"""[out:json][timeout:18];
(
  way["leisure"~"park|garden|nature_reserve|recreation_ground"](around:{r},{lat},{lon});
  way["landuse"~"forest|grass|meadow|village_green|recreation"](around:{r},{lat},{lon});
  way["natural"~"wood|water|scrub|grassland"](around:{r},{lat},{lon});
  relation["leisure"="park"](around:{r},{lat},{lon});
);
out center 300;"""
    pts = []
    for host in OVERPASS_HOSTS:
        try:
            resp = requests.get(host, params={"data": q}, timeout=20,
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
                 emphasis: list[str] | None = None, n: int = 6,
                 hotspots: list | None = None, green_bias: float = 0.0) -> list[dict]:
    """Candidate loops that are actually different from one another.

    OpenRouteService's round_trip helper reseeded a few times tends to return the
    same loop with minor variations, which is useless: re-weighting the score can
    only choose between the routes you gave it. So instead of reseeding, we send
    the loop off in a different COMPASS DIRECTION each time — two via-points placed
    on a bearing, routed start -> p1 -> p2 -> start. Six bearings, six loops that
    fan out across the map and genuinely trade off against each other.

    Preferences also change the request itself: asking for a safe route makes ORS
    avoid steps and ferries, so the geometry differs too, not just the ranking.
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

    # A triangle start -> p1 -> p2 -> start of side r has perimeter about 3r along
    # straight lines; real streets add roughly 25%, so aim a little short.
    r = (target_m / 3.0) * 0.78
    # Alternating the reach as well as the bearing pushes some loops further out,
    # where the streets, surfaces and greenery genuinely differ from the blocks
    # right around the start.
    spread = [(0, 1.0), (60, 0.75), (120, 1.15), (180, 1.0), (240, 0.75), (300, 1.15)]

    # green_bias > 0 means the runner wants greenery: aim loops straight at the
    # densest parks nearby. green_bias < 0 means they do not: aim away from them.
    # Half the candidates follow the bias, half stay spread out, so there is always
    # something to compare against.
    bearings = []
    if hotspots and abs(green_bias) > 0.15:
        for h in hotspots[:3]:
            b = h["bearing"] if green_bias > 0 else (h["bearing"] + 180) % 360
            reach = max(0.55, min(1.35, (h["distance_m"] / max(1.0, r)))) \
                if green_bias > 0 else 1.0
            bearings.append((b, reach))
            bearings.append(((b + 40) % 360, reach * 0.85))
    for spec in spread:
        if len(bearings) >= n:
            break
        bearings.append(spec)
    bearings = bearings[:n]

    def one(spec):
        """One bearing and reach, one loop. Parallel, or the request times out."""
        b, scale = spec
        rr = r * scale
        p1 = offset(lat, lon, b, rr)
        p2 = offset(lat, lon, b + 72, rr)
        body = {
            "coordinates": [[lon, lat], [p1[1], p1[0]], [p2[1], p2[0]], [lon, lat]],
            "elevation": True,
            "instructions": True,
            "preference": "shortest" if "exact" in emphasis else "recommended",
            "extra_info": ["surface", "waytype", "steepness", "green", "noise"],
        }
        if avoid:
            body["options"] = {"avoid_features": avoid}
        try:
            resp = requests.post(ORS_URL.format(profile=profile), json=body,
                                 headers={"Authorization": ORS_KEY,
                                          "Content-Type": "application/json"}, timeout=18)
            if resp.status_code != 200:
                return {"error": f"ORS {resp.status_code}: {resp.text[:160]}"}
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    # Six sequential calls at 30s each can exceed the platform request timeout, which
    # returns an HTML error page and breaks the client. Fan out instead.
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        out = list(pool.map(one, bearings))

    # If every bearing failed, fall back to the round_trip helper so the app still
    # returns something rather than an empty page.
    if not any("error" not in x for x in out):
        for i in range(2):
            body = {"coordinates": [[lon, lat]], "elevation": True, "instructions": True,
                    "extra_info": ["surface", "waytype", "steepness", "green", "noise"],
                    "options": {"round_trip": {"length": int(target_m), "points": 3 + i,
                                               "seed": 1 + i * 11}}}
            try:
                resp = requests.post(ORS_URL.format(profile=profile), json=body,
                                     headers={"Authorization": ORS_KEY,
                                              "Content-Type": "application/json"},
                                     timeout=15)
                if resp.status_code == 200:
                    out.append(resp.json())
            except Exception:
                pass
    return out


# ────────────────────────────────────────────────────────────── model calls

def parse_request(prompt: str) -> dict:
    """Free text → target distance in metres and activity. Model picks, Python validates."""
    fallback = _regex_parse(prompt)
    if not FEATHERLESS_KEY:
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
        return {"target_m": max(400.0, min(50000.0, metres)), "activity": act,
                "notes": got.get("notes"), "parsed_by": PARSE_MODEL,
                "emphasis": emph, "stated": f"{val} {unit}"}
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
        "extras": {k: v.get("values", []) for k, v in (extras or {}).items()},
        "elevation": {"points": [round(e, 1) for e in eles[::max(1, len(eles)//120)]],
                      "min": round(min(eles), 1) if eles else None,
                      "max": round(max(eles), 1) if eles else None,
                      "descent": round(float(props.get("descent") or 0))},
        "geojson": f,
    }


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
    """A turn-by-turn link the user can open in Google Maps.

    Google's URL scheme takes an origin, a destination and up to nine waypoints, so
    we sample the loop evenly. It is the same loop, walkable turn by turn on a phone.
    """
    if not coords or len(coords) < 2:
        return ""
    picked = sample_points(coords, 10)
    pts = [(c[1], c[0]) for c in picked]          # GeoJSON is [lon, lat]
    start = pts[0]
    way = pts[1:-1][:8]
    mode = {"cycle": "bicycling"}.get(activity, "walking")
    fmt = lambda p: f"{p[0]:.5f},{p[1]:.5f}"
    url = ("https://www.google.com/maps/dir/?api=1"
           f"&origin={fmt(start)}&destination={fmt(start)}&travelmode={mode}")
    if way:
        url += "&waypoints=" + "|".join(fmt(p) for p in way)
    return url


# ────────────────────────────────────────────────────────────── API

class PlanRequest(BaseModel):
    lat: float
    lon: float
    prompt: str = "I want to run 5 km."
    # -1 aims the loops away from parks, +1 aims them straight at the nearest ones.
    green_bias: float | None = None


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

    parsed = parse_request(req.prompt)
    emphasis = parsed.get("emphasis") or []
    base_w = WEIGHTS.get(parsed["activity"], WEIGHTS["run"])
    weights = apply_emphasis(base_w, emphasis)
    climb_ideal = next((CLIMB_OVERRIDE[e] for e in emphasis if e in CLIMB_OVERRIDE), None)

    # context lookups run together; none of them should be able to stall the plan
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        f_air = pool.submit(fetch_air, req.lat, req.lon)
        f_wx = pool.submit(fetch_weather, req.lat, req.lon)
        f_gr = pool.submit(fetch_green, req.lat, req.lon, parsed["target_m"] * 0.8)
        air = f_air.result()
        weather = f_wx.result()
        try:
            green_pts = f_gr.result(timeout=22)
        except Exception:
            green_pts = []

    hotspots = green_hotspots(req.lat, req.lon, green_pts)

    # Where the greenery bias comes from: an explicit slider value if the client
    # sent one, otherwise the preference the model read from the request text.
    if req.green_bias is not None:
        bias = max(-1.0, min(1.0, req.green_bias))
    elif "green" in emphasis or "scenic" in emphasis:
        bias = 1.0
    else:
        bias = 0.0

    try:
        raw = fetch_routes(req.lat, req.lon, parsed["target_m"], parsed["activity"],
                           emphasis, 6, hotspots, bias)
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
    uniform_by_design = ["air", "weather"]

    routes.sort(key=lambda r: -r["score"])
    for rank, r in enumerate(routes, 1):
        r["rank"] = rank
    # Drop loops that cover the same ground. Two routes are "the same" if their
    # midpoints and their bounding boxes nearly coincide — distance alone is not
    # enough, since two different loops can be the same length.
    def signature(r):
        cs = r["geojson"]["geometry"]["coordinates"]
        lats = [c[1] for c in cs]
        lons = [c[0] for c in cs]
        return (round(sum(lats) / len(lats), 3), round(sum(lons) / len(lons), 3),
                round(max(lats) - min(lats), 3), round(max(lons) - min(lons), 3))

    seen, unique = set(), []
    for r in routes:
        sig = signature(r)
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(r)
    routes = unique[:6]

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
<title>Route Planner — exercise routes scored on nine criteria</title>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --pine:#0F1F1B; --panel:#16302A; --card:#1D3B34; --line:#2C544A;
  --ink:#EAF2ED; --dim:#8FAFA4;
  --blaze:#E8622C;              /* trail blaze orange */
  --chalk:#9EC4D2; --good:#5FBF8B; --warn:#E8B84B;
  --d:"Barlow Condensed",system-ui,sans-serif;
  --b:"Inter",system-ui,sans-serif;
  --m:"JetBrains Mono",ui-monospace,monospace;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%;font-family:var(--b);background:var(--pine);color:var(--ink)}
button{font-family:inherit}

#app{display:grid;grid-template-columns:35% 65%;height:100vh}
aside{background:var(--panel);overflow-y:auto;padding:22px;border-right:1px solid var(--line)}
#map{height:100vh;background:#0B1714}

h1{font-family:var(--d);font-weight:700;font-size:30px;letter-spacing:.01em;margin:0;
  text-transform:uppercase;line-height:1}
h1 span{color:var(--blaze)}
.sub{font-size:11.5px;color:var(--dim);letter-spacing:.06em;margin:5px 0 20px}

label{display:block;font-family:var(--d);font-weight:600;font-size:12px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--dim);margin:0 0 7px}
input,textarea{width:100%;background:#0F2721;color:var(--ink);border:1px solid var(--line);
  border-radius:3px;padding:10px;font-family:var(--m);font-size:13px}
textarea{height:74px;resize:vertical;font-family:var(--b)}
input:focus,textarea:focus,button:focus-visible{outline:2px solid var(--blaze);outline-offset:1px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:9px}
.row{margin-bottom:14px}

.ghost{background:transparent;color:var(--chalk);border:1px solid var(--line);border-radius:3px;
  padding:8px 12px;font-size:12px;cursor:pointer;width:100%;margin-top:8px}
.ghost:hover{border-color:var(--chalk);color:#fff}
.primary{width:100%;background:var(--blaze);color:#1A0B04;border:0;border-radius:3px;
  padding:13px;font-family:var(--d);font-weight:700;font-size:16px;letter-spacing:.12em;
  text-transform:uppercase;cursor:pointer;margin-top:4px}
.primary:disabled{opacity:.5;cursor:not-allowed}

.chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
.chip{background:#0F2721;border:1px solid var(--line);color:var(--dim);border-radius:20px;
  padding:5px 11px;font-size:11.5px;cursor:pointer}
.chip:hover{border-color:var(--blaze);color:var(--ink)}

hr{border:0;border-top:1px solid var(--line);margin:20px 0}

.cond{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}
.cond div{background:var(--card);border-radius:3px;padding:8px 11px;flex:1;min-width:78px}
.cond b{display:block;font-family:var(--d);font-weight:700;font-size:21px;line-height:1.1}
.cond small{font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim)}

.summary{background:var(--card);border-left:3px solid var(--blaze);padding:14px 16px;
  font-size:14px;line-height:1.6;margin-bottom:18px}
.summary .tagline{font-family:var(--d);font-weight:600;font-size:11px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--blaze);margin-bottom:7px}

/* race-bib style candidate cards */
.cand{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--line);
  border-radius:3px;padding:12px 14px;margin-bottom:9px;cursor:pointer;width:100%;
  text-align:left;color:inherit;display:block}
.cand:hover{border-color:var(--chalk)}
.cand.on{border-left-color:var(--blaze);background:#22463D}
.cand .top{display:flex;align-items:baseline;gap:10px}
.bib{font-family:var(--d);font-weight:700;font-size:27px;line-height:1;color:var(--blaze);
  min-width:44px}
.cand .facts{font-family:var(--m);font-size:11.5px;color:var(--dim);margin-top:3px}
.cand .facts b{color:var(--ink);font-weight:500}
.bar{height:5px;background:#0F2721;border-radius:3px;margin-top:9px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--blaze)}

.breakdown{margin-top:11px;display:none}
.cand.on .breakdown{display:block}
.crit{display:grid;grid-template-columns:88px 1fr 34px;gap:8px;align-items:center;
  font-size:11px;margin-bottom:4px;color:var(--dim)}
.crit .track{height:6px;background:#0F2721;border-radius:3px;overflow:hidden}
.crit .track i{display:block;height:100%;background:var(--chalk)}
.crit .track i.hi{background:var(--good)}
.crit .track i.lo{background:var(--warn)}
.crit .v{font-family:var(--m);color:var(--ink);text-align:right}
.crit .est{color:var(--warn)}
.note{font-size:10.5px;color:var(--dim);line-height:1.5;margin-top:9px}

.prefs{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 12px}
.pref{background:#2A5348;border:1px solid var(--blaze);color:#FFD9C6;border-radius:20px;
  padding:4px 11px;font-size:11.5px;font-weight:500}
.pref.none{background:#1D3B34;border-color:var(--line);color:var(--dim)}
.boosted{color:var(--blaze) !important;font-weight:600}
.nav{display:block;text-align:center;background:#2A5348;border:1px solid var(--blaze);
  color:#FFD9C6;text-decoration:none;border-radius:3px;padding:9px;margin-top:10px;
  font-family:var(--d);font-weight:600;font-size:13px;letter-spacing:.1em;
  text-transform:uppercase}
.nav:hover{background:var(--blaze);color:#1A0B04}
.paint{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:12px}
.paint button{background:#0F2721;border:1px solid var(--line);color:var(--dim);
  border-radius:20px;padding:5px 11px;font-size:11px;cursor:pointer}
.paint button[aria-pressed="true"]{border-color:var(--blaze);color:#FFD9C6;background:#2A5348}
.legend{display:flex;gap:12px;font-size:10px;color:var(--dim);margin:-6px 0 12px;
  align-items:center;flex-wrap:wrap}
.legend i{display:inline-block;width:22px;height:6px;border-radius:2px;margin-right:4px;
  vertical-align:middle}
.elev{margin-top:9px;background:#0F2721;border-radius:3px;padding:7px 8px 4px}
.elev svg{display:block;width:100%;height:52px}
.elev .cap{display:flex;justify-content:space-between;font-size:9.5px;color:var(--dim);
  margin-top:2px}
.dark{background:#4A3A18;color:#F5E3B8;border-left:3px solid var(--warn);padding:10px 13px;
  font-size:12.5px;line-height:1.5;margin-bottom:14px}
.dark b{color:#FFF3D6}
.gpx{display:block;text-align:center;background:transparent;border:1px solid var(--line);
  color:var(--chalk);border-radius:3px;padding:8px;margin-top:7px;font-size:12px;
  cursor:pointer;width:100%}
.gpx:hover{border-color:var(--chalk);color:#fff}
.radar{background:#0F2721;border-radius:3px;padding:8px;margin-top:10px}
.radar svg{display:block;width:100%;height:190px}
.sliders{background:#1D3B34;border:1px solid var(--line);border-radius:3px;padding:13px 14px;
  margin-bottom:14px}
.sliders h4{font-family:var(--d);font-weight:700;font-size:12px;letter-spacing:.14em;
  text-transform:uppercase;margin:0 0 3px;color:var(--blaze)}
.sliders p{font-size:11px;color:var(--dim);margin:0 0 11px;line-height:1.5}
.sl{display:grid;grid-template-columns:82px 1fr 30px;gap:8px;align-items:center;
  font-size:11px;color:var(--dim);margin-bottom:5px}
.sl input{-webkit-appearance:none;appearance:none;height:4px;background:#0F2721;
  border:0;border-radius:2px;padding:0}
.sl input::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;
  border-radius:50%;background:var(--blaze);cursor:pointer}
.sl input::-moz-range-thumb{width:14px;height:14px;border:0;border-radius:50%;
  background:var(--blaze);cursor:pointer}
.sl b{font-family:var(--m);color:var(--ink);font-weight:500;text-align:right}
.slfoot{display:flex;gap:7px;margin-top:10px}
.slfoot button{flex:1;background:transparent;border:1px solid var(--line);color:var(--chalk);
  border-radius:3px;padding:7px;font-size:11px;cursor:pointer}
.slfoot button:hover{border-color:var(--blaze);color:#fff}
.replan{width:100%;background:var(--blaze);color:#1A0B04;border:0;border-radius:3px;
  padding:10px;margin-top:9px;font-family:var(--d);font-weight:700;font-size:13px;
  letter-spacing:.1em;text-transform:uppercase;cursor:pointer}
.replan:disabled{opacity:.5;cursor:not-allowed}
.navwrap{margin-top:10px}
.nav2{display:block;text-align:center;background:transparent;border:1px solid var(--line);
  color:var(--chalk);text-decoration:none;border-radius:3px;padding:7px;margin-top:6px;
  font-size:11.5px}
.nav2:hover{border-color:var(--chalk);color:#fff}
.hotspot{font-size:10.5px;color:var(--chalk);margin:-6px 0 12px}
.reorder{color:var(--blaze);font-size:11px;margin-top:8px;display:none}
.reorder.on{display:block}
.play{display:flex;gap:7px;margin-top:9px}
.play button{flex:1;background:#2A5348;border:1px solid var(--blaze);color:#FFD9C6;
  border-radius:3px;padding:8px;font-size:12px;cursor:pointer}
.play button:hover{background:var(--blaze);color:#1A0B04}
.live{font-family:var(--m);font-size:11px;color:var(--chalk);margin-top:6px;display:none}
.live.on{display:block}
.err{background:#3A1C10;border-left:3px solid var(--blaze);padding:12px 14px;font-size:13px;
  line-height:1.55}
.spin{display:inline-block;width:13px;height:13px;border:2px solid rgba(255,255,255,.25);
  border-top-color:#1A0B04;border-radius:50%;animation:sp .7s linear infinite;
  vertical-align:-2px;margin-right:7px}
@keyframes sp{to{transform:rotate(360deg)}}
@media(prefers-reduced-motion:reduce){.spin{animation:none}}
@media(max-width:900px){#app{grid-template-columns:1fr;height:auto}#map{height:60vh}}
</style>
</head>
<body>

<div id="app">
<aside>
  <h1>Route<span>·</span>Planner</h1>
  <p class="sub">Loops scored on nine criteria, not just distance</p>

  <div class="row">
    <label>Start point</label>
    <div class="pair">
      <input id="lat" placeholder="latitude" value="37.6624">
      <input id="lon" placeholder="longitude" value="-121.8747">
    </div>
    <button class="ghost" id="locate">Use my location</button>
  </div>

  <div class="row">
    <label for="prompt">What do you want to do</label>
    <textarea id="prompt">I want to run about 5 km, as safe as possible.</textarea>
    <div class="chips">
      <button class="chip">5k easy run</button>
      <button class="chip">3 mile walk, as flat as possible</button>
      <button class="chip">10 km hike, hilly and green</button>
      <button class="chip">15 km bike ride, quiet roads</button>
    </div>
  </div>

  <button class="primary" id="go">Find best route</button>

  <div id="out"></div>
  <hr>
  <div id="criteria" class="note"></div>
</aside>
<div id="map"></div>
</div>

<script>
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

/* ── map adapter ───────────────────────────────────────────────────────────
   Google Maps when a key is configured, Leaflet otherwise. Both expose the same
   four methods, so nothing downstream cares which one is running. A missing or
   rejected key falls back rather than leaving a blank rectangle. */
let MAP = null, layers = [], startMarker = null, DATA = null, LABELS = {};
const HOME = [37.6624, -121.8747];

function makeLeaflet() {
  const m = L.map('map', {zoomControl:true}).setView(HOME, 14);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'}).addTo(m);
  m.on('click', e => onMapClick(e.latlng.lat, e.latlng.lng));
  return {
    engine: 'leaflet',
    setStart(lat, lon) {
      if (startMarker) m.removeLayer(startMarker);
      startMarker = L.circleMarker([lat, lon], {radius:7, color:'#E8622C', weight:3,
        fillColor:'#0F1F1B', fillOpacity:1}).addTo(m).bindTooltip('Start');
    },
    clear() { layers.forEach(l => m.removeLayer(l)); layers = []; },
    line(coords, on, i) {
      const l = L.polyline(coords.map(c => [c[1], c[0]]), {
        color: on ? '#E8622C' : '#9EC4D2', weight: on ? 6 : 3,
        opacity: on ? 1 : .45, lineJoin:'round'}).addTo(m);
      if (!on) l.on('click', () => select(i));
      layers.push(l);
      return l;
    },
    lineColor(coords, color, weight) {
      const l = L.polyline(coords.map(c => [c[1], c[0]]),
        {color, weight, opacity:1, lineJoin:'round'}).addTo(m);
      layers.push(l);
      return l;
    },
    runner(lat, lon) {
      if (!this._run) {
        this._run = L.circleMarker([lat, lon], {radius:9, color:'#fff', weight:3,
          fillColor:'#E8622C', fillOpacity:1}).addTo(m);
      } else this._run.setLatLng([lat, lon]);
      this._run.bringToFront();
    },
    clearRunner() { if (this._run) { m.removeLayer(this._run); this._run = null; } },
    panTo(lat, lon) { m.panTo([lat, lon], {animate:true, duration:.3}); },
    focus(idx) {
      const all = layers.filter(Boolean);
      if (!all.length) return;
      let b = null;
      all.forEach(l => { b = b ? b.extend(l.getBounds()) : l.getBounds(); });
      if (b) m.fitBounds(b, {padding:[35,35]});
    }
  };
}

function makeGoogle() {
  const g = new google.maps.Map(document.getElementById('map'), {
    center: {lat: HOME[0], lng: HOME[1]}, zoom: 14, mapTypeControl: true,
    streetViewControl: true, fullscreenControl: true,
    mapTypeId: google.maps.MapTypeId.ROADMAP
  });
  g.addListener('click', e => onMapClick(e.latLng.lat(), e.latLng.lng()));
  return {
    engine: 'google',
    setStart(lat, lon) {
      if (startMarker) startMarker.setMap(null);
      startMarker = new google.maps.Marker({position:{lat, lng:lon}, map:g, title:'Start',
        icon:{path: google.maps.SymbolPath.CIRCLE, scale:8, fillColor:'#0F1F1B',
              fillOpacity:1, strokeColor:'#E8622C', strokeWeight:3}});
    },
    clear() { layers.forEach(l => l.setMap(null)); layers = []; },
    line(coords, on, i) {
      const path = coords.map(c => ({lat: c[1], lng: c[0]}));
      const l = new google.maps.Polyline({path, map:g, geodesic:false,
        strokeColor: on ? '#E8622C' : '#5A8DA6', strokeWeight: on ? 6 : 3,
        strokeOpacity: on ? 1 : .6, zIndex: on ? 10 : 1});
      if (!on) l.addListener('click', () => select(i));
      layers.push(l);
      return l;
    },
    lineColor(coords, color, weight) {
      const l = new google.maps.Polyline({path: coords.map(c => ({lat:c[1], lng:c[0]})),
        map:g, strokeColor:color, strokeWeight:weight, strokeOpacity:1, zIndex:15});
      layers.push(l);
      return l;
    },
    runner(lat, lon) {
      if (!this._run) {
        this._run = new google.maps.Marker({position:{lat, lng:lon}, map:g, zIndex:99,
          icon:{path: google.maps.SymbolPath.CIRCLE, scale:9, fillColor:'#E8622C',
                fillOpacity:1, strokeColor:'#fff', strokeWeight:3}});
      } else this._run.setPosition({lat, lng:lon});
    },
    clearRunner() { if (this._run) { this._run.setMap(null); this._run = null; } },
    panTo(lat, lon) { g.panTo({lat, lng:lon}); },
    focus() {
      if (!layers.length) return;
      const b = new google.maps.LatLngBounds();
      layers.forEach(l => l.getPath().forEach(pt => b.extend(pt)));
      g.fitBounds(b, 40);
    }
  };
}

function onMapClick(lat, lon) {
  document.getElementById('lat').value = lat.toFixed(5);
  document.getElementById('lon').value = lon.toFixed(5);
  MAP.setStart(lat, lon);
}

function startLeaflet(reason) {
  const css = document.createElement('link');
  css.rel = 'stylesheet';
  css.href = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';
  document.head.appendChild(css);
  const js = document.createElement('script');
  js.src = 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';
  js.onload = () => { MAP = makeLeaflet(); MAP.setStart(...HOME); if (DATA) draw(0); };
  document.head.appendChild(js);
  if (reason) console.warn('Falling back to OpenStreetMap:', reason);
}

window.gmapsReady = () => { MAP = makeGoogle(); MAP.setStart(...HOME); if (DATA) draw(0); };
window.gm_authFailure = () => startLeaflet('Google Maps rejected the key');

fetch('/api/config').then(r => r.json()).then(c => {
  if (!c.google_maps_key) return startLeaflet(null);
  const t = setTimeout(() => { if (!MAP) startLeaflet('Google Maps did not load'); }, 6000);
  const sc = document.createElement('script');
  sc.src = 'https://maps.googleapis.com/maps/api/js?key=' +
           encodeURIComponent(c.google_maps_key) + '&callback=gmapsReady&loading=async';
  sc.async = true;
  sc.onload = () => clearTimeout(t);
  sc.onerror = () => { clearTimeout(t); startLeaflet('Google Maps script blocked'); };
  document.head.appendChild(sc);
}).catch(() => startLeaflet('config unavailable'));

fetch('/api/criteria').then(r => r.json()).then(c => {
  LABELS = Object.fromEntries(c.criteria.map(x => [x.key, x.label]));
  $('#criteria').innerHTML = '<b style="color:var(--chalk)">What the score measures</b><br>' +
    c.criteria.map(x => `${esc(x.label)} — ${esc(x.source)}`).join('<br>');
}).catch(()=>{});

$('#locate').addEventListener('click', () => {
  if (!navigator.geolocation) return alert('This browser has no geolocation.');
  $('#locate').textContent = 'Locating…';
  navigator.geolocation.getCurrentPosition(p => {
    $('#lat').value = p.coords.latitude.toFixed(5);
    $('#lon').value = p.coords.longitude.toFixed(5);
    placeStart(p.coords.latitude, p.coords.longitude);
    $('#locate').textContent = 'Use my location';
  }, e => {
    $('#locate').textContent = 'Use my location';
    alert('Could not get your location: ' + e.message);
  }, {enableHighAccuracy:true, timeout:10000});
});

document.querySelectorAll('.chip').forEach(b =>
  b.addEventListener('click', () => { $('#prompt').value = b.textContent; }));

function placeStart(lat, lon) { if (MAP) MAP.setStart(lat, lon); }

function draw(selected) {
  if (!MAP || !DATA) return;
  SELECTED = selected;
  MAP.clear();
  // unselected loops stay faint underneath so you can see the alternatives
  DATA.routes.forEach((r, i) => {
    if (i !== selected) MAP.line(r.geojson.geometry.coordinates, false, i);
  });
  paintRoute(DATA.routes[selected], selected);
  MAP.focus(selected);
}

function renderCands() {
  const d = DATA;
  document.getElementById('cands').innerHTML = d.routes.map((r, i) => `
    <button class="cand ${i === SELECTED ? 'on' : ''}" data-i="${i}">
      <div class="top"><span class="bib">${r.score.toFixed(0)}</span>
        ${r.delta ? `<span style="font-family:var(--m);font-size:11px;color:${
          r.delta > 0 ? 'var(--good)' : 'var(--warn)'}">${
          r.delta > 0 ? '+' : ''}${r.delta}</span>` : ''}
        <span><b style="font-family:var(--d);font-size:17px">${r.distance_mi} mi</b>
          <span style="color:var(--dim);font-size:12px"> · ${r.distance_km} km</span>
          <div class="facts">${r.estimated_minutes} min · <b>${r.elevation_gain_m} m</b> climb
            · <b>${r.turns}</b> turns · AQI <b>${r.aqi}</b></div></span></div>
      <div class="bar"><i style="width:${r.score}%"></i></div>
      <div class="breakdown">
        ${Object.keys(r.scores).map(k => critRow(k, r.scores[k], r.weights[k],
            r.estimated[k])).join('')}
        ${radar(r, d.routes[i === 0 ? 1 : 0])}
        ${elevSvg(r)}
        <div class="play">
          <button data-fly="${i}">Fly the route</button>
          <button data-gpx="${i}">Download GPX</button></div>
        <div class="live" id="live"></div>
        <div class="navwrap">
          ${r.google_maps_url ? `<a class="nav" href="${esc(r.google_maps_url)}"
            target="_blank" rel="noopener">Navigate this route</a>` : ''}
          ${r.graphhopper_url ? `<a class="nav2" href="${esc(r.graphhopper_url)}"
            target="_blank" rel="noopener">Open the exact loop on a full map</a>` : ''}
        </div>
      </div>
    </button>`).join('');

  document.querySelectorAll('.cand').forEach(el =>
    el.addEventListener('click', e => {
      if (e.target.closest('.nav') || e.target.closest('.nav2') ||
          e.target.closest('.gpx') || e.target.closest('[data-fly]')) return;
      select(+el.dataset.i);
    }));
  document.querySelectorAll('[data-gpx]').forEach(b =>
    b.addEventListener('click', e => {
      e.stopPropagation();
      toGPX(DATA.routes[+b.dataset.gpx], +b.dataset.gpx);
    }));
  document.querySelectorAll('[data-fly]').forEach(b =>
    b.addEventListener('click', e => { e.stopPropagation(); fly(+b.dataset.fly); }));
}

function wireSliders() {
  const box = document.getElementById('slidebox');
  if (!box || !DATA) return;
  box.innerHTML = sliderPanel();
  box.querySelectorAll('[data-w]').forEach(inp =>
    inp.addEventListener('input', () => {
      USER_W = USER_W || {...DATA.weights.applied};
      USER_W[inp.dataset.w] = +inp.value / 100;
      box.querySelector(`[data-wv="${inp.dataset.w}"]`).textContent = inp.value;
      rescore();
    }));
  const rp = document.getElementById('replan');
  if (rp) rp.addEventListener('click', async () => {
    const w = USER_W || DATA.weights.applied;
    const base = DATA.weights.base.green ?? 0.1;
    // above the default greenery weight means seek parks, below means avoid them
    const bias = Math.max(-1, Math.min(1, ((w.green ?? base) - base) / 0.3));
    rp.disabled = true;
    rp.innerHTML = '<span class="spin"></span>Finding new routes';
    try {
      const res = await fetch('/api/plan-route', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({lat: parseFloat($('#lat').value),
                              lon: parseFloat($('#lon').value),
                              prompt: $('#prompt').value, green_bias: bias})});
      const d = await res.json();
      if (d.error) { alert(d.error); }
      else { const keep = USER_W; render(d); USER_W = keep; wireSliders(); rescore(); }
    } catch (e) { alert('Could not fetch new routes: ' + e.message); }
    finally { rp.disabled = false; rp.textContent = 'Find new routes for these weights'; }
  });
  box.querySelectorAll('[data-preset]').forEach(b =>
    b.addEventListener('click', () => {
      const p = b.dataset.preset;
      if (p === 'ai') USER_W = null;
      if (p === 'safe') USER_W = {...DATA.weights.applied, safety: .40, simplicity: .12};
      if (p === 'green') USER_W = {...DATA.weights.applied, green: .45, noise: .14};
      if (p === 'nogreen') USER_W = {...DATA.weights.applied, green: .00, distance: .35};
      wireSliders();
      rescore();
    }));
}

function select(i) {
  document.querySelectorAll('.cand').forEach((el, j) => el.classList.toggle('on', j === i));
  draw(i);
  refreshPaintBar();
}

function refreshPaintBar() {
  const bar = document.getElementById('paintbar');
  if (!bar || !DATA) return;
  bar.innerHTML = paintBar();
  bar.querySelectorAll('[data-paint]').forEach(b =>
    b.addEventListener('click', () => {
      PAINT_MODE = b.dataset.paint;
      draw(SELECTED);
      refreshPaintBar();
    }));
}

let BOOSTED = [];
function critRow(key, val, weight, estimated) {
  const cls = val >= 75 ? 'hi' : val < 45 ? 'lo' : '';
  const up = BOOSTED.includes(key) ? ' boosted' : '';
  return `<div class="crit"><span class="${estimated?'est':''}${up}">${esc(LABELS[key]||key)}${up?' \u2191':''}${estimated?' *':''}</span>
    <span class="track"><i class="${cls}" style="width:${Math.max(2,val)}%"></i></span>
    <span class="v">${Math.round(val)}</span></div>`;
}

/* ── paint the route by what the data says, segment by segment ──────────────
   OpenRouteService returns extra_info as [fromIndex, toIndex, value] spans along
   the geometry. Colouring each span turns the abstract score into something you
   can see on the map: where the greenery is, where the noise is, where it climbs. */
const PAINT = {
  route:     {label:'Route',      scale:null},
  green:     {label:'Greenery',   scale:v => v/10,        good:'high'},
  noise:     {label:'Noise',      scale:v => v/10,        good:'low'},
  steepness: {label:'Steepness',  scale:v => Math.min(1, Math.abs(v)/5), good:'low'},
  surface:   {label:'Surface',    scale:v => 1 - (SURF[v] ?? .5), good:'low'},
  waytype:   {label:'Path type',  scale:v => 1 - (WAY[v] ?? .5),  good:'low'},
};
const SURF = {1:.80,2:.95,3:.90,4:.85,5:.70,6:.60,7:.55,8:.45,9:.50,10:.40,11:.55,
              12:.35,13:.30,14:.25,15:.40,16:.65,17:.30,18:.20,20:.50};
const WAY  = {0:.55,1:.20,2:.40,3:.55,4:.90,5:.85,6:.88,7:.92,8:.60,9:.30,10:.15};
let PAINT_MODE = 'route';

function ramp(t, goodIsHigh) {
  const x = Math.max(0, Math.min(1, goodIsHigh ? t : 1 - t));   // 1 = good
  const stops = [[0,'#C7482A'], [0.5,'#E8B84B'], [1,'#4FA96E']];
  let a = stops[0], b = stops[2];
  for (let i = 0; i < stops.length - 1; i++)
    if (x >= stops[i][0] && x <= stops[i+1][0]) { a = stops[i]; b = stops[i+1]; }
  const f = (b[0] - a[0]) ? (x - a[0]) / (b[0] - a[0]) : 0;
  const hex = h => [1,3,5].map(i => parseInt(h.slice(i, i+2), 16));
  const [r1,g1,b1] = hex(a[1]), [r2,g2,b2] = hex(b[1]);
  const m = (p,q) => Math.round(p + (q - p) * f);
  return `rgb(${m(r1,r2)},${m(g1,g2)},${m(b1,b2)})`;
}

function paintRoute(r, idx) {
  const spans = (r.extras || {})[PAINT_MODE];
  const coords = r.geojson.geometry.coordinates;
  if (PAINT_MODE === 'route' || !spans || !spans.length) {
    MAP.line(coords, true, idx);
    return false;
  }
  const cfg = PAINT[PAINT_MODE];
  spans.forEach(([a, b, v]) => {
    const seg = coords.slice(a, Math.min(b + 1, coords.length));
    if (seg.length < 2) return;
    MAP.lineColor(seg, ramp(cfg.scale(v), cfg.good === 'high'), 6);
  });
  return true;
}

function paintBar() {
  const r = DATA.routes[SELECTED] || {};
  const avail = Object.keys(PAINT).filter(k =>
    k === 'route' || ((r.extras || {})[k] || []).length);
  const legend = PAINT_MODE === 'route' ? '' :
    `<div class="legend"><span><i style="background:#4FA96E"></i>better</span>
     <span><i style="background:#E8B84B"></i>mixed</span>
     <span><i style="background:#C7482A"></i>worse</span>
     <span style="margin-left:auto">every segment coloured from the routing data</span></div>`;
  return `<label>Paint the map by</label><div class="paint">${
    avail.map(k => `<button data-paint="${k}" aria-pressed="${k===PAINT_MODE}">${
      PAINT[k].label}</button>`).join('')}</div>${legend}`;
}

function elevSvg(r) {
  const pts = (r.elevation || {}).points || [];
  if (pts.length < 3) return '';
  const lo = Math.min(...pts), hi = Math.max(...pts), span = Math.max(1, hi - lo);
  const d = pts.map((v, i) =>
    `${(i / (pts.length - 1)) * 100},${100 - ((v - lo) / span) * 100}`).join(' ');
  return `<div class="elev">
    <svg viewBox="0 0 100 100" preserveAspectRatio="none">
      <polygon points="0,100 ${d} 100,100" fill="#E8622C" opacity=".22"/>
      <polyline points="${d}" fill="none" stroke="#E8622C" stroke-width="1.6"
        vector-effect="non-scaling-stroke"/></svg>
    <div class="cap"><span>${Math.round(lo)} m</span>
      <span>+${r.elevation_gain_m} m climb · ${r.climb_per_km} m/km</span>
      <span>${Math.round(hi)} m</span></div></div>`;
}

function toGPX(r, i) {
  const pts = r.geojson.geometry.coordinates.map(c =>
    `<trkpt lat="${c[1].toFixed(6)}" lon="${c[0].toFixed(6)}">${
      c.length > 2 ? `<ele>${c[2].toFixed(1)}</ele>` : ''}</trkpt>`).join('');
  const gpx = `<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Route Planner" xmlns="http://www.topografix.com/GPX/1/1">
<metadata><name>${r.distance_mi} mi loop, score ${r.score}</name></metadata>
<trk><name>Route ${i + 1}</name><trkseg>${pts}</trkseg></trk></gpx>`;
  const url = URL.createObjectURL(new Blob([gpx], {type:'application/gpx+xml'}));
  const a = document.createElement('a');
  a.href = url;
  a.download = `route-${r.distance_mi}mi-score${Math.round(r.score)}.gpx`;
  a.click();
  URL.revokeObjectURL(url);
}

let SELECTED = 0, USER_W = null, ANIM = null;

/* ── radar: all nine criteria at a glance ──────────────────────────────────
   A bar chart shows nine numbers. A radar shows the SHAPE of a route — you can
   see instantly that one is fast but filthy and another is slow but clean. */
function radar(r, rival) {
  const keys = Object.keys(r.scores), n = keys.length, R = 66, cx = 100, cy = 88;
  const pt = (i, v) => {
    const a = (Math.PI * 2 * i) / n - Math.PI / 2;
    return [cx + Math.cos(a) * R * (v / 100), cy + Math.sin(a) * R * (v / 100)];
  };
  const poly = o => keys.map((k, i) => pt(i, o.scores[k]).map(x => x.toFixed(1)).join(',')).join(' ');
  const rings = [25, 50, 75, 100].map(p =>
    `<polygon points="${keys.map((_, i) => pt(i, p).map(x => x.toFixed(1)).join(',')).join(' ')}"
      fill="none" stroke="#2C544A" stroke-width=".7"/>`).join('');
  const spokes = keys.map((k, i) => {
    const [x, y] = pt(i, 100);
    const [lx, ly] = pt(i, 128);
    return `<line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}"
      stroke="#2C544A" stroke-width=".6"/>
      <text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" fill="#8FAFA4" font-size="6.5"
        text-anchor="middle" dominant-baseline="middle">${
        (LABELS[k] || k).split(' ')[0]}</text>`;
  }).join('');
  const rivalPoly = rival ?
    `<polygon points="${poly(rival)}" fill="none" stroke="#5A8DA6" stroke-width="1.2"
      stroke-dasharray="3 2"/>` : '';
  return `<div class="radar"><svg viewBox="0 0 200 176">
    ${rings}${spokes}${rivalPoly}
    <polygon points="${poly(r)}" fill="#E8622C" fill-opacity=".28" stroke="#E8622C"
      stroke-width="1.8"/></svg>
    <div style="font-size:10px;color:var(--dim);text-align:center;margin-top:-4px">
      solid = this route${rival ? ' · dashed = next best' : ''}</div></div>`;
}

/* ── weight sliders: hand the judges the controls ──────────────────────────
   All nine sub-scores are already computed, so re-ranking is instant and local.
   Drag safety up and watch the winner change in front of you. */
function sliderPanel() {
  const w = USER_W || DATA.weights.applied;
  return `<div class="sliders">
    <h4>Tune it yourself</h4>
    <p>These are the weights behind the ranking. Move one and everything re-ranks
       instantly — no request, no waiting. A criterion marked <em>tied</em> scores the
       same on every candidate, so moving it honestly cannot change anything.</p>
    ${Object.keys(w).map(k => {
      const sp = (DATA.spread || {})[k] ?? 99;
      const byDesign = (DATA.uniform_by_design || []).includes(k);
      const dead = sp < 0.5 && !byDesign;
      const note = byDesign
        ? ' <span style="font-size:9px">(same for all)</span>'
        : dead ? ' <span style="font-size:9px">(tied)</span>' : '';
      return `<div class="sl" title="${byDesign
        ? 'Measured once at your start point, so it is identical for every candidate'
        : dead
        ? 'Every candidate scores the same here, so this slider cannot change the ranking'
        : 'Candidates differ by ' + sp + ' points on this criterion'}">
      <span style="${dead ? 'opacity:.45' : ''}">${esc(LABELS[k] || k)}${note}</span>
      <input type="range" min="0" max="40" value="${Math.round(w[k] * 100)}"
        data-w="${k}" ${dead ? 'style="opacity:.4"' : ''}>
      <b data-wv="${k}">${(w[k] * 100).toFixed(0)}</b></div>`;
    }).join('')}
    <div class="slfoot">
      <button data-preset="ai">Reset</button>
      <button data-preset="safe">Max safety</button>
      <button data-preset="green">Max greenery</button>
      <button data-preset="nogreen">Avoid green</button>
    </div>
    <button class="replan" id="replan">Find new routes for these weights</button>
    <p style="font-size:10.5px;color:var(--dim);margin:7px 0 0;line-height:1.5">
      Moving a slider re-ranks the loops you already have. This asks for
      <em>new</em> loops aimed at, or away from, the nearest parks.</p>
    <div class="reorder" id="reorder">The ranking changed — a different loop now wins.</div>
  </div>`;
}

function rescore() {
  const raw = USER_W || DATA.weights.applied;
  const tot = Object.values(raw).reduce((a, b) => a + b, 0) || 1;
  const w = Object.fromEntries(Object.entries(raw).map(([k, v]) => [k, v / tot]));
  const before = DATA.routes.map(r => r.id).join(',');
  DATA.routes.forEach(r => {
    const prev = r.score;
    // rank on the normalised values, exactly as the server does, so a criterion
    // whose absolute range is small still carries its full weight
    const src = r.rel || r.scores;
    r.score = +Object.keys(w).reduce((s, k) => s + w[k] * (src[k] ?? 50), 0).toFixed(1);
    r.delta = +(r.score - prev).toFixed(1);
    r.weights = w;
  });
  DATA.routes.sort((a, b) => b.score - a.score);
  SELECTED = 0;
  renderCands();
  const after = DATA.routes.map(r => r.id).join(',');
  const el = document.getElementById('reorder');
  if (el) {
    const winnerChanged = before.split(',')[0] !== after.split(',')[0];
    el.textContent = winnerChanged
      ? 'The ranking changed — a different loop now wins.'
      : 'The ranking changed — the order below just shifted.';
    el.classList.toggle('on', before !== after);
  }
  select(0);
}

/* ── flythrough: run the route in front of them ──────────────────────────── */
function fly(i) {
  const r = DATA.routes[i], pts = r.geojson.geometry.coordinates;
  if (ANIM) { cancelAnimationFrame(ANIM.raf); ANIM = null; MAP.clearRunner(); }
  const live = document.getElementById('live');
  if (live) live.classList.add('on');
  const t0 = performance.now(), dur = 9000;
  const step = now => {
    const t = Math.min(1, (now - t0) / dur);
    const idx = Math.min(pts.length - 1, Math.floor(t * (pts.length - 1)));
    const c = pts[idx];
    MAP.runner(c[1], c[0]);
    if (idx % 12 === 0) MAP.panTo(c[1], c[0]);
    if (live) live.textContent =
      `${(r.distance_km * t).toFixed(2)} km of ${r.distance_km} km` +
      (c.length > 2 ? ` · ${c[2].toFixed(0)} m elevation` : '') +
      ` · ${Math.round(r.estimated_minutes * t)} min`;
    if (t < 1) ANIM = {raf: requestAnimationFrame(step)};
    else { ANIM = null; MAP.focus(i); }
  };
  ANIM = {raf: requestAnimationFrame(step)};
}

function render(d) {
  DATA = d;
  BOOSTED = (d.weights && d.weights.changed) || [];
  const w = d.weather || {}, a = d.air || {};
  const est = [];
  $('#out').innerHTML = `
    <div class="cond">
      <div><b>${a.aqi ?? '—'}</b><small>AQI</small></div>
      <div><b>${w.apparent_temperature ?? '—'}&deg;</b><small>feels like</small></div>
      <div><b>${d.request.target_mi}</b><small>target mi</small></div>
      <div><b>${esc(d.request.activity)}</b><small>activity</small></div>
    </div>
    <label>What I understood</label>
    <div class="prefs">${
      (d.request.emphasis_labels && d.request.emphasis_labels.length)
        ? d.request.emphasis_labels.map(x => `<span class="pref">${esc(x)}</span>`).join('')
        : '<span class="pref none">no preference stated — using the default balance for ' +
          esc(d.request.activity) + '</span>'}</div>
    ${d.daylight && d.daylight.finishes_in_dark
      ? `<div class="dark">Sunset is at <b>${esc(d.daylight.sunset)}</b> and this takes about
         <b>${d.daylight.minutes_needed} min</b>. You would be out in the dark for the last
         <b>${d.daylight.dark_minutes} minutes</b> — take a light, or pick a shorter loop.</div>`
      : ''}
    <div class="summary"><div class="tagline">Recommendation</div>${esc(d.summary)}</div>
    ${(d.hotspots && d.hotspots.length) ? `<div class="hotspot">Nearest green:
      ${d.hotspots.slice(0,2).map(h =>
        `${h.features} features ${Math.round(h.distance_m)} m away`).join(' · ')}
      ${d.green_bias > 0.15 ? '— loops aimed toward it'
        : d.green_bias < -0.15 ? '— loops aimed away from it' : ''}</div>` : ''}
    <div id="paintbar"></div>
    <div id="slidebox"></div>
    <label>Candidate loops &mdash; ${d.routes.length} different directions</label>
    <div id="cands"></div>
    <p class="note">Air data: ${esc(a.source||'—')}. Greenery measured against
      ${d.green_features || 0} parks, woods and water features from OpenStreetMap. Distance parsed by ${esc(d.request.parsed_by)}.
      Scores are computed in Python from routing, elevation, air and weather data —
      the model reads your request and writes the summary, and never produces a number.
      An arrow marks a criterion you asked me to prioritise, which was weighted up before
      ranking. An asterisk marks a criterion OpenRouteService did not supply for this route,
      where a documented baseline was used instead.</p>`;

  renderCands();
  draw(0);
  refreshPaintBar();
  wireSliders();
  placeStart(...d.routes[0].geojson.geometry.coordinates[0].slice(0,2).reverse());
}

$('#go').addEventListener('click', async () => {
  const lat = parseFloat($('#lat').value), lon = parseFloat($('#lon').value);
  if (Number.isNaN(lat) || Number.isNaN(lon))
    return $('#out').innerHTML = '<div class="err">Enter a latitude and longitude, or click the map.</div>';
  $('#go').disabled = true;
  $('#go').innerHTML = '<span class="spin"></span>Planning';
  $('#out').innerHTML = '<p class="note">Reading your request, checking air quality and weather, requesting three loops…</p>';
  try {
    const res = await fetch('/api/plan-route', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({lat, lon, prompt: $('#prompt').value})});
    const d = await res.json();
    if (d.error) {
      $('#out').innerHTML = `<div class="err"><b>${esc(d.error)}</b>${
        d.detail ? '<div class="note">' + esc(d.detail.join(' · ')) + '</div>' : ''}</div>`;
    } else render(d);
  } catch (e) {
    $('#out').innerHTML = `<div class="err">Request failed: ${esc(e.message)}</div>`;
  } finally {
    $('#go').disabled = false;
    $('#go').textContent = 'Find best route';
  }
});

</script>
</body>
</html>
"""

