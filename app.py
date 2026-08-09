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


def s_green(extras: dict) -> tuple[float, bool]:
    """ORS 'green' index: 0 (no vegetation) to 10 (dense). Real, not a constant."""
    block = (extras or {}).get("green")
    if not block or not block.get("values"):
        return 85.0, False          # brief's baseline, flagged as estimated
    num = den = 0.0
    for a, b, val in block["values"]:
        span = max(1, b - a)
        num += (int(val) / 10.0) * span
        den += span
    return 100.0 * (num / den), True


def s_noise(extras: dict) -> tuple[float, bool]:
    """ORS 'noise' index: 0 quiet to 10 loud. Inverted."""
    block = (extras or {}).get("noise")
    if not block or not block.get("values"):
        return 70.0, False
    num = den = 0.0
    for a, b, val in block["values"]:
        span = max(1, b - a)
        num += (1.0 - int(val) / 10.0) * span
        den += span
    return 100.0 * (num / den), True


def s_safety(extras: dict, steps: int, km: float) -> tuple[float, bool]:
    """Separated paths beat streets beat state roads; frequent junctions cost."""
    base = _extra_fraction(extras, "waytypes", WAYTYPE_SAFETY, 0)
    real = base is not None
    if base is None:
        base = 90.0                 # brief's baseline
    junction_rate = steps / max(0.4, km)          # manoeuvres per km
    penalty = min(25.0, max(0.0, (junction_rate - 4.0) * 3.0))
    return max(0.0, base - penalty), real


def s_surface(extras: dict) -> tuple[float, bool]:
    v = _extra_fraction(extras, "surface", SURFACE_Q, 0)
    return (v, True) if v is not None else (75.0, False)


def s_elevation(gain_m: float, km: float, activity: str) -> float:
    """Distance from the activity's ideal climb rate, both directions."""
    if km <= 0:
        return 50.0
    rate = gain_m / km
    ideal = IDEAL_CLIMB.get(activity, 12.0)
    return 100.0 * math.exp(-abs(rate - ideal) / (ideal * 1.6))


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


def fetch_routes(lat: float, lon: float, target_m: float, activity: str,
                 n: int = 3) -> list[dict]:
    """Round trips from OpenRouteService. Different seeds give genuinely different loops."""
    if not ORS_KEY:
        raise RuntimeError("ORS_API_KEY is not set")
    profile = ORS_PROFILE.get(activity, "foot-walking")
    out = []
    for i in range(n):
        body = {
            "coordinates": [[lon, lat]],
            "elevation": True,
            "instructions": True,
            "extra_info": ["surface", "waytype", "steepness", "green", "noise"],
            "options": {"round_trip": {"length": int(target_m), "points": 4 + i,
                                       "seed": 1 + i * 7}},
        }
        try:
            r = requests.post(ORS_URL.format(profile=profile), json=body,
                              headers={"Authorization": ORS_KEY,
                                       "Content-Type": "application/json"}, timeout=35)
            if r.status_code != 200:
                out.append({"error": f"ORS {r.status_code}: {r.text[:200]}"})
                continue
            out.append(r.json())
        except Exception as e:
            out.append({"error": str(e)})
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
                 '"activity": "run|walk|hike|cycle", "notes": "<any preference mentioned, '
                 'or null>"}. If no distance is stated use 5 and unit "km". Never invent '
                 "a location."},
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
        return {"target_m": max(400.0, min(50000.0, metres)), "activity": act,
                "notes": got.get("notes"), "parsed_by": PARSE_MODEL,
                "stated": f"{val} {unit}"}
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
    return {"target_m": max(400.0, min(50000.0, metres)), "activity": act,
            "notes": None, "parsed_by": "regex fallback", "stated": f"{val} {unit}"}


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
                air: dict, weather: dict, idx: int) -> dict | None:
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

    green, green_real = s_green(extras)
    noise, noise_real = s_noise(extras)
    safety, safety_real = s_safety(extras, steps, km)
    surface, surface_real = s_surface(extras)

    sc = {
        "distance": s_distance(dist_m, target_m),
        "air": s_air(air["aqi"]),
        "green": green,
        "noise": noise,
        "safety": safety,
        "elevation": s_elevation(ascent, km, activity),
        "surface": surface,
        "simplicity": s_simplicity(steps, km),
        "weather": s_weather(weather, activity),
    }
    w = WEIGHTS.get(activity, WEIGHTS["run"])
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
        "geojson": f,
    }


# ────────────────────────────────────────────────────────────── API

class PlanRequest(BaseModel):
    lat: float
    lon: float
    prompt: str = "I want to run 5 km."


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get("/api/health")
def health() -> dict[str, Any]:
    out = {"featherless_key": bool(FEATHERLESS_KEY), "ors_key": bool(ORS_KEY),
           "openaq_key": bool(OPENAQ_KEY), "models": {}}
    if FEATHERLESS_KEY:
        try:
            ids = {m.id for m in ai.models.list().data}
            out["models"] = {m: (m in ids) for m in {PARSE_MODEL, WRITE_MODEL}}
            out["featherless_catalog_size"] = len(ids)
        except Exception as e:
            out["models"] = {"error": str(e)[:200]}
    return out


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


@app.post("/api/plan-route")
def plan_route(req: PlanRequest):
    if not (-90 <= req.lat <= 90 and -180 <= req.lon <= 180):
        return {"error": "Those coordinates are not on Earth. Check latitude and longitude."}

    parsed = parse_request(req.prompt)
    air = fetch_air(req.lat, req.lon)
    weather = fetch_weather(req.lat, req.lon)

    try:
        raw = fetch_routes(req.lat, req.lon, parsed["target_m"], parsed["activity"])
    except RuntimeError as e:
        return {"error": str(e)}

    routes, errors = [], []
    for i, fc in enumerate(raw):
        if "error" in fc:
            errors.append(fc["error"])
            continue
        s = score_route(fc, parsed["target_m"], parsed["activity"], air, weather, i + 1)
        if s:
            routes.append(s)
    if not routes:
        return {"error": "OpenRouteService returned no usable loops here. Try a different "
                         "starting point or a shorter distance.",
                "detail": errors[:2]}

    routes.sort(key=lambda r: -r["score"])
    for rank, r in enumerate(routes, 1):
        r["rank"] = rank

    ctx = {**parsed, "air": air, "weather": weather}
    summary = write_summary(routes[0], ctx)

    return {
        "request": {"target_m": round(parsed["target_m"]),
                    "target_mi": round(parsed["target_m"] / 1609.34, 2),
                    "activity": parsed["activity"], "stated": parsed["stated"],
                    "notes": parsed.get("notes"), "parsed_by": parsed["parsed_by"]},
        "air": air,
        "weather": weather,
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
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
      integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
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
    <textarea id="prompt">I want to run about 5 miles somewhere green and quiet.</textarea>
    <div class="chips">
      <button class="chip">5k easy run</button>
      <button class="chip">3 mile walk, flat</button>
      <button class="chip">10 km hilly hike</button>
      <button class="chip">15 km bike ride</button>
    </div>
  </div>

  <button class="primary" id="go">Find best route</button>

  <div id="out"></div>
  <hr>
  <div id="criteria" class="note"></div>
</aside>
<div id="map"></div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
        integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
<script>
const $ = s => document.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

const map = L.map('map', {zoomControl:true}).setView([37.6624,-121.8747], 14);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

let layers = [], startMarker = null, DATA = null, LABELS = {};

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
    map.setView([p.coords.latitude, p.coords.longitude], 15);
    $('#locate').textContent = 'Use my location';
  }, e => {
    $('#locate').textContent = 'Use my location';
    alert('Could not get your location: ' + e.message);
  }, {enableHighAccuracy:true, timeout:10000});
});

document.querySelectorAll('.chip').forEach(b =>
  b.addEventListener('click', () => { $('#prompt').value = b.textContent; }));

map.on('click', e => {
  $('#lat').value = e.latlng.lat.toFixed(5);
  $('#lon').value = e.latlng.lng.toFixed(5);
  placeStart(e.latlng.lat, e.latlng.lng);
});

function placeStart(lat, lon) {
  if (startMarker) map.removeLayer(startMarker);
  startMarker = L.circleMarker([lat, lon], {radius:7, color:'#E8622C', weight:3,
    fillColor:'#0F1F1B', fillOpacity:1}).addTo(map).bindTooltip('Start');
}

function clearRoutes(){ layers.forEach(l => map.removeLayer(l)); layers = []; }

function draw(selected) {
  clearRoutes();
  DATA.routes.forEach((r, i) => {
    const on = i === selected;
    const l = L.geoJSON(r.geojson, {style:{
      color: on ? '#E8622C' : '#9EC4D2', weight: on ? 6 : 3,
      opacity: on ? 1 : 0.45, lineJoin:'round'}}).addTo(map);
    if (!on) l.on('click', () => select(i));
    layers.push(l);
  });
  const sel = layers[selected];
  if (sel) { sel.bringToFront(); map.fitBounds(sel.getBounds(), {padding:[35,35]}); }
}

function select(i) {
  document.querySelectorAll('.cand').forEach((el, j) => el.classList.toggle('on', j === i));
  draw(i);
}

function critRow(key, val, weight, estimated) {
  const cls = val >= 75 ? 'hi' : val < 45 ? 'lo' : '';
  return `<div class="crit"><span class="${estimated?'est':''}">${esc(LABELS[key]||key)}${estimated?' *':''}</span>
    <span class="track"><i class="${cls}" style="width:${Math.max(2,val)}%"></i></span>
    <span class="v">${Math.round(val)}</span></div>`;
}

function render(d) {
  DATA = d;
  const w = d.weather || {}, a = d.air || {};
  const est = [];
  $('#out').innerHTML = `
    <div class="cond">
      <div><b>${a.aqi ?? '—'}</b><small>AQI</small></div>
      <div><b>${w.apparent_temperature ?? '—'}&deg;</b><small>feels like</small></div>
      <div><b>${d.request.target_mi}</b><small>target mi</small></div>
      <div><b>${esc(d.request.activity)}</b><small>activity</small></div>
    </div>
    <div class="summary"><div class="tagline">Recommendation</div>${esc(d.summary)}</div>
    <label>Candidate loops</label>
    <div id="cands"></div>
    <p class="note">Air data: ${esc(a.source||'—')}. Distance parsed by ${esc(d.request.parsed_by)}.
      Scores are computed in Python from routing, elevation, air and weather data —
      the model reads your request and writes the summary, and never produces a number.
      An asterisk marks a criterion OpenRouteService did not supply for this route,
      where a documented baseline was used instead.</p>`;

  $('#cands').innerHTML = d.routes.map((r, i) => `
    <button class="cand ${i===0?'on':''}" data-i="${i}">
      <div class="top"><span class="bib">${r.score.toFixed(0)}</span>
        <span><b style="font-family:var(--d);font-size:17px">${r.distance_mi} mi</b>
          <span style="color:var(--dim);font-size:12px"> · ${r.distance_km} km</span>
          <div class="facts">${r.estimated_minutes} min · <b>${r.elevation_gain_m} m</b> climb
            · <b>${r.turns}</b> turns · AQI <b>${r.aqi}</b></div></span></div>
      <div class="bar"><i style="width:${r.score}%"></i></div>
      <div class="breakdown">
        ${Object.keys(r.scores).map(k => critRow(k, r.scores[k], r.weights[k],
            r.estimated[k])).join('')}
      </div>
    </button>`).join('');

  document.querySelectorAll('.cand').forEach(el =>
    el.addEventListener('click', () => select(+el.dataset.i)));

  draw(0);
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

placeStart(37.6624, -121.8747);
</script>
</body>
</html>
"""

