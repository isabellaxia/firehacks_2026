"""The investigation loop.

The agent decides which tools to call and in what order. It cannot compute anything
itself — every quantity in its answer has to come back from a tool result.
"""
import json
import os
import time

from llm import _sem, client
from tools import REGISTRY, TOOL_SCHEMAS

AGENT_MODEL = os.environ.get("AGENT_MODEL", "Qwen/Qwen3-32B")
MAX_STEPS = int(os.environ.get("MAX_AGENT_STEPS", "12"))

SYSTEM = """You are a safety analyst investigating aviation near-miss reports.

If the question does not name an airport, start with scan_corpus — it sweeps every
airport and every failure mode at once and ranks what is rising. Then investigate the
top one or two properly using the method below.

Method, in order:
1. Look at what clusters exist at the airport (cluster_incidents).
2. Pick the one or two most concerning and measure them (compute_trend).
3. Always check whether it is local or national (compare_baseline). A cluster rising
   everywhere is a reporting artifact, not an airport risk. Say so if that's the case.
4. Read two or three underlying reports (search_reports, extract_causal_chain) so your
   explanation describes what crews actually wrote, not what you assume.
5. Check plausibility against the field's layout (get_airport_context).
6. If asked how early something was detectable, call lead_time — it walks the cutoff
   month backwards and reports the months of warning before a known incident.
7. Before answering, call verify_finding on your one-sentence conclusion.

Hard rules:
- Never state a count, rate, ratio, p-value or date that did not come back from a tool.
  If you need a number, call a tool for it.
- If the trend is not significant, say the pattern is not rising. A null result is a
  correct answer and you should give it plainly.
- Quote at most a short phrase from any report. Summarise instead.
- Finish with a short written finding: what is rising, how fast, why it is plausible,
  which reports show it, and what the verifier said.
"""


async def run(question: str, emit) -> None:
    """emit(event_type, payload) is awaited for each step so the UI can stream."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    t0 = time.time()

    for step in range(MAX_STEPS):
        async with _sem:
            resp = await client.chat.completions.create(
                model=AGENT_MODEL, messages=messages, tools=TOOL_SCHEMAS,
                tool_choice="auto", max_tokens=1200, temperature=0.3)

        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            await emit("answer", {"text": msg.content or "(no answer)",
                                  "steps": step, "seconds": round(time.time() - t0, 1)})
            return

        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            await emit("tool_call", {"step": step + 1, "name": name, "args": args})

            fn = REGISTRY.get(name)
            if fn is None:
                result = {"error": f"no such tool: {name}"}
            else:
                try:
                    result = await fn(**args)
                except TypeError as e:
                    result = {"error": f"bad arguments: {e}"}
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {e}"}

            await emit("tool_result", {"step": step + 1, "name": name, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result)[:12000]})

    await emit("answer", {"text": "Stopped after the maximum number of investigation steps "
                                  "without reaching a conclusion. Try a narrower question.",
                          "steps": MAX_STEPS, "seconds": round(time.time() - t0, 1)})

