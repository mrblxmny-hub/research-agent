import os
import json
import time
import re
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import google.generativeai as genai
from tavily import TavilyClient

app = Flask(__name__)
CORS(app)

# ── API KEYS ──────────────────────────────────────────────
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

# ── FILE-BASED PERSISTENCE ────────────────────────────────
DATA_FILE = "research_data.json"

def load_store():
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Load error: {e}")
    return []

def save_store(data):
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Save error: {e}")

# ── AGENT CONFIGS ─────────────────────────────────────────
AGENT_CONFIGS = {
    "Scout": {
        "color": "#4fc3f7",
        "emoji": "🔵",
        "role": "You are Scout, a research agent specialized in finding real problems people have with mobile and web apps in 2024-2026. Search for user complaints, bad reviews, Reddit threads, and pain points. Focus on: UX failures, missing features, performance issues, monetization frustrations. Be specific and cite real apps.",
        "search_queries": lambda topic: [
            f"{topic} app problems 2024 2025 user complaints",
            f"{topic} app Reddit complaints 2025",
        ]
    },
    "Analyst": {
        "color": "#69f0ae",
        "emoji": "📈",
        "role": "You are Analyst, a pattern-recognition agent. You look at app market trends, analyze what categories of apps are failing users, and identify systemic gaps. Focus on data, trends, and statistics from 2024-2026. Look for patterns across multiple apps in the same category.",
        "search_queries": lambda topic: [
            f"{topic} app market trends 2025 analysis",
            f"why {topic} apps fail 2025 statistics",
        ]
    },
    "Diver": {
        "color": "#ce93d8",
        "emoji": "🔮",
        "role": "You are Diver, a deep-research agent. You go deep on one specific app or problem, finding technical details, user stories, and nuanced issues that others miss. Look at app store reviews, developer forums, and niche communities. Focus on 2024-2026 timeframe.",
        "search_queries": lambda topic: [
            f"{topic} app store reviews 1 star complaints 2024 2025",
            f"{topic} developer community problems 2025",
        ]
    },
    "Critic": {
        "color": "#ff7043",
        "emoji": "⚡",
        "role": "You are Critic, a skeptical agent who challenges assumptions. You find counter-arguments, identify why proposed solutions might fail, and spot overlooked risks. For app ideas, you find why they might not work. Be harsh but fair.",
        "search_queries": lambda topic: [
            f"why {topic} app solutions fail 2025",
            f"{topic} app startup failure reasons 2024 2025",
        ]
    },
    "Memory": {
        "color": "#ffd54f",
        "emoji": "🧠",
        "role": "You are Memory, an agent that synthesizes and indexes research. You create connections between different findings, identify recurring themes, and build a knowledge graph of app problems and opportunities. Summarize and connect findings from 2024-2026.",
        "search_queries": lambda topic: [
            f"{topic} app ecosystem overview 2025",
            f"{topic} app opportunities gaps 2024 2025",
        ]
    },
    "Synth": {
        "color": "#f48fb1",
        "emoji": "✨",
        "role": "You are Synth, a creative synthesis agent. Based on real app problems and market gaps from 2024-2026, you generate concrete, actionable app ideas. For each idea include: the problem it solves, target users, key features, why now is the right time, and potential revenue model.",
        "search_queries": lambda topic: [
            f"app ideas solving {topic} problems 2025",
            f"untapped {topic} app market opportunity 2025",
        ]
    }
}


# ── WEB SEARCH ────────────────────────────────────────────
def search_web(queries, max_results=5):
    if not TAVILY_KEY:
        return [{"title": "Demo mode — add TAVILY_API_KEY", "content": "No search results in demo mode.", "url": "#"}]

    client = TavilyClient(api_key=TAVILY_KEY)
    all_results = []
    for query in queries[:2]:
        try:
            results = client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_answer=True
            )
            if results.get("results"):
                all_results.extend(results["results"])
            time.sleep(0.5)
        except Exception as e:
            print(f"Tavily error: {e}")
    return all_results[:8]


# ── SINGLE AGENT RUNNER ───────────────────────────────────
def run_agent(agent_name, topic):
    config = AGENT_CONFIGS.get(agent_name)
    if not config:
        return None

    queries = config["search_queries"](topic)
    search_results = search_web(queries)

    search_context = "\n\n".join([
        f"SOURCE: {r.get('title', 'Unknown')}\nURL: {r.get('url', '#')}\nCONTENT: {r.get('content', '')[:500]}"
        for r in search_results
    ])

    prompt = f"""{config['role']}

TOPIC: {topic}

SEARCH RESULTS FROM THE WEB:
{search_context}

Based on these real search results, provide a detailed research report about "{topic}" related to app problems and opportunities (2024-2026).

Return ONLY a valid JSON object with these exact fields (no markdown, no backticks):
{{
  "summary": "2-3 sentence overview of findings",
  "key_findings": ["finding 1", "finding 2", "finding 3", "finding 4", "finding 5"],
  "specific_apps": ["app name 1", "app name 2", "app name 3"],
  "pain_points": ["pain point 1", "pain point 2", "pain point 3"],
  "opportunities": ["opportunity 1", "opportunity 2"],
  "timeframe": "2024-2026",
  "confidence": "high",
  "tags": ["tag1", "tag2", "tag3"]
}}"""

    if not GEMINI_KEY:
        return _demo_result(agent_name, topic, config)

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        raw = response.text.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'^```\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw).strip()

        data = json.loads(raw)
        data["agent"] = agent_name
        data["topic"] = topic
        data["color"] = config["color"]
        data["emoji"] = config["emoji"]
        data["sources"] = [{"title": r.get("title", ""), "url": r.get("url", "#")} for r in search_results[:4]]
        data["timestamp"] = datetime.now().isoformat()
        data["id"] = f"{agent_name}_{int(time.time() * 1000)}"
        return data

    except json.JSONDecodeError as e:
        print(f"JSON parse error ({agent_name}): {e}")
        return _error_result(agent_name, topic, config)
    except Exception as e:
        print(f"Agent error ({agent_name}): {e}")
        return _error_result(agent_name, topic, config)


def _demo_result(agent_name, topic, config):
    return {
        "agent": agent_name, "topic": topic,
        "color": config["color"], "emoji": config["emoji"],
        "summary": f"[Demo] {agent_name} analysis of '{topic}'. Add GEMINI_API_KEY for real results.",
        "key_findings": ["Add GEMINI_API_KEY at aistudio.google.com", "Add TAVILY_API_KEY at tavily.com", "Both are free tiers"],
        "specific_apps": [], "pain_points": [], "opportunities": [],
        "timeframe": "2024-2026", "confidence": "low",
        "tags": ["demo", topic.lower()], "sources": [],
        "timestamp": datetime.now().isoformat(),
        "id": f"{agent_name}_{int(time.time() * 1000)}"
    }


def _error_result(agent_name, topic, config):
    return {
        "agent": agent_name, "topic": topic,
        "color": config["color"], "emoji": config["emoji"],
        "summary": f"{agent_name} collected data on '{topic}' but had a formatting issue. Try again.",
        "key_findings": ["Research collected but could not be parsed", "Try running the agent again"],
        "specific_apps": [], "pain_points": [], "opportunities": [],
        "timeframe": "2024-2026", "confidence": "low",
        "tags": [topic.lower()], "sources": [],
        "timestamp": datetime.now().isoformat(),
        "id": f"{agent_name}_{int(time.time() * 1000)}"
    }


# ── ROUTES ────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/research", methods=["POST"])
def run_research():
    """Run multiple agents — streams progress via SSE-like JSON lines."""
    body = request.get_json()
    topic = body.get("topic", "").strip()
    agents = body.get("agents", list(AGENT_CONFIGS.keys()))

    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    store = load_store()
    results = []

    for agent_name in agents:
        if agent_name not in AGENT_CONFIGS:
            continue
        result = run_agent(agent_name, topic)
        if result:
            store.append(result)
            results.append(result)
        time.sleep(0.8)

    save_store(store)
    return jsonify({"results": results, "total": len(results)})


@app.route("/api/research/single", methods=["POST"])
def run_single():
    body = request.get_json()
    topic = body.get("topic", "").strip()
    agent_name = body.get("agent", "Scout")

    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    result = run_agent(agent_name, topic)
    if result:
        store = load_store()
        store.append(result)
        save_store(store)
        return jsonify(result)

    return jsonify({"error": "Agent failed"}), 500


@app.route("/api/store", methods=["GET"])
def get_store():
    store = load_store()
    return jsonify({"results": store, "total": len(store)})


@app.route("/api/store", methods=["DELETE"])
def clear_store():
    save_store([])
    return jsonify({"message": "Store cleared"})


@app.route("/api/agents", methods=["GET"])
def get_agents():
    return jsonify({
        name: {"color": cfg["color"], "emoji": cfg["emoji"]}
        for name, cfg in AGENT_CONFIGS.items()
    })


@app.route("/api/health", methods=["GET"])
def health():
    store = load_store()
    return jsonify({
        "status": "ok",
        "gemini": bool(GEMINI_KEY),
        "tavily": bool(TAVILY_KEY),
        "nodes": len(store)
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
