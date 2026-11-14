"""Versió web de The Adaptive Trainer.

Només fa servir la biblioteca estàndard de Python (no cal pip) i reutilitza
la lògica d'adaptive_trainer.py. Ha d'estar a la mateixa carpeta.

    python3 web_trainer.py                 # http://10.91.127.216:8000
    python3 web_trainer.py --port 8001     # si el port 8000 està ocupat
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import adaptive_trainer as trainer

# IP de la placa a la xarxa. Si canvia, només cal modificar-la aquí.
BOARD_IP = "10.91.127.216"
DEFAULT_PORT = 8000


class ApiError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------- lògica

def clean_name(value):
    name = str(value or "").strip()[:40]
    return name or "demo"


def public_problem(problem):
    """El problema tal com el veu l'alumne: sense la resposta correcta."""
    if problem is None:
        return None
    options = json.loads(problem["options"])
    return {
        "id": problem["id"],
        "topic": problem["topic"],
        "band": problem["band"],
        "text": problem["text"],
        "options": {
            letter: options.get(letter, "No disponible") for letter in "ABCDE"
        },
    }


def get_topics(db):
    rows = db.execute(
        "SELECT topic, name FROM topics ORDER BY topic"
    ).fetchall()
    return [{"topic": r["topic"], "name": r["name"]} for r in rows]


def get_problem(db, problem_id):
    return db.execute("""
        SELECT p.id, p.topic, p.band, p.label, pt.text, pt.options
        FROM problems p
        JOIN problem_texts pt
          ON pt.id = p.id AND pt.language = 'cat'
        WHERE p.id = ?
    """, (problem_id,)).fetchone()


def api_start(db, body):
    student = clean_name(body.get("student"))

    try:
        topic = int(body.get("topic"))
    except (TypeError, ValueError):
        raise ApiError("Tema no vàlid.")
    if topic not in {t["topic"] for t in get_topics(db)}:
        raise ApiError("Tema no vàlid.")

    profile = db.execute("""
        SELECT est_band FROM student_profile
        WHERE student = ? AND topic = ?
    """, (student, topic)).fetchone()

    band = round(profile["est_band"]) if profile else 1
    problem = trainer.choose_problem(db, student, topic, band, None)

    return {
        "student": student,
        "topic": topic,
        "band": band,
        "returning": profile is not None,
        "problem": public_problem(problem),
    }


def api_answer(db, body):
    student = clean_name(body.get("student"))
    answer = str(body.get("answer", "")).strip().upper()

    if len(answer) != 1 or answer not in "ABCDE":
        raise ApiError("Resposta no vàlida.")

    problem = get_problem(db, body.get("problem_id"))
    if problem is None:
        raise ApiError("Aquest problema no existeix.", 404)

    already = db.execute(
        "SELECT 1 FROM student_log WHERE student = ? AND problem_id = ?",
        (student, problem["id"]),
    ).fetchone()
    if already:
        raise ApiError("Aquest problema ja està respost.", 409)

    correct = answer == problem["label"]
    band = trainer.save_answer(db, student, problem, correct)
    next_problem = trainer.choose_problem(
        db, student, problem["topic"], band, problem["id"]
    )

    return {
        "correct": correct,
        "label": problem["label"],
        "band": band,
        "problem": public_problem(next_problem),
    }


# ---------------------------------------------------------------- pàgina

PAGE = """<!doctype html>
<html lang="ca">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Adaptive Trainer</title>
<style>
  [hidden] { display: none !important; }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: #f4f6f8; color: #1d2733;
    font: 18px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  main { max-width: 640px; margin: 0 auto; padding: 24px 16px 48px; }
  h1 { font-size: 1.5rem; margin: 0 0 20px; }
  section {
    background: #fff; border-radius: 12px; padding: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
  }
  label { display: block; font-weight: 600; margin: 14px 0 6px; }
  input, select {
    width: 100%; padding: 10px 12px; font: inherit;
    border: 1px solid #b9c3cd; border-radius: 8px; background: #fff;
  }
  button {
    font: inherit; cursor: pointer; border: 0; border-radius: 8px;
    padding: 10px 18px; background: #0b6e8a; color: #fff;
  }
  button:hover:not(:disabled) { background: #095a72; }
  button:disabled { cursor: default; opacity: .85; }
  button.secondary { background: #e3e8ee; color: #1d2733; }
  .actions { display: flex; gap: 10px; margin-top: 20px; flex-wrap: wrap; }
  .meta { color: #5b6b7a; font-size: .85rem; margin-bottom: 12px; }
  #text { white-space: pre-wrap; margin: 0 0 16px; }
  #options { display: grid; gap: 8px; }
  #options button {
    text-align: left; background: #fff; color: #1d2733;
    border: 2px solid #c9d2db;
  }
  #options button:hover:not(:disabled) { background: #eef3f7; }
  #options button.right { background: #d9f2e0; border-color: #2e9e57; }
  #options button.wrong { background: #fadcdc; border-color: #c93c3c; }
  #feedback { margin-top: 14px; font-weight: 600; min-height: 1.5em; }
  #feedback.ok { color: #1f7a42; }
  #feedback.ko { color: #b32d2d; }
</style>
</head>
<body>
<main>
  <h1>The Adaptive Trainer</h1>

  <section id="start">
    <label for="student">Nom de l'alumne</label>
    <input id="student" maxlength="40" placeholder="demo" autocomplete="off">
    <label for="topic">Tema que vols practicar</label>
    <select id="topic"></select>
    <div class="actions"><button id="go">Comença</button></div>
    <p id="startError" class="meta"></p>
  </section>

  <section id="quiz" hidden>
    <div class="meta" id="meta"></div>
    <p id="text"></p>
    <div id="options"></div>
    <div id="feedback"></div>
    <div class="actions">
      <button id="next" hidden>Següent</button>
      <button id="quit" class="secondary">Sortir</button>
    </div>
  </section>

  <section id="end" hidden>
    <p id="endMsg"></p>
    <div class="actions"><button id="again">Tornar a començar</button></div>
  </section>
</main>

<script>
const $ = (id) => document.getElementById(id);
const state = { student: "demo", topic: null, current: null, pending: null };

async function api(path, body) {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Error del servidor");
  return data;
}

function show(id) {
  for (const s of ["start", "quiz", "end"]) $(s).hidden = s !== id;
}

function showProblem(problem, band) {
  state.current = problem;
  $("meta").textContent =
    "Alumne: " + state.student + " · Tema " + problem.topic +
    " · Banda objectiu: " + band;
  $("text").textContent = problem.text;
  $("feedback").textContent = "";
  $("feedback").className = "";
  $("next").hidden = true;

  const box = $("options");
  box.replaceChildren();
  for (const letter of "ABCDE") {
    const b = document.createElement("button");
    b.dataset.letter = letter;
    b.textContent = letter + ") " + problem.options[letter];
    b.addEventListener("click", () => answer(letter));
    box.appendChild(b);
  }
  show("quiz");
}

function finish(message) {
  $("endMsg").textContent = message;
  show("end");
}

async function answer(letter) {
  const buttons = [...$("options").querySelectorAll("button")];
  buttons.forEach((b) => (b.disabled = true));
  try {
    const res = await api("/api/answer", {
      student: state.student,
      problem_id: state.current.id,
      answer: letter,
    });
    for (const b of buttons) {
      if (b.dataset.letter === res.label) b.classList.add("right");
      else if (b.dataset.letter === letter) b.classList.add("wrong");
    }
    const fb = $("feedback");
    fb.className = res.correct ? "ok" : "ko";
    fb.textContent = res.correct
      ? "CORRECTE! Nova banda objectiu: " + res.band
      : "INCORRECTE! Resposta correcta: " + res.label +
        ". Nova banda objectiu: " + res.band;
    state.pending = { problem: res.problem, band: res.band };
    $("next").textContent = res.problem ? "Següent" : "Acabar";
    $("next").hidden = false;
  } catch (err) {
    $("feedback").className = "ko";
    $("feedback").textContent = err.message;
    buttons.forEach((b) => (b.disabled = false));
  }
}

$("next").addEventListener("click", () => {
  const p = state.pending;
  if (p && p.problem) showProblem(p.problem, p.band);
  else finish("No queden problemes disponibles d'aquest tema.");
});

$("quit").addEventListener("click", () => finish("Sessió finalitzada."));
$("again").addEventListener("click", () => show("start"));

$("go").addEventListener("click", async () => {
  $("startError").textContent = "";
  try {
    const res = await api("/api/start", {
      student: $("student").value,
      topic: $("topic").value,
    });
    state.student = res.student;
    state.topic = res.topic;
    if (res.problem) showProblem(res.problem, res.band);
    else finish("No queden problemes disponibles d'aquest tema.");
  } catch (err) {
    $("startError").textContent = err.message;
  }
});

(async () => {
  try {
    const topics = await api("/api/topics");
    for (const t of topics) {
      const o = document.createElement("option");
      o.value = t.topic;
      o.textContent = t.topic + ": " + t.name;
      $("topic").appendChild(o);
    }
  } catch (err) {
    $("startError").textContent = err.message;
  }
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------- servidor

class Handler(BaseHTTPRequestHandler):
    db = None  # es posa a main()

    def _send(self, status, content_type, payload):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, data, status=200):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send(status, "application/json; charset=utf-8", payload)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif self.path == "/api/topics":
            self._json(get_topics(self.db))
        else:
            self._json({"error": "No trobat"}, 404)

    def do_POST(self):
        routes = {"/api/start": api_start, "/api/answer": api_answer}
        handler = routes.get(self.path)
        if handler is None:
            self._json({"error": "No trobat"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 10_000:
                raise ApiError("Petició no vàlida.")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ApiError("Petició no vàlida.")
            self._json(handler(self.db, body))
        except ApiError as err:
            self._json({"error": str(err)}, err.status)
        except (ValueError, UnicodeDecodeError):
            self._json({"error": "Petició no vàlida."}, 400)


def main():
    parser = argparse.ArgumentParser(description="The Adaptive Trainer (web)")
    # 0.0.0.0 = escolta a totes les interfícies de la placa: serveix per a la
    # IP de la xarxa i també per a 127.0.0.1 (USB amb adb forward / ssh -L).
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    db = trainer.open_db()
    if db is None:
        return 1
    Handler.db = db

    # HTTPServer atén una petició cada cop, tot al fil principal:
    # així la connexió SQLite no es comparteix entre fils.
    server = HTTPServer((args.host, args.port), Handler)
    print("Servidor en marxa. Obre al navegador:")
    print(f"  http://{BOARD_IP}:{args.port}   (des de la mateixa xarxa)")
    print(f"  http://127.0.0.1:{args.port}    (des de la placa o per USB)")
    print("Ctrl+C per aturar-lo.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor aturat.")
    finally:
        server.server_close()
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
