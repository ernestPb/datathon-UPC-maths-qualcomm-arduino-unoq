import json
import sqlite3
import sys
from pathlib import Path

# Carpeta on hi ha aquest fitxer: així el programa funciona des de
# qualsevol directori de la placa (ssh, adb shell, tmux...).
HERE = Path(__file__).resolve().parent
DB_CANDIDATES = [
    HERE / "data" / "trainer.db",
    HERE.parent / "data" / "trainer.db",
    Path.cwd() / "data" / "trainer.db",
]


def open_db():
    for path in DB_CANDIDATES:
        if path.is_file():
            db = sqlite3.connect(path)
            db.row_factory = sqlite3.Row
            return db
    print("No trobo la base de dades data/trainer.db.")
    print("Busco a:")
    for path in DB_CANDIDATES:
        print(f"  - {path}")
    return None


def ask(prompt):
    """input() que sempre retorna text net i surt net amb Ctrl+C / Ctrl+D."""
    try:
        return input(prompt).strip()
    except EOFError:
        raise SystemExit(
            "\nNo puc llegir del teclat. Executa el programa des d'un "
            "terminal de la placa (SSH o `adb shell`), no des d'App Lab."
        )
    except KeyboardInterrupt:
        raise SystemExit("\nSessió interrompuda.")


def choose_problem(db, student, topic, target_band, previous_id=None):
    # Busca problemes del mateix tema que l'alumne no ha resolt.
    # Prioritza la banda desitjada i després els veïns similars.
    query = """
        SELECT p.id, p.topic, p.band, p.label,
               pt.text, pt.options,
               COALESCE(n.sim, -1) AS similarity,
               CASE WHEN n.neighbor_id IS NULL THEN 1 ELSE 0 END
                    AS not_neighbor
        FROM problems p
        JOIN problem_texts pt
          ON pt.id = p.id AND pt.language = 'cat'
        LEFT JOIN neighbors n
          ON n.id = ? AND n.neighbor_id = p.id
        WHERE p.topic = ?
          AND pt.has_options = 1
          AND NOT EXISTS (
              SELECT 1 FROM student_log sl
              WHERE sl.student = ? AND sl.problem_id = p.id
          )
        ORDER BY
          ABS(p.band - ?) ASC,
          not_neighbor ASC,
          similarity DESC,
          p.id ASC
        LIMIT 1
    """

    return db.execute(
        query, (previous_id, topic, student, target_band)
    ).fetchone()


def save_answer(db, student, problem, correct):
    db.execute("""
        INSERT INTO student_log
        (student, problem_id, correct, hints, seconds, ts)
        VALUES (?, ?, ?, 0, NULL, datetime('now'))
    """, (student, problem["id"], int(correct)))

    next_band = max(
        0, min(3, problem["band"] + (1 if correct else -1))
    )

    db.execute("""
        INSERT INTO student_profile
        (student, topic, est_band, seen, correct)
        VALUES (?, ?, ?, 1, ?)
        ON CONFLICT(student, topic)
        DO UPDATE SET
            est_band = excluded.est_band,
            seen = seen + 1,
            correct = correct + excluded.correct
    """, (
        student, problem["topic"], next_band, int(correct)
    ))

    db.commit()
    return next_band


def show_problem(problem):
    print("\n" + "=" * 55)
    print(f"Tema: {problem['topic']} | Banda: {problem['band']}")
    print("=" * 55)
    print(problem["text"])

    options = json.loads(problem["options"])
    for letter in "ABCDE":
        print(f"{letter}) {options.get(letter, 'No disponible')}")


def ask_student():
    name = ask("Nom de l'alumne (Enter = demo): ")
    return name or "demo"


def ask_topic(db):
    topics = db.execute(
        "SELECT topic, name FROM topics ORDER BY topic"
    ).fetchall()

    if not topics:
        print("La base de dades no té cap tema.")
        return None

    print("\nTEMES DISPONIBLES:")
    for t in topics:
        print(f"  {t['topic']}: {t['name']}")

    valid_topics = {t["topic"] for t in topics}

    while True:
        raw = ask("\nEscull el número del tema: ")
        try:
            topic = int(raw)
        except ValueError:
            topic = None
        if topic in valid_topics:
            return topic
        print("Tema no vàlid.")


def main():
    print("=== THE ADAPTIVE TRAINER ===\n")

    db = open_db()
    if db is None:
        return 1

    try:
        student = ask_student()
        topic = ask_topic(db)
        if topic is None:
            return 1

        profile = db.execute("""
            SELECT est_band FROM student_profile
            WHERE student = ? AND topic = ?
        """, (student, topic)).fetchone()

        if profile:
            print(f"\nHola de nou, {student}! Continues al tema {topic}.")
        else:
            print(f"\nHola, {student}! Comencem el tema {topic}.")

        band = round(profile["est_band"]) if profile else 1
        previous_id = None

        while True:
            problem = choose_problem(
                db, student, topic, band, previous_id
            )

            if problem is None:
                print("No queden problemes disponibles d'aquest tema.")
                break

            show_problem(problem)

            answer = ask(
                "\nResposta (A-E) o Q per sortir: "
            ).upper()

            if answer == "Q":
                break

            # len == 1: evita que "" o "AB" comptin com a resposta
            if len(answer) != 1 or answer not in "ABCDE":
                print("Resposta no vàlida. Torna a intentar-ho.")
                continue

            correct = answer == problem["label"]

            if correct:
                print("CORRECTE!")
            else:
                print("INCORRECTE!")
                print(f"Resposta correcta: {problem['label']}")

            band = save_answer(db, student, problem, correct)
            previous_id = problem["id"]

            print(f"Nova banda objectiu: {band}")

        print("\nSessió finalitzada.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())