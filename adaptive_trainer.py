
import sqlite3
import json

DB = "data/trainer.db"
db = sqlite3.connect(DB)
db.row_factory = sqlite3.Row


def choose_problem(student, topic, target_band, previous_id=None):
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


def save_answer(student, problem, correct):
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


def main():
    print("=== THE ADAPTIVE TRAINER ===")

    student = input("Nom de l'alumne: ").strip() or "demo"

    print("\nTEMES DISPONIBLES:")
    topics = db.execute(
        "SELECT topic, name FROM topics ORDER BY topic"
    ).fetchall()

    for t in topics:
        print(f"{t['topic']}: {t['name']}")

    valid_topics = {t["topic"] for t in topics}

    while True:
        try:
            topic = int(input("\nEscull el número del tema: "))
            if topic in valid_topics:
                break
        except ValueError:
            pass
        print("Tema no vàlid.")

    profile = db.execute("""
        SELECT est_band FROM student_profile
        WHERE student = ? AND topic = ?
    """, (student, topic)).fetchone()

    band = round(profile["est_band"]) if profile else 1
    previous_id = None

    while True:
        problem = choose_problem(
            student, topic, band, previous_id
        )

        if problem is None:
            print("No queden problemes disponibles d'aquest tema.")
            break

        show_problem(problem)

        answer = input(
            "\nResposta (A-E) o Q per sortir: "
        ).strip().upper()

        if answer == "Q":
            break

        if answer not in "ABCDE":
            print("Resposta no vàlida. Torna a intentar-ho.")
            continue

        correct = answer == problem["label"]

        if correct:
            print("CORRECTE!")
        else:
            print("INCORRECTE!")
            print(f"Resposta correcta: {problem['label']}")

        band = save_answer(student, problem, correct)
        previous_id = problem["id"]

        print(f"Nova banda objectiu: {band}")

    print("\nSessió finalitzada.")
    db.close()


if __name__ == "__main__":
    main()
