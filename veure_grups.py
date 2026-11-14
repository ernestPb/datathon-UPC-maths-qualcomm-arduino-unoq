import sqlite3

db = sqlite3.connect("data/trainer.db")

query = """
SELECT p.id, p.topic, p.band,
       SUBSTR(pt.text, 1, 100)
FROM problems p
JOIN problem_texts pt ON pt.id = p.id
WHERE pt.language = 'eng'
ORDER BY p.topic, p.id
"""

rows = db.execute(query).fetchall()

for topic in range(6):
    print(f"\n========== TOPIC {topic} ==========")
    shown = 0
    for pid, t, band, txt in rows:
        if t == topic:
            print(f"[Band {band}] {pid}: {txt}")
            shown += 1
            if shown == 8:
                break

db.close()
