
import sqlite3
from pathlib import Path

# La base sera stockée dans le dossier data du projet.
DB_PATH = Path(__file__).resolve().parent / "data" / "chat_history.db"


def get_connection():
    """Ouvre une connexion SQLite."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    return conn


def init_db():
    """Crée les tables si elles n'existent pas."""

    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'Nouvelle conversation',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL
                    CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (session_id)
                    REFERENCES sessions(session_id)
                    ON DELETE CASCADE
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id)
        """)

def create_session(session_id, title="Nouvelle conversation"):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO sessions (session_id, title)
            VALUES (?, ?)
            """,
            (session_id, title)
        )


def save_message(session_id, role, content):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO messages (session_id, role, content)
            VALUES (?, ?, ?)
            """,
            (session_id, role, content)
        )


def get_history(session_id):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY message_id ASC
            """,
            (session_id,)
        ).fetchall()

        return [dict(row) for row in rows]

def list_sessions():
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                s.session_id,
                s.title,
                s.created_at,
                (
                    SELECT content
                    FROM messages m
                    WHERE m.session_id = s.session_id
                      AND m.role = 'user'
                    ORDER BY m.message_id ASC
                    LIMIT 1
                ) AS first_question
            FROM sessions s
            ORDER BY s.created_at DESC
            """
        ).fetchall()

        return [dict(row) for row in rows]


if __name__ == "__main__":
    init_db()
    print(f"Base SQLite créée ou vérifiée : {DB_PATH}")
