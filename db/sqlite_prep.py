import sqlite3


conn = sqlite3.connect("db/brandbastion.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS users;")
cursor.execute("DROP TABLE IF EXISTS comments;")


cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    comment_text TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    emotion TEXT NOT NULL,
    language TEXT NOT NULL,
    categories TEXT,  
    tags TEXT,       
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_id INTEGER NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
""")


conn.commit()
conn.close()


