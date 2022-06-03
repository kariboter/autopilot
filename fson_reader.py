import sqlite3

db = sqlite3.connect('server.db')
sql = db.cursor()

sql.execute("""CREAT TABLE IF NOT EXISTS users()""")
