import sqlitecloud

CLOUD_DB_URL = "sqlitecloud://cb8aau8ivz.g3.sqlite.cloud:8860/auth.sqlitecloud?apikey=dab0wipt5QghDDG1ZU5JdX0wkhaFWvCCu5jOfwryfkw"

def get_connection():
    return sqlitecloud.connect(CLOUD_DB_URL)
