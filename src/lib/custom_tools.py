from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///titanic.db")

@tool
def get_titanic_character_info(name: str) -> list[dict]:
    """Useful when you want to get information about the Titanic movie characters."""
    return db.run(
        "SELECT * FROM titanic WHERE name like :name", parameters={"name": f"%{name}%"}
    )
