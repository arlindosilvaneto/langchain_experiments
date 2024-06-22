from typing import Annotated
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langchain_experimental.utilities import PythonREPL


db = SQLDatabase.from_uri("sqlite:///titanic.db")
repl = PythonREPL()

@tool
def get_titanic_character_info(name: str) -> list[dict]:
    """Useful when you want to get information about the Titanic movie characters."""
    return db.run(
        "SELECT * FROM titanic WHERE name like :name", parameters={"name": f"%{name}%"}
    )


@tool
def python_repl(
    code: Annotated[str, "The python code to execute."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )
