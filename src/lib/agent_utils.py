import operator
from typing import TypedDict, Annotated, Sequence, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str


def create_manager_agent(llm: Any, system_message: str, route_options: list[str]):
    """Create a manager agent that can route messages to the correct agent for further processing."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided input to determine which assistant to route the message to, given the options: {options}."
                " ONLY return the option selected without any additional text."
                " ALWAYS attempt to route the message to the correct assistant."
                " \n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(options=", ".join(route_options))
    return prompt | llm


def create_agent(llm: Any, system_message: str, tools: list[Any] = []):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK, another assistant with different tools "
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any of the other assistants have the final answer or deliverable,"
                    " prefix your response with FINAL ANSWER so the team knows to stop."
                    " You have access to the following tools: {tool_names}. "
                    if len(tools) > 0
                    else "" " \n{system_message}"
                ),
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)

    if len(tools) > 0:
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        return prompt | llm.bind_tools(tools)

    return prompt | llm


# Helper function to create a node for a given agent
def create_agent_node(state: AgentState, agent: Any, name: str) -> AgentState:
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)

    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }
