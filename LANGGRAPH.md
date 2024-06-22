# LangGraph Project Experiment Details

LangGraph is a library that allows us to manage a multi-agent LLM apps with a higher level of control compared to other libraries, like CrewAI, for instance.

Fist, lets talk about LangChain and LangGraph required concepts.

## LangChain Tools

A tool is just a plain old function, that can receive parameters and return values, with some required metadata.

LangChain provides a `tool` decorator to be used as a wrapper for functions, adding the required metadata so this can be used as an agent tool.

Tools are designed to give LLMs the ability to talk to the external world, by interacting with databases, Rest APIs, doing complex calculations, and so on. 

They are the workforce behind an agent and what makes LLM really powerful for task automations.


## LangChain Agents

LangChain Agents are simply LLM instances that can generate messages to interact with tools.

An agent will not run the tool by itself, instead, by binding tools to the LLM instance, it can decide wheather a call to an specific tool is required and provide all parameters required for it to run.

The raw flow can be seen in the `src/titanic_chain_tool.py` script, that uses chains to emulate an agent flow to call a tool and process its result. 

On the other hand, as we can see at the `src/titanic_agent_tool.py` script, LangChain provides the `create_tool_calling_agent` method to abstract the previous chain for simple use cases.


### Why Multi-Agent is Better?

For simple tasks, when the user give an input and we need to understand and run a tool to process it, like querying a database table, a single agent is more than enough. It can manage this by giving the tool the required parameters from the user prompt and return a processed result with a really high level of confidence.

Like any other LLM powered solution, an agent will be as much as good as the prompt it uses to describe its scope. We can provide a really complex and well engineered prompt to create an efficient agent, but if we need to accomplish multiple tasks, a single prompt should lead to undesirable results, as it will become hard for it to understand what it needs to do and when.

For these scenarios, a multi-agent architecture, where they can colaborate with each other, will be much more efficient as each one will be able to accomplish its own task and provide context for any extra interaction.

That's what LangGraph allows us to do.

## LangGraph State

LangGraph allows us to manage multiple agents and make them collaborate with each other by designing a workflow where messages can flow from node to node, giving each agent all information it should need to accomplish its task.

A workflow state can be anything. A string with the last step result, an object with multiple information or even a class, that can also have behavior to be used during the workflow execution.

States in LangChain are immutable and are managed by the library using a reducer approach.

## LangGraph Nodes

Nodes are the basic components of a LangGraph workflow. They are responsible to execute any required logic based on the current workflow state.

A node is nothing more than a function that receives the current state instance and return a new state value. They can wrap any kind of logic, but their main goal is to work as an agent executor, by invoking them and returning their results, with or without any extra atributes or transformation. 

The result of a node is then merged to the current state and then send to the next one or returned to the caller as the workflow result.

This is how LangGraph works. It's not an agent executor, it can execute anything in a way that each step can provide context to the next, by adding routing rules between them.

## LangGraph Edges

The main use case to create a workflow is to orchestrate multiple nodes executions in a way they can make use of context from previous steps and provide new information to next ones. So, if we have multiple nodes, how to decide which next steps we need to do based on the current one?

Edges are the way we connect nodes to each other. They can be defined as static or dynamic edges. 

Static edges are used when we know the sequence between two nodes. We just need to provide the source and the destination node to set the path between them.

Dynamic edges are used when the next step depends on the previous one results. The main scenario is when the node result depends on the user's input. This how we can create agent collaboration, by using their results as the decision point to what to do next, even by calling the previous step again to refine it's answer or to return right away if the task can't be accomplished.

Dynamic edges uses the workflow state to make decisions.

# LangGraph State Graph Experiment

This experiment uses a manager agent to interpret the user's input and select the next agent to process it.

## Code Explanation

### Defining the Workflow State

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
```

We start by defining the workflow state. It will be a list of messages, where messages will be added to the list, and the last node executed, identified here as the sender. The sender parameter will allow us to route properly, when required.

### Creating Agents

```python
manager_agent = create_manager_agent(
    llm,
    system_message="""Route the user's input to the correct agent based on the user's input.""",
    route_options=["Calculate", "Elaborate"],
)

calculate_expression_agent = create_agent(
    llm,
    system_message="""Calculate the expression based on the user's input""",
    tools=[python_repl],
)

elaborate_agent = create_agent(
    llm,
    system_message="""Elaborate the user's input for a better understanding""",
    tools=[],
)
```

Here we are creating the agents using a helper method that receives parameters to improve the main agent prompt and also to provide tools each agent can make use of.

The manager agent will process the user input and is only responsible to return one of the options. These will be used later for routing purposes.

### Creating Nodes

```python
manager_node = functools.partial(create_agent_node, agent=manager_agent, name="manager")
calculate_expression_node = functools.partial(
    create_agent_node, agent=calculate_expression_agent, name="calculate"
)
elaborate_node = functools.partial(
    create_agent_node, agent=elaborate_agent, name="elaborate"
)
tool_node = ToolNode([python_repl])
```

Then we create the nodes that will going to trigger each agent and will update the state with their responses, by adding a new AIMessage instance to the message list.

The `ToolNode` class, wraps the logic to run tools when a node return as message with call details, and also to provide a `ToolMessage` instance to be added to the state for further processing.

### Adding Nodes to the Workflow

```python
workflow = StateGraph(AgentState)
workflow.add_node("manager", manager_node)
workflow.add_node("calculate", calculate_expression_node)
workflow.add_node("elaborate", elaborate_node)
workflow.add_node("call_tool", tool_node)
```

Now we create the workflow and add all node references required to process the tasks.

The workflow is created by defining the state object used to validate and keep the state. 

### Workflow Entry Point

```python
workflow.set_entry_point("manager")
```

Here we just say that any message sent to the workflow will be routed to the manager node.

### Defining Dynamic Edge Routers

```python
def manager_router(state) -> Literal["Calculate", "Elaborate"]:
    messages = state["messages"]
    last_message = messages[-1]

    return last_message.content

def router(state) -> Literal[
    "call_tool",
    "__end__",
]:
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"

    # Any agent decided the work is done
    return "__end__"
```

These helper methods are used to get the last message in the current state to decide what to do next.

The first one, returns the return value from the manager, as we defined earlier. This will be used by a dynamic edge to select the next step to route to.

The second one will be used to route from the other nodes. It's required to make sure the node will end the workflow if its current returned message is not a tool call request.

> By returning `__end__` we tell LangGraph to end the workflow and return the state to the caller, which will reply back to the user.

### Defining the Edges

```python
workflow.add_conditional_edges(
    "manager",
    router_router,
    {
        "Calculate": "calculate",
        "Elaborate": "elaborate",
    },
)

workflow.add_conditional_edges(
    "calculate",
    router,
)

workflow.add_conditional_edges(
    "elaborate",
    router,
)

workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
)
```

Here we define the route rules. We always use dynamic rules, as we don't know what each node will require in advance.

The first dynamic edge has an extra parameter because we set the name of the nodes differently of the expected output from the router method it uses to decide the next step. So, this we can map the router result to a node by simply passing a map to the edge setup.

The last edge definition tell the tools node (`call_tool`) to return to its caller node. A tool node only returns a `ToolMessage` object for the state update, but it doesn't change the last sender parameter value. This way, we can get the caller name from the state and know where to send the tool result back to improve the response.

> At this time, as the tool result is returned to the node, the next result will not be a tool call message, but a reply to the user. The router will check it again, and will route to the end.

### Executing the Workflow

```python
memory = SqliteSaver.from_conn_string(":memory:")

# Compile the graph
app = workflow.compile(checkpointer=memory)

# Draw the graph image
graph_image = app.get_graph().draw_png()


# Chainlit Interface Hooks
@cl.on_chat_start
async def on_start():
    image = cl.Image(content=graph_image, name="Current Graph", display="inline")

    await cl.Message(
        content="Here is the graph being executed", elements=[image], author="system"
    ).send()


@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": "inventory_manager_thread"}}

    # define the messages for the input
    inputs = {
        "messages": [HumanMessage(content=msg.content)],
    }

    response: AgentState = await make_async(app.invoke)(inputs, config, debug=True)

    await cl.Message(content=response["messages"][-1].content, author="system").send()
```

Here we finish the workflow compilation, passing it a memory component to keep track of the execution, and generating the workflow chart to show in the Chainlit chat interface.

The Chainlit hooks are used by the cli to start the interface and initiate the conversation, as well as listen to any new message and trigger the workflow.

```shell
$ chainlit run src/langraph_state_graph.py
```

#### Exemple Questions
- Calculate 10 + 40 * 5
- Explain the previous calculation












