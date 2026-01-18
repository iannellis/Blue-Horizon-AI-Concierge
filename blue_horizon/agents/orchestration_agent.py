from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from blue_horizon.agents.information_agent import (
    InfoAgentFactory,
    InfoRagResources,
    get_redis_url,
    load_info_config,
)
from blue_horizon.agents.rooms_sql_agent import (
    RoomsAgentFactory,
    RoomsSqlResources,
    get_pgsql_db_url,
    load_rooms_sql_config,
)

load_dotenv()

gpt_model = "gpt-5-nano"

info_rag_config = load_info_config(Path.cwd() / "../blue_horizon/info_config.toml")
info_rag_resources = InfoRagResources(redis_url=get_redis_url(), config=info_rag_config)
await info_rag_resources.startup_check()
info_agent_factory = InfoAgentFactory(resources=info_rag_resources, config=info_rag_config)
info_agent = info_agent_factory.build()

rooms_sql_config = load_rooms_sql_config(Path.cwd() / "../blue_horizon/rooms_sql_config.toml")
rooms_sql_resources = RoomsSqlResources(pgsql_db_url=get_pgsql_db_url(), config=rooms_sql_config)
await rooms_sql_resources.startup_check()
rooms_agent_factory = RoomsAgentFactory(config=rooms_sql_config, resources=rooms_sql_resources)
rooms_agent = rooms_agent_factory.build()

llm = ChatOpenAI(model=gpt_model, temperature=0)


class Route(BaseModel):
    step: Literal["info", "rooms", "none"] = Field(...,
        description="The next step to take in answering the user's query.")

class State(MessagesState):
    route: str | None

router = llm.with_structured_output(Route)

prompts_folder = "../system_prompts"
prompts_dir = (Path(__file__).parent / prompts_folder).resolve()
filename = "orchestration_prompt.txt"
prompt_file = (prompts_dir / filename).resolve()

system_prompt = prompt_file.read_text(encoding="utf-8")

async def router_llm(state: State) -> dict:
    """Route the user's query to the appropriate agent."""
    messages = state["messages"]
    decision = await router.ainvoke([SystemMessage(content = system_prompt), *messages])
    return {"route": getattr(decision, "step", "none")}

async def info_agent_call(state: State):
    return await info_agent.ainvoke(state)

async def rooms_agent_call(state: State):
    return await rooms_agent.ainvoke(state)

def refuse_call(state: State):
    refusal_message = "I'm sorry, I cannot help with that query. I can only provide "
    refusal_message += "information about the hotel and help with room bookings."
    state["messages"].append(AIMessage(refusal_message))

def route_decision(state: State):
    return state["route"]

def get_orchestration_agent():
    agent_graph = StateGraph(State)

    agent_graph.add_node("router_llm", router_llm)
    agent_graph.add_node("info_agent_call", info_agent_call)
    agent_graph.add_node("rooms_agent_call", rooms_agent_call)
    agent_graph.add_node("refuse_call", refuse_call)

    agent_graph.add_edge(START, "router_llm")
    agent_graph.add_conditional_edges("router_llm", route_decision,
                                    {"info": "info_agent_call",
                                    "rooms": "rooms_agent_call",
                                    "none": "refuse_call"})
    agent_graph.add_edge("info_agent_call", END)
    agent_graph.add_edge("rooms_agent_call", END)
    agent_graph.add_edge("refuse_call", END)

    agent_built = agent_graph.compile()
    return agent_built
