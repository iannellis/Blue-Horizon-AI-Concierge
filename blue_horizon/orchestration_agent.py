from typing import Literal

from dotenv import load_dotenv
from langchain.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from blue_horizon.information_agent import get_info_agent
from blue_horizon.rooms_sql_agent import get_rooms_agent

load_dotenv()

gpt_model = "gpt-5-nano"

info_agent = get_info_agent()
rooms_agent = get_rooms_agent()

llm = ChatOpenAI(model=gpt_model, temperature=0)


class Route(BaseModel):
    step: Literal["info", "rooms", "none"] = Field(
        description="The next step to take in answering the user's query.")

class State(MessagesState):
    route: str

router = llm.with_structured_output(Route)

system_prompt = """You are an orchestartion agent for a hotel chat-bot. Given a user's
query and chat history, determine whether the user is currently searching for
information about rooms or trying to book a room ("rooms" option), asking for other
information about the hotel ("info" option), or nither ("none" option).
"""

def router_llm(state: State) -> dict:
    """Route the user's query to the appropriate agent."""
    messages = state["messages"]
    decision = router.invoke([SystemMessage(content = system_prompt), *messages])
    return {"route": decision.step}

def info_agent_call(state: State):
    return info_agent.invoke(state)

def rooms_agent_call(state: State):
    return rooms_agent.invoke(state)

def refuse_call(state: State):
    refusal_message = "I'm sorry, I cannot help with that query. I can only provide "
    refusal_message += "information about the hotel and help with room bookings."
    state["messages"].append(AIMessage(refusal_message))

def route_decision(state: State):
    return state["route"]

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

def get_orchestration_agent():
    return agent_built
