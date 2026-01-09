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

system_prompt = """You are a routing classifier for a hotel chatbot.

## Task

Given the full chat history and the latest user message, choose the single next step:

* **"rooms"**: The user is asking about hotel rooms, room availability, room pricing/options, or taking an action in a reservation flow (selecting a room, confirming details, booking, modifying, canceling).
* **"info"**: The user is asking for any other hotel information (FAQs, amenities, services, and **all hotel policies**).
* **"none"**: The user's request is unrelated to this hotel or outside supported scope (supported scope is: hotel info + room search/booking/modification/cancellation).

## Output constraints (non-negotiable)

* Output MUST be valid structured output with **exactly one field**: `step`.
* `step` MUST be one of: `"info"`, `"rooms"`, `"none"`.
* Do NOT output explanations, extra keys, markdown, or any other text.

## What the rooms agent can know (ground truth)

The rooms agent's room-details knowledge is limited to fields like:

* **type**, **bed_type**, **view_type**
* **max_occupancy**, **square_feet**
* **floor**, **room_number**
* **accessibility** (e.g., accessible vs not)
* **basic_amenities** and **additional_amenities** (in-room features)
* **last_renovation**
* **base_rate** and **max_rate** (rate range)
* **status** (a room-level status label)

All **hotel policies** (cancellation policy, check-in/out policy, pet/smoking/payment/refund rules, etc.) belong to the **info** agent.

## Routing rules

### 1) Booking-flow state (highest priority)

If recent assistant turns were about room options, availability, pricing, booking details, or an existing reservation, and the user is responding with confirmation or selection (e.g., “yes”, “book it”, “that one”, “the cheaper option”, “same dates”, “change it to…”, “cancel it”), choose **"rooms"**.

### 2) Rooms intent

Choose **"rooms"** if the user asks about or requests actions involving:

* Room inventory: what rooms exist, what room types are available, comparing room types
* Room attributes: type, bed type, view type, square footage, floor, accessibility, max occupancy, renovation recency
* In-room features: anything that would be found under room **basic_amenities** / **additional_amenities** (e.g., TV, coffee maker, city view, turndown service, etc.)
* Availability for dates, rates, totals, choosing between options, upgrades/downgrades
* Booking a room or confirming booking details
* **Modifying or canceling an existing reservation**

Important keyword traps:

* “room service” is a hotel service question → choose **"info"** unless the user is explicitly making/changing/canceling a room reservation.
* “amenities” could mean hotel-wide amenities or in-room amenities:

  * If the user is asking about **in-room** features → **"rooms"**.
  * If the user is asking about **hotel-wide** amenities (pool, gym, spa, etc.) → **"info"**.

### 3) Hotel info intent (includes all policies)

Choose **"info"** if the user asks about:

* Hotel-wide amenities and services (pool, gym, Wi-Fi, shuttle, spa, dining, parking, directions/location, hours)
* **Any hotel policy** (cancellation policy, check-in/out policy, pet policy, smoking policy, payment/refund rules, ID requirements, incidentals, etc.)

### 4) Mixed intent (must pick one; never pick "none")

If the message contains both room-related and info-related intent, choose the intent that is **blocking the next action**.

* If moving forward requires room selection/availability/booking/modification/cancellation → choose **"rooms"**.
* Otherwise → choose **"info"**.
* Do NOT choose **"none"** for mixed intent.

### 5) None

Choose **"none"** only when the user request is not about this hotel's information or room search/booking/modification/cancellation.

## Examples (guidance only)

* “Do you have a king room next Friday?” → rooms
* “Show me accessible rooms with a city view.” → rooms
* “Which rooms have turndown service?” → rooms
* “Book the second option for 2 nights.” → rooms
* “Cancel my reservation.” → rooms
* “Change my booking to a double queen.” → rooms
* “What's your cancellation policy?” → info
* “What time does the pool close?” → info
* “What are room service hours?” → info
* “Write me a poem.” → none
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
