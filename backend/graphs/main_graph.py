from uuid import uuid4
from typing import Literal

from langchain_core.messages import HumanMessage, AIMessage

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from backend.state.graph_state import GraphState


def intake_node(state: GraphState) -> dict:
    """
    Receives and validates the user's question.

    Initializes request and conversation identifiers
    and records the user's message.
    """
    
    user_question = state.get("user_question", "").strip()
    
    if not user_question:
        return {
            "status": "error",
            "next_step": "stop",
            "error": "User question cannot be empty."
        }
    
    request_id = state.get("request_id") or str(uuid4())

    conversation_id = (
        state.get("conversation_id") or str(uuid4())
    )
    
    return {
        "user_question": user_question,
        "request_id": request_id,
        "conversation_id": conversation_id,
        "messages": [
            HumanMessage(content=user_question)
        ],
        "status": "received",
        "next_step": "planning",
        "error": None
    }


def route_after_intake(state: GraphState) -> Literal["continue", "stop"]:
    """
    Determines whether the workflow should continue
    or move to the error node.
    """

    if state.get("error"):
        return "stop"

    if state.get("status") == "error":
        return "stop"

    return "continue"


def planning_node(state: GraphState) -> dict:
    """
    Initial planning node.

    Actual AI-powered query planning will be implemented
    in a later development stage.
    """

    return {
        "intent": "pending",

        "query_plan": {
            "status": "not_implemented",
            "description": (
                "The actual query planning agent "
                "will be implemented in a later stage."
            )
        },

        "status": "planning_placeholder_complete",
        "next_step": "completion"
    } 


def completion_node(state: GraphState) -> dict:
    """
    Marks successful completion of the foundation workflow.
    """

    return {
        "status": "foundation_complete",
        "next_step": "end",

        "messages": [
            AIMessage(
                content=(
                    "Talk2SQL foundation workflow "
                    "completed successfully. "
                    "The actual SQL processing agents "
                    "will be added in subsequent stages."
                )
            )
        ]
    }   
   
 
def error_node(state: GraphState) -> dict:
    """
    Handles invalid input and workflow errors.
    """

    return {
        "status": "failed",
        "next_step": "end"
    }


def build_graph():
    """
    Constructs and compiles the main Talk2SQL LangGraph.

    Uses MemorySaver for in-memory checkpointing.
    """

    # Create the workflow
    workflow = StateGraph(GraphState)

    # Register nodes
    workflow.add_node("intake", intake_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("completion", completion_node)
    workflow.add_node("error", error_node)

    # Define starting point
    workflow.add_edge(START, "intake")

    # Conditional routing after intake
    workflow.add_conditional_edges(
        "intake",
        route_after_intake,
        {
            "continue": "planning",
            "stop": "error"
        }
    )

    # Connect planning to completion
    workflow.add_edge("planning", "completion")

    # Connect terminal nodes
    workflow.add_edge("completion", END)
    workflow.add_edge("error", END)

    # Create in-memory checkpointer
    memory = MemorySaver()

    # Compile workflow
    compiled_graph = workflow.compile(
        checkpointer=memory
    )

    return compiled_graph
   

graph = build_graph()
