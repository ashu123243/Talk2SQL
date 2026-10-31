from typing import Annotated, Any, TypedDict, Required

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class GraphState(TypedDict, total=False):
    """
    Central state shared across Talk2SQL LangGraph nodes.

    Each node reads the information it needs and returns
    structured updates to this state.
    """
    
    # 1. CONVERSATION
    messages: Annotated[list[BaseMessage], add_messages]

    user_question: Required[str]
    request_id: str
    conversation_id: str

    # 2. QUERY PLANNING
    intent: str
    query_plan: dict[str, Any]

    required_metrics: list[str]
    required_dimensions: list[str]
    filters: list[dict[str, Any]]
    
    # 3. DATABASE SCHEMA
    schema_snapshot: dict[str, Any]
    relevant_tables: list[str]

    table_relationships: list[dict[str, Any]]
    foreign_keys: list[dict[str, Any]]
    constraints: list[dict[str, Any]]
    
    # 4. SQL GENERATION AND VALIDATION
    generated_sql: str
    validated_sql: str

    sql_valid: bool
    validation_errors: list[str]
    
    # 5. DATABASE EXECUTION
    query_result: list[dict[str, Any]]

    execution_time: float
    execution_metrics: dict[str, Any]
    execution_error: str | None
    
    # 6. RESULT VERIFICATION
    verification_checks: list[dict[str, Any]]

    verification_passed: bool
    verification_warnings: list[str]

    confidence_score: float | None

    # 7. RETRY MANAGEMENT
    retry_count: int
    max_retries: int
    retry_reason: str
    
    # 8. ANALYTICS
    analysis_result: dict[str, Any]
    chart_data: list[dict[str, Any]]
    insights: list[str]
    
    # 9. REPORT GENERATION
    final_report: str
    report_status: str
    
    # 10. WORKFLOW MANAGEMENT
    status: str
    next_step: str
    error: str | None
    