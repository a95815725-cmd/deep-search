import operator
from typing import Annotated, Dict, List, TypedDict


class SubQuery(TypedDict):
    query: str
    status: str  # "pending" | "searched"
    results: List[Dict]


class Citation(TypedDict):
    doc_name: str
    page_num: int
    section: str
    text_excerpt: str


class AgentState(TypedDict):
    # Input
    original_question: str
    model_name: str

    # Planning
    sub_queries: List[SubQuery]
    search_strategy: str  # desrciption of the search plan

    # Retrieval
    retrieved_documents: Annotated[List[Dict], operator.add]  # accumulats  iterations
    iteration_count: int

    # Reflection
    reflection_notes: str
    gaps_identified: List[str]
    sufficient_context: bool

    # Output
    final_answer: str
    citations: List[Citation]
    confidence_score: float
    reasoning_trace: Annotated[List[str], operator.add]  # log of agent steps
