# Install required packages
# pip install langchain langchain-community langsmith langgraph openai

import os
from typing import Annotated, List, Tuple, TypedDict, Union, cast

# LangChain imports
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

# LangGraph imports for orchestration
from langgraph.graph import END, StateGraph
from langgraph.checkpoint import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

# Define our state type
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    query: str
    research_results: List[str]
    code_results: str
    summary: str
    final_answer: str

# Agent specialized for research tasks
def create_research_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a specialized research agent that can find information on various topics.
Your task is to provide concise and accurate research results based on the query.
Focus on finding facts and reliable information."""),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{query}")
    ])
    
    model = ChatOpenAI(model="gpt-4", temperature=0)
    research_chain = prompt | model
    
    def research_agent(state: AgentState) -> AgentState:
        query = state["query"]
        messages = state.get("messages", [])
        
        response = research_chain.invoke({"messages": messages, "query": query})
        state["research_results"] = [response.content]
        state["messages"] = messages + [HumanMessage(content=query), response]
        
        return state
    
    return research_agent

# Agent specialized for code generation
def create_coding_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a specialized coding agent that can generate Python code.
Your task is to write clean, efficient, and well-documented code based on requirements.
The code should be ready to use."""),
        MessagesPlaceholder(variable_name="messages"),
        ("human", """Based on these research results, generate appropriate Python code:
{research_results}

Original query: {query}""")
    ])
    
    model = ChatOpenAI(model="gpt-4", temperature=0)
    coding_chain = prompt | model
    
    def coding_agent(state: AgentState) -> AgentState:
        query = state["query"]
        research_results = state["research_results"]
        messages = state.get("messages", [])
        
        response = coding_chain.invoke({
            "messages": messages,
            "query": query,
            "research_results": research_results
        })
        
        state["code_results"] = response.content
        state["messages"] = messages + [
            HumanMessage(content=f"Generate code based on: {research_results}"),
            response
        ]
        
        return state
    
    return coding_agent

# Agent for summarizing results
def create_summary_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a specialized summarization agent.
Your task is to create a concise summary that integrates research findings and code solutions."""),
        MessagesPlaceholder(variable_name="messages"),
        ("human", """Summarize the following:
Research results: {research_results}
Code: {code_results}

Original query: {query}""")
    ])
    
    model = ChatOpenAI(model="gpt-4", temperature=0)
    summary_chain = prompt | model
    
    def summary_agent(state: AgentState) -> AgentState:
        query = state["query"]
        research_results = state["research_results"]
        code_results = state["code_results"]
        messages = state.get("messages", [])
        
        response = summary_chain.invoke({
            "messages": messages,
            "query": query,
            "research_results": research_results,
            "code_results": code_results
        })
        
        state["summary"] = response.content
        state["messages"] = messages + [
            HumanMessage(content=f"Summarize research and code results"),
            response
        ]
        
        return state
    
    return summary_agent

# Decision-making agent to determine next steps
def create_decision_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a workflow coordinator that decides the next step in the process.
Your options are:
- "research": If more research is needed
- "code": If we should proceed to coding
- "summarize": If we should summarize the findings
- "finalize": If we should prepare the final answer
- "end": If the process is complete

Choose the most appropriate next step based on the current state."""),
        MessagesPlaceholder(variable_name="messages"),
        ("human", """Current state:
Query: {query}
Research results: {research_results}
Code results: {code_results}
Summary: {summary}

What should be the next step?""")
    ])
    
    parser = JsonOutputParser()
    model = ChatOpenAI(model="gpt-4", temperature=0)
    decision_chain = prompt | model | parser
    
    def decision_agent(state: AgentState) -> str:
        # Extract relevant state information
        decision_input = {k: state.get(k, "") for k in ["query", "research_results", "code_results", "summary"]}
        decision_input["messages"] = state.get("messages", [])
        
        # Make decision about next step
        result = decision_chain.invoke(decision_input)
        return result.get("decision", "research")
    
    return decision_agent

# Agent for final answer generation
def create_final_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are responsible for providing the final comprehensive answer.
Integrate all the information collected during the process into a cohesive response."""),
        MessagesPlaceholder(variable_name="messages"),
        ("human", """Create a final answer based on:
Query: {query}
Research results: {research_results}
Code results: {code_results}
Summary: {summary}""")
    ])
    
    model = ChatOpenAI(model="gpt-4", temperature=0)
    final_chain = prompt | model
    
    def final_agent(state: AgentState) -> AgentState:
        query = state["query"]
        research_results = state["research_results"]
        code_results = state["code_results"]
        summary = state["summary"]
        messages = state.get("messages", [])
        
        response = final_chain.invoke({
            "messages": messages,
            "query": query,
            "research_results": research_results,
            "code_results": code_results,
            "summary": summary
        })
        
        state["final_answer"] = response.content
        state["messages"] = messages + [
            HumanMessage(content="Generate final comprehensive answer"),
            response
        ]
        
        return state
    
    return final_agent

# Create our orchestration graph
def create_agent_graph():
    # Initialize specialized agents
    research_agent = create_research_agent()
    coding_agent = create_coding_agent()
    summary_agent = create_summary_agent()
    decision_agent = create_decision_agent()
    final_agent = create_final_agent()
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("research", research_agent)
    workflow.add_node("code", coding_agent)
    workflow.add_node("summarize", summary_agent)
    workflow.add_node("finalize", final_agent)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "research",
        decision_agent,
        {
            "research": "research",  # Need more research
            "code": "code",          # Proceed to coding
            "summarize": "summarize", # Summarize findings
            "finalize": "finalize",   # Create final answer
            "end": END                # End the process
        }
    )
    
    workflow.add_conditional_edges(
        "code",
        decision_agent,
        {
            "research": "research",
            "code": "code",
            "summarize": "summarize",
            "finalize": "finalize",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "summarize",
        decision_agent,
        {
            "research": "research",
            "code": "code",
            "summarize": "summarize",
            "finalize": "finalize",
            "end": END
        }
    )
    
    # The finalize node always ends the process
    workflow.add_edge("finalize", END)
    
    # Set the entry point
    workflow.set_entry_point("research")
    
    # Compile the graph
    return workflow.compile()

# Example usage
def main():
    # Create the agent graph
    agent_graph = create_agent_graph()
    
    # Create memory saver for state persistence
    memory = MemorySaver()
    
    # Initialize the state
    initial_state = AgentState(
        messages=[],
        query="Create a Python function to analyze stock price data and identify potential buy/sell signals using moving averages",
        research_results=[],
        code_results="",
        summary="",
        final_answer=""
    )
    
    # Execute the graph
    result = agent_graph.invoke(initial_state, config={"recursion_limit": 25})
    
    # Print the final result
    print("Final Answer:")
    print(result["final_answer"])
    print("\nGenerated Code:")
    print(result["code_results"])

if __name__ == "__main__":
    main()
