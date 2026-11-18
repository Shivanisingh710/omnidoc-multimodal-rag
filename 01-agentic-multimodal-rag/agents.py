from pydoc import text
from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage

from tools import pdf_retrieval_tool


# Define Agent State
class AgentState(TypedDict):
    # Sequence[BaseMessage] - List of values (it will only hold user message or Assistant response)
    # Annotated - Used to add metadata to the type, in this case, the add_messages function is used to specify that 
    # this field should be treated as a list of messages in the LangGraph framework.
    # add_messages - A function that adds messages to the state, ensuring that the messages are properly handled and 
    # stored within the graph's state management system.
    messages: Annotated[Sequence[BaseMessage], add_messages]


def get_llm(provider: str, api_key: str):
    """Instantiate the appropriate multimodal LLM."""
    if provider == "Google Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-3.6-flash", 
            google_api_key=api_key,
            temperature=0.2
        )
    elif provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini", 
            api_key=api_key,
            temperature=0.2
        )
    else:
        raise ValueError("Unsupported LLM Provider.")


def create_multimodal_rag_graph(provider: str, api_key: str):
    """Builds and compiles the LangGraph state graph."""
    
    # 1. Initialize Model & Tools
    tools = [pdf_retrieval_tool]
    llm = get_llm(provider, api_key)
    llm_with_tools = llm.bind_tools(tools)

    # 2. Define Node Execution
    def agent_node(state: AgentState):
        system_prompt = SystemMessage(content=(
            "You are an expert multimodal document assistant capable of analyzing text, "
            "tables, graphs, and images inside PDFs. "
            "Always use the `pdf_retrieval_tool` to search for answers when user queries relate "
            "to uploaded files. If retrieved results contain image base64 data, analyze visual content carefully."
        ))
        
        messages = [system_prompt] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response.content[0]['text']]}
        # return {"messages": [response]}  
        

    # 3. Construct Graph
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))

    # 4. Set Edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile()