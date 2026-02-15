import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import tempfile

# LangChain Core & Google
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Agent & Tools
# Direct source imports to bypass the broken __init__ file
from langchain.agents.agent import AgentExecutor
from langchain.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain.tools import tool
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# RAG Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ═════════════════════════════════════════════
# 1. CORE CONFIGURATION
# ═════════════════════════════════════════════

def get_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0
    )

# ═════════════════════════════════════════════
# 2. TOOL DEFINITIONS FOR THE AGENT
# ═════════════════════════════════════════════

def create_eda_tools(df: pd.DataFrame):
    @tool
    def get_data_summary() -> str:
        """Returns the column names, data types, and missing values of the dataset."""
        summary = {
            "columns": df.columns.tolist(),
            "types": df.dtypes.astype(str).to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "shape": df.shape
        }
        return json.dumps(summary)

    @tool
    def generate_visualization(plot_type: str, column: str) -> str:
        """
        Creates a plot. plot_type must be 'Histogram' or 'Boxplot'.
        Returns 'Success' if the plot is rendered.
        """
        fig, ax = plt.subplots(figsize=(8, 4))
        if plot_type.lower() == "histogram":
            sns.histplot(df[column], kde=True, ax=ax)
        elif plot_type.lower() == "boxplot":
            sns.boxplot(x=df[column], ax=ax)

        st.pyplot(fig)
        plt.close(fig)
        return "Plot displayed successfully in the UI."

    return [get_data_summary, generate_visualization]

# ═════════════════════════════════════════════
# 3. STREAMLIT UI SETUP
# ═════════════════════════════════════════════

st.set_page_config(page_title="Gemini AI Nexus", layout="wide")

with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input("Enter Google API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
    chat_mode = st.selectbox("Choose Agent", ["📝 General Chat", "📊 Data Analyst", "📄 Document RAG"])

    if not api_key:
        st.warning("Please enter an API Key.")
        st.stop()

    llm = get_llm(api_key)

# ═════════════════════════════════════════════
# 4. AGENT LOGIC
# ═════════════════════════════════════════════

# --- DATA ANALYST AGENT ---
if chat_mode == "📊 Data Analyst":
    st.header("Exploratory Data Agent")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview", df.head(3))

        # Setup Chat History
        msgs = StreamlitChatMessageHistory(key="data_chat_history")
        if len(msgs.messages) == 0:
            msgs.add_ai_message("Dataset loaded! Ask me for a summary or a visualization.")

        # Setup Agent
        tools = create_eda_tools(df)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Data Scientist. Use tools to analyze data. ALWAYS check the summary before guessing column names."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # Display Chat History
        for msg in msgs.messages:
            st.chat_message(msg.type).write(msg.content)

        if user_input := st.chat_input("Show me a histogram of..."):
            st.chat_message("human").write(user_input)

            with st.chat_message("ai"):
                with st.spinner("Analyzing..."):
                    # Process Agent call
                    response = agent_executor.invoke({
                        "input": user_input,
                        "chat_history": msgs.messages
                    })
                    output = response["output"]
                    st.write(output)
                    msgs.add_user_message(user_input)
                    msgs.add_ai_message(output)

# --- GENERAL CHAT (Simple Chain) ---
elif chat_mode == "📝 General Chat":
    st.header("General Assistant")
    msgs = StreamlitChatMessageHistory(key="gen_chat_history")

    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input():
        st.chat_message("human").write(prompt)
        res = llm.invoke(prompt)
        st.chat_message("ai").write(res.content)
        msgs.add_user_message(prompt)
        msgs.add_ai_message(res.content)

      # --- DOCUMENT RAG ---
elif chat_mode == "📄 Document RAG":
      st.header("Knowledge Base Agent")

      # 1. Setup Chat History for RAG
      msgs = StreamlitChatMessageHistory(key="rag_chat_history")

      pdf_file = st.file_uploader("Upload PDF", type="pdf")

      if pdf_file:
          # Create a temporary file to load the PDF
          with tempfile.NamedTemporaryFile(delete=False) as tmp:
              tmp.write(pdf_file.getvalue())
              tmp_path = tmp.name

          with st.spinner("Processing document..."):
              # Load and Split Document
              loader = PyPDFLoader(tmp_path)
              docs = loader.load()
              text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
              splits = text_splitter.split_documents(docs)

              # Create Vector Store
              # Note: HuggingFaceEmbeddings runs locally. Ensure 'sentence-transformers' is installed.
              embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
              vectorstore = FAISS.from_documents(splits, embeddings)
              retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

          # 2. Display Previous Messages
          for msg in msgs.messages:
              st.chat_message(msg.type).write(msg.content)

          # 3. Chat Input
          if query := st.chat_input("What is this document about?"):
              st.chat_message("human").write(query)
              msgs.add_user_message(query)

              with st.chat_message("ai"):
                  with st.spinner("Searching knowledge base..."):
                      # Retrieve relevant context
                      context_docs = retriever.invoke(query)
                      context_text = "\n\n".join([d.page_content for d in context_docs])

                      # Build RAG Prompt with Context and History
                      rag_prompt = f"""
                      You are a helpful assistant. Use the following context to answer the user's question.
                      If the answer isn't in the context, say you don't know based on the document.

                      Context:
                      {context_text}

                      Question: {query}
                      """

                      # Get response from LLM
                      response = llm.invoke(rag_prompt)
                      st.write(res := response.content)
                      msgs.add_ai_message(res)

          # Cleanup temporary file
          if os.path.exists(tmp_path):
              os.remove(tmp_path)
