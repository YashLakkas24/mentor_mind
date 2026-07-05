import streamlit as st
import asyncio
from streamlit_agraph import agraph, Node, Edge, Config
from main import (
    chat,
    get_profile,
    generate_plan,
    generate_insights, 
    build_interactive_graph, 
    calculate_progress,
)

if "memory_graph" not in st.session_state:
    st.session_state.memory_graph = []

if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(page_title="MentorMind", layout="wide")

# st.title("🧠 MentorMind")
st.sidebar.title("🧠 MentorMind")

st.sidebar.markdown("""
Your AI Learning Companion

---

""") 

page = st.sidebar.radio(
    "Navigation",
    [
        "💬 Chat",
        "📊 Dashboard",
        "👤 Profile",
        "📅 Study Plan",
        "📈 Growth Insights",
        "🧠 Knowledge Graph",
    ],
)

if page == "💬 Chat":
    st.info(
        "I remember your goals, projects and learning progress across conversations."
    )
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user = st.chat_input("Ask anything...")

    if user:
        st.session_state.messages.append({"role": "user", "content": user})

        with st.spinner("Thinking..."):
            answer, _ = asyncio.run(chat(user, "chat_1", st.session_state.memory_graph))
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

elif page == "👤 Profile":

    # if st.button("👤 Generate Profile"):

        with st.spinner("Generating profile..."):
            profile = asyncio.run(get_profile("chat_1"))

        st.markdown(profile)

elif page == "📅 Study Plan":
    # if st.button("🚀 Generate Today's Plan"):
        with st.spinner("Creating study plan..."):
            plan = asyncio.run(generate_plan("chat_1"))
        st.markdown(plan)

elif page == "📈 Growth Insights":

    # if st.button("Analyze My Progress"):
        with st.spinner("Analyzing progress..."):
            insights = asyncio.run(generate_insights("chat_1"))

        st.markdown(insights)

elif page == "🧠 Knowledge Graph":

    # if st.button("🧠 View Knowledge Graph"):

        with st.spinner("Building knowledge graph..."):
            nodes, edges = build_interactive_graph(
                {
                    "memories": st.session_state.memory_graph
                }
            )

            config = Config(
                width=900,
                height=650,
                directed=True,
                physics=True,
                hierarchical=False,
            )

            agraph(
                nodes=nodes,
                edges=edges,
                config=config,
            )

elif page == "📊 Dashboard":
    progress = calculate_progress(st.session_state.memory_graph)

    st.metric("Learning Progress", f"{progress}%")

    st.progress(progress / 100)

    skills = 0
    goals = 0
    projects = 0
    completed = 0
    learning = 0

    for memory in st.session_state.memory_graph:

        if memory["category"] == "skill":
            skills += 1

        elif memory["category"] == "goal":
            goals += 1

        elif memory["category"] == "project":
            projects += 1

        elif memory["category"] == "completed":
            completed += 1

        elif memory["category"] == "learning":
            learning += 1

    col1, col2, col3 = st.columns(3)

    col1.metric("Skills", skills)
    col1.metric("Projects", projects)
    col1.metric("Goals", goals)

    col1, col2 = st.columns(2)

    col1.metric("Learning", learning)
    col2.metric("Completed", completed)

    st.subheader("Memory Store")
    st.json(st.session_state.memory_graph)
