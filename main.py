import asyncio, os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from streamlit_agraph import Node, Edge
from datetime import datetime

load_dotenv(override=True)
import cognee

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def calculate_progress(memory_graph):
    completed = 0
    learning = 0

    for memory in memory_graph:
        if memory["category"] == "completed":
            completed += 1

        elif memory["category"] == "learning":
            learning += 1

    total = completed + learning

    if total == 0:
        return 0
    return int((completed / total) * 100)


def build_context(memories):
    context = ""

    for memory in memories:
        if hasattr(memory, "answer"):
            context += memory.answer + "\n"

        elif hasattr(memory, "text"):
            context += memory.text + "\n"
    return context


async def generate_insights(session_id):
    memories = await cognee.recall(
        query_text="What do you know about me?", session_id=session_id
    )

    context = build_context(memories)

    prompt = f"""
        You are an AI Learning Coach.

        Based ONLY on the memories below, analyze the user's learning journey.

        Return:

        ## Strengths
        - ...

        ## Weaknesses
        - ...

        ## Progress
        - ...

        ## Recommended Next Steps
        - ...

        ## Risk Areas
        - ...

        Keep it concise.

        Memories:

        {context}
        """

    insights = await ask_llm(prompt)
    return insights


async def extract_memory(user_message: str):

    prompt = f"""
You are the Memory Manager for an AI Learning Assistant.

Your job is to decide whether new information should be remembered.

Return ONLY valid JSON.

====================================================
OUTPUT FORMAT
====================================================

If nothing should be stored:

{{
    "store": false,
    "memories": []
}}

Otherwise:

{{
    "store": true,
    "memories": [
        {{
            "operation": "AUTO",
            "category": "skill",
            "entity": "Python",
            "fact": "User knows Python."
        }}
    ]
}}

====================================================
AVAILABLE OPERATIONS
====================================================

AUTO
ADD
UPDATE
DELETE
IGNORE

Use AUTO unless the operation is obvious.

Examples:

"I know Python."
→ AUTO

"I completed Python."
→ UPDATE

"I don't know Python anymore."
→ DELETE

"Hi"
→ IGNORE

====================================================
AVAILABLE CATEGORIES
====================================================

personal
goal
skill
learning
completed
weakness
project
achievement
preference

====================================================
ENTITY
====================================================

Extract the main entity.

Examples

"I know Python."

entity:
Python

------------------

"I completed Machine Learning."

entity:
Machine Learning

------------------

"I'm building an AI Mentor."

entity:
AI Mentor

------------------

"I want to become an AI Engineer."

entity:
AI Engineer

====================================================
RULES
====================================================

Store ONLY long-term information.

Ignore:

- greetings
- jokes
- temporary requests
- one-time questions
- assistant responses
- weather
- explanations
- filler conversation

Extract ALL memories.

Use third-person facts.

Never invent information.

====================================================
OPERATION RULES
====================================================

ADD

Brand new information.

Example

"I know FastAPI."

UPDATE

The user has progressed.

Examples

"I'm learning Python."

↓

"I completed Python."

"I'm learning Machine Learning."

↓

"I know Machine Learning."

DELETE

The user explicitly says previous information is no longer true.

Examples

"I don't know Java anymore."

"I never learned Machine Learning."

"I stopped using Flask."

IGNORE

No long-term memory.

Examples

"Hello"

"Thanks"

"What's recursion?"

AUTO

Use AUTO whenever the intent cannot be determined confidently.

====================================================
EXAMPLES
====================================================

Input:

Hi

Output

{{
    "store": false,
    "memories": []
}}

------------------------------------

Input

I know Python and Java.

Output

{{
    "store": true,
    "memories": [
        {{
            "operation": "AUTO",
            "category": "skill",
            "entity": "Python",
            "fact": "User knows Python."
        }},
        {{
            "operation": "AUTO",
            "category": "skill",
            "entity": "Java",
            "fact": "User knows Java."
        }}
    ]
}}

------------------------------------

Input

I'm learning LangGraph.

Output

{{
    "store": true,
    "memories": [
        {{
            "operation": "AUTO",
            "category": "learning",
            "entity": "LangGraph",
            "fact": "User is learning LangGraph."
        }}
    ]
}}

------------------------------------

Input

I completed NumPy.

Output

{{
    "store": true,
    "memories": [
        {{
            "operation": "UPDATE",
            "category": "completed",
            "entity": "NumPy",
            "fact": "User completed NumPy."
        }}
    ]
}}

------------------------------------

Input

I don't know Machine Learning anymore.

Output

{{
    "store": true,
    "memories": [
        {{
            "operation": "DELETE",
            "category": "skill",
            "entity": "Machine Learning",
            "fact": "User no longer knows Machine Learning."
        }}
    ]
}}

------------------------------------

Input

I know Python. I completed NumPy. I'm building an AI Mentor.

Output

{{
    "store": true,
    "memories": [
        {{
            "operation": "AUTO",
            "category": "skill",
            "entity": "Python",
            "fact": "User knows Python."
        }},
        {{
            "operation": "UPDATE",
            "category": "completed",
            "entity": "NumPy",
            "fact": "User completed NumPy."
        }},
        {{
            "operation": "AUTO",
            "category": "project",
            "entity": "AI Mentor",
            "fact": "User is building an AI Mentor."
        }}
    ]
}}

====================================================
USER MESSAGE
====================================================

{user_message}
"""

    response = await client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )

    import json

    return json.loads(response.choices[0].message.content)


async def ask_llm(prompt: str):
    response = await client.chat.completions.create(
        model="gpt-4.1-mini", messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


async def process_memory(memory_json, session_id, memory_graph):

    if not memory_json["store"]:
        return

    priority = {
        "learning": 1,
        "skill": 2,
        "completed": 3,
    }

    for memory in memory_json["memories"]:

        category = memory["category"]
        fact = memory["fact"]
        entity = memory["entity"]
        operation = memory["operation"]

        # ---------------- IGNORE ----------------

        if operation == "IGNORE":
            continue

        # ---------------- DELETE ----------------

        if operation == "DELETE":

            memory_graph[:] = [
                m for m in memory_graph if m["entity"].lower() != entity.lower()
            ]

            print(f"DELETE : {entity}")
            continue

        # ---------------- SEARCH ----------------

        existing = None

        for old_memory in memory_graph:

            if old_memory["entity"].lower() == entity.lower():
                existing = old_memory
                break

        # ---------------- ADD ----------------

        if existing is None:

            memory["updated_at"] = datetime.now().isoformat()
            memory["confidence"] = 1.0

            memory_graph.append(memory)

            await cognee.remember(fact, session_id=session_id)

            print(f"ADD : {entity}")

            continue

        # ---------------- DUPLICATE ----------------

        if existing["category"] == category and existing["fact"] == fact:

            existing["confidence"] += 0.1

            print(f"IGNORE : {entity}")

            continue

        # ---------------- UPDATE ----------------

        if priority.get(category, 0) >= priority.get(existing["category"], 0):

            existing["category"] = category
            existing["fact"] = fact
            existing["operation"] = operation
            existing["updated_at"] = datetime.now().isoformat()
            existing["confidence"] += 0.2

            await cognee.remember(fact, session_id=session_id)

            print(f"UPDATE : {entity}")

            continue

        # ---------------- CONTRADICTION ----------------

        existing["confidence"] -= 0.5

        print(f"CONTRADICTION : {entity}")


async def get_profile(session_id):

    memories = await cognee.recall(
        query_text="What do you know about me?", session_id=session_id
    )

    context = build_context(memories)

    prompt = f"""
        You are an AI profile generator.

        Using only the provided memories, organize the information into:

        Name
        Goals
        Skills
        Learning
        Completed
        Projects
        Weaknesses
        Achievements
        Preferences

        If a section has no information, write "Not available."

        Memories:
        {context} 
    """

    profile = await ask_llm(prompt)
    return profile


async def generate_plan(session_id):

    memories = await cognee.recall(
        query_text="Generate today's study plan", session_id=session_id
    )

    context = build_context(memories)

    prompt = f"""
    You are an AI mentor.

        Using ONLY the memories below,

        create today's study plan.

        Prioritize:

        1. Weaknesses
        2. Unfinished learning
        3. Current projects
        4. User's long-term goal

        Explain briefly why you recommended each task.

        Memories:

        {context}
            
    """
    plan = await ask_llm(prompt)
    return plan


def extract_entity(fact):

    prefixes = [
        "User knows ",
        "User is learning ",
        "User completed ",
        "User is building ",
        "User wants to become ",
        "User struggles with ",
        "User prefers ",
        "User's name is ",
    ]

    for prefix in prefixes:
        if fact.startswith(prefix):
            return fact.replace(prefix, "").strip(". ")
    return fact.strip(". ")


def build_interactive_graph(memory_json):

    nodes = {}
    edges = []

    # User node
    nodes["User"] = Node(
        id="User",
        label="👤 User",
        size=35,
    )

    for memory in memory_json["memories"]:

        category = memory["category"]
        entity = extract_entity(memory["fact"])

        # Category node
        if category not in nodes:
            nodes[category] = Node(
                id=category,
                label=category.capitalize(),
                size=25,
            )

        # Entity node
        if entity not in nodes:
            nodes[entity] = Node(
                id=entity,
                label=entity,
                size=20,
            )

        # Edges
        edge1 = Edge(source="User", target=category)
        edge2 = Edge(source=category, target=entity)

        if edge1 not in edges:
            edges.append(edge1)

        if edge2 not in edges:
            edges.append(edge2)

    return list(nodes.values()), edges


async def chat(user, session_id, memory_graph):

    memory_fact = await extract_memory(user)

    await process_memory(memory_fact, session_id, memory_graph)

    memory = await cognee.recall(
        query_text=user,
        session_id=session_id,
    )

    context = build_context(memory)

    prompt = f"""
        You are MentorMind, an AI learning mentor.

        Use the retrieved memories whenever they are relevant.

        Relevant memories:

        {context}

        Current user message:

        {user}

        Provide a helpful and personalized response.
        """

    answer = await ask_llm(prompt)

    return answer, memory_fact


async def main():
    
    while True:
        user = input("You: ")

        if user == "exit":
            await cognee.improve(session_ids=["chat_1"])
            break

        if user == "profile":
            profile = await get_profile("chat_1")
            print(profile)
            continue

        if user == "plan":
            plan = await generate_plan("chat_1")
            print(plan)
            continue

        answer, _ = await chat(user, "chat_1", [])
        print("AI: ", answer)


if __name__ == "__main__":
    asyncio.run(main())
