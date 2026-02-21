import os
import json
from dotenv import load_dotenv
from groq import Groq

from agent.tools import (
    answer_from_document,
    calculator,
    get_current_time,
)

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# 🔧 Tool Schemas for Groq
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "answer_from_document",
            "description": "Answer questions using the uploaded PDF document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current system time.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


# 🔁 Multi-Step Agent Loop
def run_agent(user_query: str):
    messages = [
        {
            "role": "system",
            "content": (
        "You are IntelliDoc Voice, a professional AI assistant.\n"
        "You may ONLY use the provided tools.\n"
         "You DO NOT have access to web search.\n"
        "If a question contains multiple parts, you MUST answer all parts.\n"
        "You may call multiple tools when necessary.\n"
        "After receiving tool results, combine them into one clear and natural final response.\n"
        "Do NOT repeat tool results verbatim unless needed.\n"
        "Do NOT ask follow-up questions unless the user explicitly requests clarification.\n"
        "Keep responses concise and professional."
             ),
        },
        {"role": "user", "content": user_query},
    ]

    for _ in range(5):   # 🔒 limit to 5 tool loops max
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )

        message = response.choices[0].message
        messages.append(message)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                if tool_name == "answer_from_document":
                    result = answer_from_document(arguments["query"])
                elif tool_name == "calculator":
                    result = calculator(arguments["expression"])
                elif tool_name == "get_current_time":
                    result = get_current_time()
                else:
                    result = "Unknown tool."

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

                # 🔹 Force model to produce final answer
                messages.append({
                    "role": "user",
                    "content": "Using the tool result above, provide the final answer to the original question."
                })

            continue

        return message.content

    return "Agent stopped due to too many tool calls."

if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        answer = run_agent(query)
        print("\nAnswer:\n")
        print(answer)