from dotenv import load_dotenv
load_dotenv()

import os
import requests

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from rich import print


# --------------------------------------------------
# WEATHER TOOL
# --------------------------------------------------

@tool
def weather_tool(city: str) -> str:
    """Find the weather in the given city."""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={api_key}&units=metric"
    )

    response = requests.get(url)
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {temp} Celsius, {desc}"


# --------------------------------------------------
# NEWS TOOL
# --------------------------------------------------

@tool
def news_tool(city: str) -> str:
    """Fetch the latest news of the given city."""

    tavily_client = TavilyClient(
        api_key=os.getenv("TAVILY_API_KEY")
    )

    response = tavily_client.search(
        query=f"Latest news of {city}",
        topic="news",
        search_depth="basic",
        max_results=3
    )

    news = ""

    for result in response["results"]:
        news += f"Headline: {result['title']}\n"
        news += f"URL: {result['url']}\n"
        news += f"About: {result['content'][:150]}...\n\n"

    return news


# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatOpenAI(model="gpt-5-nano")

tools = {
    "weather_tool": weather_tool,
    "news_tool": news_tool
}

llm_with_tools = llm.bind_tools(
    [weather_tool, news_tool]
)


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

while True:

    user = input("\nYou: ")

    # Exit
    if user.lower() in ["exit", "quit", "bye"]:
        print("Agent: Goodbye!")
        break

    messages = []

    # Human message
    messages.append(
        HumanMessage(content=user)
    )

    # First LLM call
    result = llm_with_tools.invoke(messages)

    messages.append(result)

    # Tool loop
    while result.tool_calls:

        for tool_call in result.tool_calls:

            tool_name = tool_call["name"]

            # HUMAN IN THE LOOP
            confirm = input(
                f"Agent wants to call {tool_name}. "
                "Type yes to confirm or no to decline: "
            )

            if confirm.lower() == "no":
                print("Tool call denied.")
                continue

            # Execute tool
            tool_result = tools[tool_name].invoke(
                tool_call["args"]
            )

            # Add ToolMessage
            messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"]
                )
            )

        # Send tool result back to LLM
        result = llm_with_tools.invoke(messages)

        messages.append(result)

    # Final response
    print("\nAgent:", result.content)