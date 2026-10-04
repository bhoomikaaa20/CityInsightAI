from dotenv import load_dotenv
load_dotenv()

import os
import requests

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call
from tavily import TavilyClient


# -----------------------------
# WEATHER TOOL
# -----------------------------

@tool
def weather_tool(city: str) -> str:
    """Find the weather in a city."""

    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={api_key}&units=metric"
    )

    response = requests.get(url)
    data = response.json()

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {temp} Celsius, {desc}"


# -----------------------------
# NEWS TOOL
# -----------------------------

@tool
def news_tool(city: str) -> str:
    """Fetch latest news about a city."""

    tavily = TavilyClient(
        api_key=os.getenv("TAVILY_API_KEY")
    )

    response = tavily.search(
        query=f"Latest news about {city}",
        topic="news",
        max_results=3
    )

    news = ""

    for result in response["results"]:
        news += f"Headline: {result['title']}\n"
        news += f"URL: {result['url']}\n"
        news += f"About: {result['content'][:150]}...\n\n"

    return news


# -----------------------------
# TOOL WRAPPER / MIDDLEWARE
# -----------------------------

@wrap_tool_call
def tool_wrapper(request, handler):

    tool_name = request.tool_call["name"]
    tool_args = request.tool_call["args"]

    print(f"\nAgent wants to use: {tool_name}")
    print(f"Arguments: {tool_args}")

    confirm = input("Do you want to allow this tool? (yes/no): ")

    if confirm.lower() == "yes":
        return handler(request)  #It means ->Continue with what LangChain normally does with this tool call.

    return "Tool call denied by the user."


# -----------------------------
# LLM
# -----------------------------

llm = ChatOpenAI(model="gpt-5-nano")


# -----------------------------
# PRE-BUILT AGENT
# -----------------------------

agent = create_agent(
    model=llm,
    tools=[weather_tool, news_tool],
    middleware=[tool_wrapper]
)


# -----------------------------
# LOOP
# -----------------------------

while True:

    user = input("You: ")

    if user.lower() == "exit":
        break

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user
                }
            ]
        }
    )

    print("\nAgent:", result["messages"][-1].content)