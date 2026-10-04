from dotenv import load_dotenv
load_dotenv()

import os
import requests

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage,SystemMessage,AIMessage
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

messages=[]


# --------------------------------------------------
# AGENT
# --------------------------------------------------

agent=create_agent(
    llm,
    tools=[weather_tool,news_tool],
    system_prompt="You are a helpful City Assistant"
    
    )

print("City Agent:Type 'exit' for exiting")

while agent:
    user_input=input("You:")
    if user_input.lower()=="exit":
        break
    messages.append(user_input)

    result=agent.invoke(
        {"messages":[{"role":"user","content":user_input}]}
    )

    print(result["messages"][-1].content)


