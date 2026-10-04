from dotenv import load_dotenv
load_dotenv()
import os
import requests

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage,ToolMessage,AIMessage
from tavily import TavilyClient
from rich import print



#weather tool
@tool
def weather_tool(city:str)->str:
    """Find the weather in the city given"""
    api_key=os.getenv("OPENWEATHER_API_KEY")
    url=(
        f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    )

    response=requests.get(url)
    data=response.json()

    print(data)


    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp=data["main"]["temp"]
    desc=data["weather"][0]["description"]

    return f"weather in {city}: {temp} celcius,{desc}"


#news tool

@tool
def get_latest_news(city:str)->str:
    """Fetch the latest news of the city given"""

    tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
    )

    response=tavily_client.search(
        query=f"Latest news of the {city}",
        topic="news",
        search_depth="basic",
        max_results=3
    )
    results = response["results"]

    news = ""

    for result in response["results"]:
        news += f"Headline: {result['title']}\n"
        news += f"URL: {result['url']}\n"
        news += f"About: {result['content'][:150]}...\n\n"


    return news



