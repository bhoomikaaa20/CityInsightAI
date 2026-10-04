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

res=weather_tool.invoke("Dubai")
print(res)