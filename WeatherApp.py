import streamlit as st
import requests
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from datetime import datetime
from collections import defaultdict

# API Keys
weather_api_key = '73e5ff9e5e82f199a767ef2ba8c1c822'  # Replace with your actual API key
unsplash_access_key = 'kMy2ZApb1zOce2F9VzQGXWQDszIJSlqKWHhf3Y5n_dw'  # Replace with your Unsplash access key

# API URLs
current_weather_api_url = f'https://api.openweathermap.org/data/2.5/weather?q={{}}&appid={weather_api_key}&units=metric'
forecast_api_url = f'https://api.openweathermap.org/data/2.5/forecast?q={{}}&appid={weather_api_key}&units=metric'
unsplash_api_url = f'https://api.unsplash.com/search/photos?query={{}}&client_id={unsplash_access_key}'

# Hugging Face model settings
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

def get_weather(location):
    url = current_weather_api_url.format(location)
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def get_forecast(location):
    url = forecast_api_url.format(location)
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def get_image(weather_condition):
    query = f"weather {weather_condition}"
    url = unsplash_api_url.format(query)
    response = requests.get(url)
    if response.status_code == 200:
        images = response.json().get('results', [])
        if images:
            return images[0]['urls']['regular']
    return None

def summarize_text(input_text):
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="Provide a concise summary of the following weather information in 3-4 lines:\n\n{text}\n\nSummary:",
    )
    llm = HuggingFacePipeline(pipeline=summarizer)
    formatted_prompt = prompt_template.template.format(text=input_text)
    generated_result = llm(prompt=formatted_prompt, max_length=100)
    if isinstance(generated_result, str):
        return generated_result
    elif isinstance(generated_result, list) and len(generated_result) > 0:
        return generated_result[0]['generated_text']
    else:
        return None

def process_forecast_data(forecast_data):
    daily_forecasts = defaultdict(lambda: {"description": "", "temp": 0, "humidity": 0, "wind_speed": 0, "icon": "", "count": 0})
    
    for entry in forecast_data['list']:
        dt = datetime.fromtimestamp(entry['dt'])
        day = dt.date()
        daily_forecasts[day]["description"] = entry['weather'][0]['description']
        daily_forecasts[day]["temp"] += entry['main']['temp']
        daily_forecasts[day]["humidity"] += entry['main']['humidity']
        daily_forecasts[day]["wind_speed"] += entry['wind']['speed']
        daily_forecasts[day]["icon"] = entry['weather'][0]['icon']
        daily_forecasts[day]["count"] += 1

    for day, data in daily_forecasts.items():
        data["temp"] /= data["count"]
        data["humidity"] /= data["count"]
        data["wind_speed"] /= data["count"]

    return daily_forecasts

def main():
    st.set_page_config(layout="wide")

    st.sidebar.title("Weather Summarizer")
    location = st.sidebar.text_input("Enter location (e.g., city name):")
    get_weather_btn = st.sidebar.button("Get Weather and Summarize")

    if get_weather_btn:
        with st.spinner("Fetching weather data..."):
            weather_data = get_weather(location)
            forecast_data = get_forecast(location)
        
        if weather_data and forecast_data:
            weather_condition = weather_data['weather'][0]['description']
            weather_summary = (
                f"Weather in {weather_data['name']} - "
                f"{weather_condition}, "
                f"temperature: {weather_data['main']['temp']}°C, "
                f"humidity: {weather_data['main']['humidity']}%, "
                f"wind speed: {weather_data['wind']['speed']} m/s"
            )
            image_url = get_image(weather_condition)
            summary = summarize_text(weather_summary)

            daily_forecasts = process_forecast_data(forecast_data)
            # Sort by date and limit to 4 days
            sorted_forecasts = sorted(daily_forecasts.items())[:4]

            col1, col2 = st.columns([3, 1])
            with col1:
                st.header(f"Weather in {weather_data['name']}")
                st.write(f"**Condition**: {weather_condition.capitalize()}")
                st.write(f"**Temperature**: {weather_data['main']['temp']}°C")
                st.write(f"**Humidity**: {weather_data['main']['humidity']}%")
                st.write(f"**Wind Speed**: {weather_data['wind']['speed']} m/s")
                st.subheader("Summarized Report")
                st.write(summary)

                st.subheader("4-Day Forecast")
                forecast_cols = st.columns(4)
                for i, (day, data) in enumerate(sorted_forecasts):
                    with forecast_cols[i]:
                        st.write(f"**Date**: {day}")
                        st.write(f"**Condition**: {data['description'].capitalize()}")
                        st.write(f"**Temperature**: {data['temp']:.1f}°C")
                        st.write(f"**Humidity**: {data['humidity']:.1f}%")
                        st.write(f"**Wind Speed**: {data['wind_speed']:.1f} m/s")
                        icon_url = f"http://openweathermap.org/img/wn/{data['icon']}@2x.png"
                        st.image(icon_url, width=50)  # Adjust width as needed

            with col2:
                if image_url:
                    st.image(image_url, caption=f"{weather_condition.capitalize()} in {weather_data['name']}")
                weather_icon_url = f"http://openweathermap.org/img/wn/{weather_data['weather'][0]['icon']}@2x.png"
                st.image(weather_icon_url, caption=f"Weather icon for {weather_condition}")

        else:
            st.error("Failed to fetch weather data. Please check the location or try again later.")
    else:
        st.title("Welcome to the Weather Summarizer!")
        st.write("Enter a location in the sidebar to get the current weather and a summarized report.")

if __name__ == "__main__":
    main()
