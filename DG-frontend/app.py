import json

import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import requests
import streamlit as st
from plotsUtils import plot_graph, trend_seasonality_plot

st.set_page_config(page_title="DataGenie Hackathon", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        max-width: 100%;
        padding: 1rem;
    }
    .stButton>button {
        background-color: #008B8B;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("POSTMAN-LIKE API CALL")

format = st.selectbox(
    "Format",
    ("daily", "monthly", "hourly", "weekly"),
)
json_file = st.file_uploader("Upload JSON File", type=["json"])
from_date = st.date_input("Start Date")
to_date = st.date_input("End Date")
period = st.number_input("Period", min_value=0, step=1)


if st.button("Do you want response?"):
    if format and from_date and to_date and json_file:
        try:
            json_content_bytes = json_file.read()
            json_content_str = json_content_bytes.decode("ascii")
            request_data = json.loads(json_content_str)
            API_ENDPOINT = "http://127.0.0.1:8000/predict?format={}&from_date={}&to_date={}&period={}".format(
                format, from_date, to_date, period
            )
            response = requests.post(API_ENDPOINT, json=request_data).json()
            st.subheader("API Response:")
            st.json(response)
            
            st.subheader("Actual vs Forecasted charts")
            st.plotly_chart(plot_graph(response), use_container_width=True)

            st.subheader("Trend and Seasonality Plots")
            st.plotly_chart(trend_seasonality_plot(response), use_container_width=True)
            
        except Exception as e:
            st.write("hello")
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please Fill all the values")