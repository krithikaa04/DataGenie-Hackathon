
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go


def plot_graph(data):
    result_data = data["result"]
    df = pd.DataFrame(result_data)

    df["point_timestamp"] = pd.to_datetime(df["point_timestamp"])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["point_timestamp"],
            y=df["point_value"],
            mode="lines+markers",
            name="Actual",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["point_timestamp"],
            y=df["yhat"],
            mode="lines+markers",
            name="Predicted",
        )
    )

    fig.update_layout(
        title=f"Forecastability Score: {data['forecastScore']} - MAPE: {data['mape']}",
        xaxis_title="Date",
        yaxis_title="Value",
        showlegend=True,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
    )

    fig.update_xaxes(rangeslider_visible=True)

    return fig


def trend_seasonality_plot(data):
    result_data = data["result"]
    df = pd.DataFrame(result_data)
    df["point_timestamp"] = pd.to_datetime(df["point_timestamp"])
    dates = df["point_timestamp"]
    values = df['point_value']
    window = 7 if len(values)>7 else len(values)-1
    trend = values.rolling(window=window, min_periods=1, center=True).mean()
    seasonal = values - trend
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=trend, mode='lines', name='Trend'))
    fig.add_trace(go.Scatter(x=dates, y=seasonal, mode='lines', name='Seasonal Component'))
    fig.update_layout(title='Trend and Seasonality', xaxis_title='Date', yaxis_title='Value')
    return fig