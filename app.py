import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from fredapi import Fred
import yfinance as yf
import requests
import datetime
import numpy as np

# Replace with your API keys
FRED_API_KEY = '6a26c495cb433d78e08bd4aa79b67616'    # FRED API key (Replace w your Key here)

# Initialize APIs
fred = Fred(api_key=FRED_API_KEY)

# Function to fetch historical GDP data from FRED
def get_historical_gdp():
    try:
        # Get quarterly GDP data from 1990 to present
        gdp = fred.get_series('GDP', observation_start='1990-01-01')  # Quarterly GDP in billions
        return gdp
    except Exception as e:
        print(f"Error fetching GDP data: {e}")
        # Fallback data (example)
        dates = pd.date_range(start='1990-01-01', end='2024-12-31', freq='Q')
        return pd.Series([5000 + i*200 for i in range(len(dates))], index=dates)

# Function to fetch historical market cap data
def get_historical_market_cap():
    try:
        # Use Wilshire 5000 Total Market Index (^W5000) from yfinance
        # This is a better proxy for total US market cap
        wilshire = yf.Ticker("^W5000")
        wilshire_data = wilshire.history(period="max", start="1990-01-01")['Close']
        
        # Convert Wilshire 5000 index to market cap (in billions)
        # The conversion factor varies over time, but approximately 1 point = $1.05 billion
        factor = 1.05
        market_cap = wilshire_data * factor
        return market_cap
    except Exception as e:
        print(f"Error fetching market cap data: {e}")
        # Fallback data (example)
        dates = pd.date_range(start='1990-01-01', end=pd.Timestamp.now(), freq='D')
        return pd.Series([4000 + i*5 for i in range(len(dates))], index=dates)

# Function to create historical Buffett Indicator ratio
def create_historical_ratio():
    gdp = get_historical_gdp()
    market_cap = get_historical_market_cap()
    
    # Print debug information
    print(f"Latest GDP date: {gdp.index[-1]}")
    print(f"Latest Market Cap date: {market_cap.index[-1]}")
    
    # Convert market_cap index to timezone-naive if it's timezone-aware
    if market_cap.index.tzinfo is not None:
        market_cap.index = market_cap.index.tz_localize(None)
    
    # Resample GDP to daily frequency, forward fill
    gdp_daily = gdp.resample('D').ffill()
    
    # Align dates
    common_dates = market_cap.index.intersection(gdp_daily.index)
    
    # Print last common date
    print(f"Last common date in data: {common_dates[-1]}")
    
    # Calculate ratio as percentage
    ratio = (market_cap[common_dates] / gdp_daily[common_dates]) * 100
    
    # Sort by date to ensure chronological order
    ratio = ratio.sort_index()
    
    return ratio

# Function to normalize the Buffett Indicator to standard deviations from mean
def normalize_ratio(ratio):
    mean = ratio.mean()
    std = ratio.std()
    normalized = (ratio - mean) / std
    return normalized, mean, std

# Function to detrend the Buffett Indicator
def detrend_ratio(ratio):
    # Create a time index for regression
    time_index = np.arange(len(ratio))
    
    # Fit a linear trend line
    coefficients = np.polyfit(time_index, ratio.values, 1)
    trend = np.polyval(coefficients, time_index)
    
    # Remove the trend
    detrended = ratio.values - trend
    
    # Convert to standard deviations
    mean_detrended = np.mean(detrended)
    std_detrended = np.std(detrended)
    normalized_detrended = (detrended - mean_detrended) / std_detrended
    
    # Create a Series with the original index
    detrended_series = pd.Series(normalized_detrended, index=ratio.index)
    
    return detrended_series, coefficients

# Function to get current market cap
def get_current_market_cap():
    try:
        # Get the most recent value using yfinance
        wilshire = yf.Ticker("^W5000")
        # Get last 5 days of data to ensure we have some recent data
        latest_data = wilshire.history(period="5d")
        if len(latest_data) == 0:
            raise Exception("No recent data available")
        latest_value = latest_data['Close'].iloc[-1]  # Get most recent available closing price
        # Convert to market cap in billions
        return latest_value * 1.05  # Using the same conversion factor
    except Exception as e:
        print(f"Error fetching current market cap: {e}")
        # Use the last value from historical data instead
        historical_cap = get_historical_market_cap()
        return historical_cap.iloc[-1]

# Function to get current Buffett Indicator ratio
def get_current_ratio():
    latest_gdp = get_historical_gdp().iloc[-1]  # Latest quarterly GDP
    current_market_cap = get_current_market_cap()
    return (current_market_cap / latest_gdp) * 100  # As percentage


# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # Add this line for deployment

# Define app layout
app.layout = html.Div([
    html.H1("Buffett Indicator (1990-Present)"),
    html.H3("Total US Market Cap to GDP Ratio"),
    dcc.Graph(id='buffett-indicator-graph'),
    html.H3("Detrended Buffett Indicator"),
    html.P("This chart shows the Buffett Indicator with the long-term trend removed, highlighting cyclical patterns. Values are shown in standard deviations from the trend."),
    dcc.Graph(id='normalized-buffett-indicator-graph'),
    html.Div([
        html.P("Data updates every minute"),
        html.P("Source: FRED Economic Data")
    ]),
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # Update every minute (in milliseconds)
        n_intervals=0
    ),
    html.Footer([
        html.Hr(),
        html.P(
            ["© Copyright 2025 ", 
             html.A("Qichen Huang", href="https://qichenhuang.com", target="_blank")],
            style={'textAlign': 'center', 'margin': '20px 0', 'color': '#666'}
        )
    ])
])

# Callback to update the graphs
@app.callback(
    [Output('buffett-indicator-graph', 'figure'),
     Output('normalized-buffett-indicator-graph', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    # Get historical and current data
    ratio = create_historical_ratio()  # 重点 
    current_ratio = get_current_ratio()
    current_date = pd.Timestamp.now()
    
    # Normalize the ratio (for fig1)
    normalized_ratio, mean_ratio, std_ratio = normalize_ratio(ratio)
    current_normalized = (current_ratio - mean_ratio) / std_ratio

    # Detrend the ratio (for fig2)
    detrended_ratio, trend_coefficients = detrend_ratio(ratio)
    
    # Calculate current detrended value
    time_index_current = len(ratio)
    trend_current = np.polyval(trend_coefficients, time_index_current)
    detrended_current = (current_ratio - trend_current)
    mean_detrended = np.mean(ratio.values - np.polyval(trend_coefficients, np.arange(len(ratio))))
    std_detrended = np.std(ratio.values - np.polyval(trend_coefficients, np.arange(len(ratio))))
    current_detrended_normalized = (detrended_current - mean_detrended) / std_detrended

    # Define valuation zones
    # Based on historical averages and Warren Buffett's comments
    strongly_undervalued = 80
    undervalued = 90
    fairly_valued = 115
    overvalued = 150
    
    # Create Plotly figure for regular Buffett Indicator
    fig1 = go.Figure()
    
    # Add valuation zones
    fig1.add_shape(type="rect", x0=ratio.index[0], x1=current_date, y0=0, y1=strongly_undervalued,
                 fillcolor="green", opacity=0.2, layer="below", line_width=0)
    fig1.add_shape(type="rect", x0=ratio.index[0], x1=current_date, y0=strongly_undervalued, y1=undervalued,
                 fillcolor="lightgreen", opacity=0.2, layer="below", line_width=0)
    fig1.add_shape(type="rect", x0=ratio.index[0], x1=current_date, y0=undervalued, y1=fairly_valued,
                 fillcolor="yellow", opacity=0.2, layer="below", line_width=0)
    fig1.add_shape(type="rect", x0=ratio.index[0], x1=current_date, y0=fairly_valued, y1=overvalued,
                 fillcolor="orange", opacity=0.2, layer="below", line_width=0)
    fig1.add_shape(type="rect", x0=ratio.index[0], x1=current_date, y0=overvalued, y1=max(ratio.max(), 200),
                 fillcolor="red", opacity=0.2, layer="below", line_width=0)
    
    # Historical ratio line
    fig1.add_trace(go.Scatter(
        x=ratio.index,
        y=ratio.values,
        mode='lines',
        name='Historical Ratio',
        line=dict(color='blue', width=2),
        hovertemplate='<b>Date</b>: %{x}<br><b>Ratio</b>: %{y:.1f}%<extra></extra>'
    ))
    
    # Current ratio marker
    fig1.add_trace(go.Scatter(
        x=[current_date],
        y=[current_ratio],
        mode='markers',
        name=f'Current: {current_ratio:.1f}%',
        marker=dict(color='red', size=12, symbol='star'),
        hovertemplate='<b>Current Ratio</b>: %{y:.1f}%<extra></extra>'
    ))
    
    # Add annotations for valuation zones
    fig1.add_annotation(x=ratio.index[0], y=strongly_undervalued/2, 
                      text="Strongly Undervalued", showarrow=False, font=dict(color="white"))
    fig1.add_annotation(x=ratio.index[0], y=(strongly_undervalued+undervalued)/2, 
                      text="Undervalued", showarrow=False)
    fig1.add_annotation(x=ratio.index[0], y=(undervalued+fairly_valued)/2, 
                      text="Fair Value", showarrow=False)
    fig1.add_annotation(x=ratio.index[0], y=(fairly_valued+overvalued)/2, 
                      text="Overvalued", showarrow=False)
    fig1.add_annotation(x=ratio.index[0], y=overvalued+20, 
                      text="Strongly Overvalued", showarrow=False)

    # Update layout
    fig1.update_layout(
        title='Buffett Indicator: Total US Market Cap / GDP (1990-Present)',
        xaxis_title='Date',
        yaxis_title='Ratio (%)',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        )
    )
    
    # Create Plotly figure for detrended Buffett Indicator
    fig2 = go.Figure()
    
    # Add standard deviation bands
    fig2.add_shape(type="rect", x0=detrended_ratio.index[0], x1=current_date, y0=-3, y1=-2,
                 fillcolor="green", opacity=0.2, layer="below", line_width=0)
    fig2.add_shape(type="rect", x0=detrended_ratio.index[0], x1=current_date, y0=-2, y1=-1,
                 fillcolor="lightgreen", opacity=0.2, layer="below", line_width=0)
    fig2.add_shape(type="rect", x0=detrended_ratio.index[0], x1=current_date, y0=-1, y1=1,
                 fillcolor="yellow", opacity=0.2, layer="below", line_width=0)
    fig2.add_shape(type="rect", x0=detrended_ratio.index[0], x1=current_date, y0=1, y1=2,
                 fillcolor="orange", opacity=0.2, layer="below", line_width=0)
    fig2.add_shape(type="rect", x0=detrended_ratio.index[0], x1=current_date, y0=2, y1=3,
                 fillcolor="red", opacity=0.2, layer="below", line_width=0)
    
    # Add standard deviation lines
    for i in range(-3, 4):
        fig2.add_shape(
            type="line", x0=detrended_ratio.index[0], x1=current_date, y0=i, y1=i,
            line=dict(color="gray", width=1, dash="dash")
        )
        if i != 0:
            fig2.add_annotation(x=detrended_ratio.index[0], y=i, 
                              text=f"{i} σ", showarrow=False, xanchor="left")
    
    # Add mean line
    fig2.add_shape(
        type="line", x0=detrended_ratio.index[0], x1=current_date, y0=0, y1=0,
        line=dict(color="black", width=2)
    )
    fig2.add_annotation(x=detrended_ratio.index[0], y=0, 
                      text="Mean", showarrow=False, xanchor="left")
    
    # Historical detrended ratio line
    fig2.add_trace(go.Scatter(
        x=detrended_ratio.index,
        y=detrended_ratio.values,
        mode='lines',
        name='Detrended SD from Trend',
        line=dict(color='blue', width=2),
        hovertemplate='<b>Date</b>: %{x}<br><b>SD from Trend</b>: %{y:.2f} σ<extra></extra>'
    ))
    
    # Current detrended ratio marker
    fig2.add_trace(go.Scatter(
        x=[current_date],
        y=[current_detrended_normalized],
        mode='markers',
        name=f'Current: {current_detrended_normalized:.2f} σ',
        marker=dict(color='red', size=12, symbol='star'),
        hovertemplate='<b>Current</b>: %{y:.2f} σ<extra></extra>'
    ))
    
    # Update layout
    fig2.update_layout(
        title='Detrended Buffett Indicator: Standard Deviations from Trend',
        xaxis_title='Date',
        yaxis_title='Standard Deviations from Trend (σ)',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(range=[-3.5, 3.5]),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        )
    )

    return fig1, fig2

# Run the app
if __name__ == '__main__':
    app.run(debug=True)


