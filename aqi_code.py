import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import requests
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from typing import List, Dict, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import joblib
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AirQualityPredictor:
    
    def __init__(self, cities: List[Dict[str, float]]):
        """
        Initialize the air quality prediction system
        
        :param cities: List of city dictionaries with lat, lon, and name
        """
        self.cities = cities
        self.data_cache = {}
        self.scaler = MinMaxScaler()
        self.model = None
        self.graph = self._create_city_graph()
        self.evaluation_metrics = {}
        
        # Create cache directory if not exists
        os.makedirs('data_cache', exist_ok=True)
    
    def _create_city_graph(self) -> nx.Graph:
        """
        Create a graph representing cities based on geographic proximity
        
        :return: NetworkX graph of cities
        """
        G = nx.Graph()
        
        # Add nodes with city information
        for city in self.cities:
            G.add_node(city['name'], 
                       pos=(city['lon'], city['lat']), 
                       lat=city['lat'], 
                       lon=city['lon'])
        
        # Connect cities based on geographic proximity (threshold-based)
        for i, city1 in enumerate(self.cities):
            for city2 in self.cities[i+1:]:
                distance = self._haversine_distance(
                    city1['lat'], city1['lon'], 
                    city2['lat'], city2['lon']
                )
                # Connect cities within 500 km
                if distance <= 500:
                    G.add_edge(city1['name'], city2['name'], weight=1/distance)
        
        return G
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2) -> float:
        """
        Calculate great circle distance between two points on earth
        
        :return: Distance in kilometers
        """
        R = 6371  # Earth's radius in kilometers
        
        # Convert latitude and longitude to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def fetch_air_quality_data(self, days: int = 90) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical air quality and weather data for all cities.

        :param days: Number of historical days to fetch.
        :return: Dictionary of DataFrames for each city.
        """
        def _fetch_city_data(city):
            # Check cache first
            cache_file = f'data_cache/{city["name"]}_air_quality.joblib'
            if os.path.exists(cache_file):
                return joblib.load(cache_file)

            # Air quality API URL
            air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
            # Weather API URL
            weather_url = "https://api.open-meteo.com/v1/forecast"

            try:
                # Air quality parameters
                air_params = {
                    "latitude": city['lat'],
                    "longitude": city['lon'],
                    "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
                    "past_days": days,
                    "timezone": "Asia/Kolkata"
                }

                # Weather parameters
                weather_params = {
                    "latitude": city['lat'],
                    "longitude": city['lon'],
                    "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m",
                    "past_days": days,
                    "timezone": "Asia/Kolkata"
                }

                # Fetch air quality data
                air_response = requests.get(air_url, params=air_params, timeout=10)
                air_response.raise_for_status()
                air_data = air_response.json()

                # Fetch weather data
                weather_response = requests.get(weather_url, params=weather_params, timeout=10)
                weather_response.raise_for_status()
                weather_data = weather_response.json()

                # Ensure required keys exist in air quality data
                required_air_keys = ['time', 'pm10', 'pm2_5', 'us_aqi', 'carbon_monoxide', 
                                    'nitrogen_dioxide', 'sulphur_dioxide', 'ozone']
                if not all(key in air_data['hourly'] for key in required_air_keys):
                    raise ValueError(f"Incomplete air quality data received for {city['name']}")

                # Ensure required keys exist in weather data
                required_weather_keys = ['time', 'temperature_2m', 'relativehumidity_2m', 'windspeed_10m']
                if not all(key in weather_data['hourly'] for key in required_weather_keys):
                    raise ValueError(f"Incomplete weather data received for {city['name']}")

                # Convert air quality response to DataFrame
                air_df = pd.DataFrame({
                    'timestamp': pd.to_datetime(air_data['hourly']['time']),
                    'pm10': [float(x) if x is not None else np.nan for x in air_data['hourly']['pm10']],
                    'pm2_5': [float(x) if x is not None else np.nan for x in air_data['hourly']['pm2_5']],
                    'co': [float(x) if x is not None else np.nan for x in air_data['hourly']['carbon_monoxide']],
                    'no2': [float(x) if x is not None else np.nan for x in air_data['hourly']['nitrogen_dioxide']],
                    'so2': [float(x) if x is not None else np.nan for x in air_data['hourly']['sulphur_dioxide']],
                    'o3': [float(x) if x is not None else np.nan for x in air_data['hourly']['ozone']],
                    'us_aqi': [float(x) if x is not None else np.nan for x in air_data['hourly']['us_aqi']]
                })

                # Convert weather response to DataFrame
                weather_df = pd.DataFrame({
                    'timestamp': pd.to_datetime(weather_data['hourly']['time']),
                    'temperature': [float(x) if x is not None else np.nan for x in weather_data['hourly']['temperature_2m']],
                    'humidity': [float(x) if x is not None else np.nan for x in weather_data['hourly']['relativehumidity_2m']],
                    'wind_speed': [float(x) if x is not None else np.nan for x in weather_data['hourly']['windspeed_10m']]
                })

                # Merge the two dataframes on timestamp
                df = pd.merge(air_df, weather_df, on='timestamp')

                # Handle missing values
                df = df.dropna()

                # Cache the fetched data
                joblib.dump(df, cache_file)

                logger.info(f"Successfully fetched data for {city['name']}")
                return df

            except requests.exceptions.RequestException as e:
                logger.error(f"API Request Error for {city['name']}: {e}")
            except ValueError as e:
                logger.error(f"Data Format Error for {city['name']}: {e}")

            # If API fails, use fallback
            if os.path.exists(cache_file):
                logger.warning(f"Using cached data for {city['name']}")
                return joblib.load(cache_file)

            # Generate synthetic data if cache is unavailable
            logger.warning(f"Generating synthetic data for {city['name']}")
            df = pd.DataFrame({
                'timestamp': pd.date_range(end=pd.Timestamp.now(), periods=days*24, freq='h'),
                'pm10': np.random.normal(50, 20, days*24),
                'pm2_5': np.random.normal(25, 10, days*24),
                'co': np.random.normal(600, 200, days*24),
                'no2': np.random.normal(40, 15, days*24),
                'so2': np.random.normal(20, 10, days*24),
                'o3': np.random.normal(60, 20, days*24),
                'us_aqi': np.random.normal(100, 30, days*24),
                'temperature': np.random.normal(25, 5, days*24),
                'humidity': np.random.normal(60, 15, days*24),
                'wind_speed': np.random.normal(10, 5, days*24)
            })
            return df

        # Fetch data for all cities
        self.data_cache = {city['name']: _fetch_city_data(city) for city in self.cities}
        return self.data_cache

    def preprocess_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess data into graph-friendly tensors
        
        :return: Node features and adjacency matrix
        """
        # Aggregate features for each city
        node_features = []
        for city_name, df in self.data_cache.items():
            # Aggregate features (last 30 days statistics)
            city_features = [
                df['pm10'].mean(),
                df['pm2_5'].mean(),
                df['us_aqi'].mean(),
                df['temperature'].mean(),
                df['humidity'].mean(),
                df['wind_speed'].mean(),
                df['co'].mean(),
                df['no2'].mean(),
                df['so2'].mean(),
                df['o3'].mean()
            ]
            node_features.append(city_features)
        
        # Normalize features
        node_features = self.scaler.fit_transform(node_features)
        
        # Create adjacency matrix from graph
        adj_matrix = nx.adjacency_matrix(self.graph, 
                                         nodelist=[city['name'] for city in self.cities]).todense()
        
        return (torch.FloatTensor(node_features), 
                torch.FloatTensor(adj_matrix))
    
    class GraphAttentionLSTM(nn.Module):

        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            # Graph Attention Layer
            self.graph_attn = nn.Linear(input_dim * 2, 1)
            
            # LSTM for temporal modeling
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            
            # Output layer
            self.fc = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x, adj_matrix):
            # Graph Attention Mechanism
            node_count = x.size(0)
            attention_scores = torch.zeros((node_count, node_count))
            
            for i in range(node_count):
                for j in range(node_count):
                    if adj_matrix[i, j] > 0:
                        # Concatenate node features
                        combined = torch.cat([x[i], x[j]])
                        attention_scores[i, j] = F.leaky_relu(self.graph_attn(combined))
            
            # Normalize attention scores
            attention_scores = F.softmax(attention_scores, dim=1)
            
            # Apply attention
            x_attended = torch.matmul(attention_scores, x)
            
            # Temporal modeling with LSTM
            x_lstm, _ = self.lstm(x_attended.unsqueeze(0))
            
            # Predict next timestep
            output = self.fc(x_lstm.squeeze(0))
            
            return output
    
    def train_model(self, node_features, adj_matrix):
        """
        Train Graph Attention LSTM model
        """
        # Model hyperparameters
        input_dim = node_features.shape[1]
        hidden_dim = 64
        output_dim = input_dim  # Predicting same features
        
        # Initialize model
        self.model = self.GraphAttentionLSTM(input_dim, hidden_dim, output_dim)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        # Training loop
        for epoch in range(100):
            # Forward pass
            predictions = self.model(node_features, adj_matrix)
            
            # Compute loss
            loss = criterion(predictions, node_features)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluate model after training
        self._evaluate_model(node_features, adj_matrix)
    
    def _evaluate_model(self, node_features, adj_matrix):
        """
        Evaluate model and calculate performance metrics
        """
        with torch.no_grad():
            predictions = self.model(node_features, adj_matrix)
        
        # Inverse transform to get actual values
        actual = self.scaler.inverse_transform(node_features.numpy())
        predicted = self.scaler.inverse_transform(predictions.numpy())
        
        # Calculate metrics for each city
        for i, city in enumerate(self.cities):
            city_name = city['name']
            
            # Calculate metrics
            rmse = math.sqrt(mean_squared_error(actual[i], predicted[i]))
            mse = mean_squared_error(actual[i], predicted[i])
            mae = mean_absolute_error(actual[i], predicted[i])
            
            # Handle potential division by zero in MAPE calculation
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = mean_absolute_percentage_error(actual[i], predicted[i]) * 100
            
            # If mape contains infinity, replace with a large value
            if np.isinf(mape) or np.isnan(mape):
                mape = 100.0
                
            r2 = r2_score(actual[i], predicted[i])
            
            self.evaluation_metrics[city_name] = {
                'RMSE': rmse,
                'MSE': mse,
                'MAE': mae,
                'MAPE': mape,
                'R2': r2,
                'actual': actual[i].tolist(),
                'predicted': predicted[i].tolist()
            }
    
    def predict(self, days: int = 7) -> Dict[str, List[float]]:
        """
        Predict air quality for the next specified days
        
        :param days: Number of days to predict
        :return: Predictions for each city
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Prepare input
        node_features, adj_matrix = self.preprocess_data()
        
        # Predict
        with torch.no_grad():
            predictions = self.model(node_features, adj_matrix)
        
        # Inverse transform to get actual values
        predictions_np = self.scaler.inverse_transform(predictions.numpy())
        
        # Organize predictions by city
        return {
            city['name']: predictions_np[i].tolist() 
            for i, city in enumerate(self.cities)
        }

def main():
    # Set page configuration to wide mode
    st.set_page_config(layout="wide")
    
    st.title('Urban Air Quality Prediction System')
    
    cities = [
        {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867},
        {"name": "Bangalore", "lat": 12.9716, "lon": 77.5946},
        {"name": "Adilabad", "lat": 19.6748, "lon": 78.5274},
        {"name": "Jogulamba Gadwal", "lat": 16.2340, "lon": 77.8056},
        {"name": "Warangal", "lat": 17.9674, "lon": 79.5889},
        {"name": "Chennai", "lat": 13.0674, "lon": 80.2376}
    ]
    
    # Initialize predictor
    predictor = AirQualityPredictor(cities)
    
    # Fetch historical data
    predictor.fetch_air_quality_data()
    
    # Prepare and train model
    node_features, adj_matrix = predictor.preprocess_data()
    predictor.train_model(node_features, adj_matrix)
    
    # Sidebar for user interaction
    st.sidebar.header('Air Quality Analysis')
    
    # City selection
    selected_cities = st.sidebar.multiselect(
        'Select Cities', 
        [city['name'] for city in cities],
        default=[city['name'] for city in cities]
    )
    
    # Prediction duration
    prediction_days = st.sidebar.slider(
        'Prediction Duration (Days)', 
        min_value=1, 
        max_value=7, 
        value=3
    )
    
    # Pollutant selection
    st.sidebar.subheader('Data Selection')
    selected_pollutants = st.sidebar.multiselect(
        'Select Pollutants',
        ['pm10', 'pm2_5', 'co', 'no2', 'so2', 'o3', 'us_aqi'],
        default=['pm10', 'pm2_5', 'us_aqi']
    )
    
    selected_weather = st.sidebar.multiselect(
        'Select Weather Parameters',
        ['temperature', 'humidity', 'wind_speed'],
        default=['temperature']
    )
    
    # Visualization sections
    st.header('Air Quality Visualizations')
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5= st.tabs([
        'City Network Graph', 
        'AQI Trend Analysis', 
        'Detailed City Insights',
        'Model Evaluation',
        'Air Quality Forecast & Recommendations'
    ])
    
    with tab1:
        st.subheader('City Pollution Interconnection Network')
        
        # Filter graph to only include selected cities
        if selected_cities:
            # Create a subgraph with selected cities
            subgraph = predictor.graph.subgraph(selected_cities)
            
            # Extract city positions for selected cities
            city_positions = {
                city['name']: (city['lon'], city['lat']) 
                for city in predictor.cities if city['name'] in selected_cities
            }
            
            # Create edge traces for the subgraph
            edge_traces = []
            for edge in subgraph.edges():
                x0, y0 = city_positions[edge[0]]
                x1, y1 = city_positions[edge[1]]
                edge_trace = go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none'
                )
                edge_traces.append(edge_trace)
            
            # Create node traces for selected cities
            node_x, node_y, node_text, node_color = [], [], [], []
            for node in subgraph.nodes():
                x, y = city_positions[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                
                # Get the latest AQI value for color coding
                current_aqi = predictor.data_cache[node]['us_aqi'].iloc[-1]
                node_color.append(current_aqi)
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    showscale=True,
                    colorscale='Viridis',
                    size=15,
                    color=node_color,
                    colorbar=dict(
                        thickness=15,
                        title='Latest AQI',
                        xanchor='left',
                        titleside='right'
                    )
                )
            )
            
            # Create figure
            fig = go.Figure(data=edge_traces + [node_trace])
            fig.update_layout(
                title='City Pollution Interconnection Network',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600
            )
            
            # Display the figure
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least one city to display the network graph.")
        
        st.markdown("""
        ### Network Visualization Insights
        - **Node Color**: Represents the current air quality index (AQI) of each city
        - **Edges**: Show geographic and pollution dispersal connections
        - **Size**: Indicates the relative impact of each city's air quality
        """)
    
    with tab2:
        st.subheader('Air Quality Index (AQI) Trends')
        
        # Filter data to selected cities
        if selected_cities:
            # Get predictions for selected duration
            predictions = predictor.predict(days=prediction_days)
            
            # Create figure for AQI trends
            fig = go.Figure()
            
            # Add historical trends for selected cities
            for city in selected_cities:
                data = predictor.data_cache[city]
                fig.add_trace(go.Scatter(
                    x=data['timestamp'], 
                    y=data['us_aqi'], 
                    mode='lines', 
                    name=f'{city} (Historical)'
                ))
                
                # Add prediction lines for selected cities
                if city in predictions:
                    pred_values = predictions[city]
                    # Create prediction timestamps (next specified days)
                    last_timestamp = pd.to_datetime(data['timestamp'].iloc[-1])
                    pred_timestamps = [last_timestamp + pd.Timedelta(days=i+1) for i in range(len(pred_values))]
                    
                    fig.add_trace(go.Scatter(
                        x=pred_timestamps, 
                        y=pred_values, 
                        mode='lines', 
                        name=f'{city} (Predicted)',
                        line=dict(dash='dot')
                    ))
            
            fig.update_layout(
                title='Air Quality Index (AQI) Trends',
                xaxis_title='Date',
                yaxis_title='US Air Quality Index',
                height=500
            )
            
            # Display the figure
            st.plotly_chart(fig, use_container_width=True)
            
            # Display data tables for selected pollutants and weather parameters
            st.subheader('Air Quality and Weather Data')
            
            # Create a table for each selected city
            for city in selected_cities:
                st.write(f"### {city} - Latest Data")
                
                # Get the most recent data
                latest_data = predictor.data_cache[city].tail(24)  # Last 24 hours
                
                # Prepare data for the table
                table_data = {
                    'Timestamp': latest_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                }
                
                # Add selected pollutants
                for pollutant in selected_pollutants:
                    if pollutant in latest_data.columns:
                        table_data[pollutant.upper()] = latest_data[pollutant]
                
                # Add selected weather parameters
                for param in selected_weather:
                    if param in latest_data.columns:
                        display_name = param.capitalize()
                        if param == 'temperature':
                            display_name += ' (°C)'
                        elif param == 'humidity':
                            display_name += ' (%)'
                        elif param == 'wind_speed':
                            display_name += ' (km/h)'
                        table_data[display_name] = latest_data[param]
                
                # Create and display the table
                table_df = pd.DataFrame(table_data)
                st.dataframe(table_df, use_container_width=True)
        else:
            st.warning("Please select at least one city to display AQI trends.")
    
    with tab3:
        st.subheader('Detailed City Air Quality Insights')
        
        if selected_cities:
            # Create rows with 2 cities per row
            num_cities = len(selected_cities)
            rows = [selected_cities[i:i+2] for i in range(0, num_cities, 2)]
            
            for row_cities in rows:
                # Create columns for each row
                cols = st.columns(len(row_cities))
                
                for i, city_name in enumerate(row_cities):
                    with cols[i]:
                        # Fetch historical data for the city
                        city_data = predictor.data_cache[city_name]
                        
                        st.markdown(f"### {city_name} Air Quality")
                        
                        # Display key statistics
                        st.metric("Current AQI", 
                                f"{city_data['us_aqi'].iloc[-1]:.2f}", 
                                f"{city_data['us_aqi'].iloc[-1] - city_data['us_aqi'].iloc[-2]:.2f}")
                        
                        # Display additional metrics in two columns
                        metrics_col1, metrics_col2 = st.columns(2)
                        
                        with metrics_col1:
                            st.metric("PM2.5", 
                                    f"{city_data['pm2_5'].iloc[-1]:.2f}", 
                                    f"{city_data['pm2_5'].iloc[-1] - city_data['pm2_5'].iloc[-2]:.2f}")
                            
                            st.metric("NO2", 
                                    f"{city_data['no2'].iloc[-1]:.2f}", 
                                    f"{city_data['no2'].iloc[-1] - city_data['no2'].iloc[-2]:.2f}")
                        
                        with metrics_col2:
                            st.metric("PM10", 
                                    f"{city_data['pm10'].iloc[-1]:.2f}", 
                                    f"{city_data['pm10'].iloc[-1] - city_data['pm10'].iloc[-2]:.2f}")
                            
                            st.metric("SO2", 
                                    f"{city_data['so2'].iloc[-1]:.2f}", 
                                    f"{city_data['so2'].iloc[-1] - city_data['so2'].iloc[-2]:.2f}")
                        
                        # Mini histogram of AQI
                        fig = px.histogram(city_data, x='us_aqi', 
                                        title=f'{city_name} AQI Distribution',
                                        labels={'us_aqi': 'Air Quality Index'})
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least one city to display detailed insights.")
    
    with tab4:
        st.subheader('Model Evaluation Metrics')
        
        if predictor.evaluation_metrics:
            # Create a table of evaluation metrics
            metrics_data = []
            
            for city_name in selected_cities:
                if city_name in predictor.evaluation_metrics:
                    metrics = predictor.evaluation_metrics[city_name]
                    metrics_data.append({
                        'City': city_name,
                        'RMSE': f"{metrics['RMSE']:.4f}",
                        'MSE': f"{metrics['MSE']:.4f}",
                        'MAE': f"{metrics['MAE']:.4f}",
                        'MAPE (%)': f"{metrics['MAPE']:.2f}",
                        'R²': f"{metrics['R2']:.4f}"
                    })
            
            # Display metrics table
            st.write("### Prediction Accuracy Metrics")
            metrics_df = pd.DataFrame(metrics_data)
            st.table(metrics_df)
            
            # Create actual vs predicted visualizations
            st.write("### Actual vs Predicted Values")
            
            # Display visualizations for selected cities
            for city_name in selected_cities:
                if city_name in predictor.evaluation_metrics:
                    metrics = predictor.evaluation_metrics[city_name]
                    
                    # Create bar chart for actual vs predicted
                    fig = go.Figure()
                    
                    # Add actual values
                    fig.add_trace(go.Bar(
                        x=[f"Feature {i+1}" for i in range(len(metrics['actual']))],
                        y=metrics['actual'],
                        name='Actual',
                        marker_color='royalblue'
                    ))
                    
                    # Add predicted values
                    fig.add_trace(go.Bar(
                        x=[f"Feature {i+1}" for i in range(len(metrics['predicted']))],
                        y=metrics['predicted'],
                        name='Predicted',
                        marker_color='lightcoral'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f'{city_name} - Actual vs Predicted Values',
                        barmode='group',
                        xaxis_title='Features',
                        yaxis_title='Values',
                        legend_title='Type'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Model improvement recommendations
            st.write("### Model Improvement Recommendations")
            
            # Calculate average MAPE across all selected cities
            avg_mape = np.mean([predictor.evaluation_metrics[city]['MAPE'] for city in selected_cities if city in predictor.evaluation_metrics])
            
            # Display recommendations based on average MAPE
            if avg_mape < 10:
                st.success(f"Model is performing excellently with an average MAPE of {avg_mape:.2f}%. No immediate improvements needed.")
            elif avg_mape < 20:
                st.info(f"Model is performing well with an average MAPE of {avg_mape:.2f}%. Consider the following improvements:")
                st.markdown("""
                - Incorporate more historical data for better seasonal patterns
                - Add more weather-related features for correlation analysis
                """)
            else:
                st.warning(f"Model needs improvement with an average MAPE of {avg_mape:.2f}%. Consider the following:")
                st.markdown("""
                - Increase model complexity with additional LSTM layers
                - Incorporate more diverse features (traffic patterns, industrial activity)
                - Extend training time and adjust hyperparameters
                - Consider ensemble methods with multiple model types
                - Add feature engineering to capture non-linear relationships
                """)
        else:
            st.warning("Model evaluation metrics are not available. Please ensure the model has been trained.")
    
    
    with tab5:
        st.subheader("Air Quality Forecast & Recommendations")
        
        if selected_cities:
            # Get predictions for selected cities
            predictions = predictor.predict(days=prediction_days)
            
            # Create forecast visualizations
            for city_name in selected_cities:
                if city_name in predictions:
                    st.write(f"### {city_name} - Air Quality Forecast")
                    
                    # Get historical data for reference
                    hist_data = predictor.data_cache[city_name]
                    
                    # Create forecast dataframe
                    last_timestamp = pd.to_datetime(hist_data['timestamp'].iloc[-1])
                    forecast_dates = [last_timestamp + pd.Timedelta(days=i+1) for i in range(prediction_days)]
                    
                    # Extract AQI prediction (assuming index 6 is AQI in the prediction array)
                    aqi_predictions = [predictions[city_name][6] for _ in range(prediction_days)]
                    
                    # Create a dataframe for the forecast
                    forecast_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'Predicted AQI': aqi_predictions
                    })
                    
                    # Display the forecast
                    st.dataframe(forecast_df, use_container_width=True)
                    
                    # Calculate average predicted AQI
                    avg_pred_aqi = np.mean(aqi_predictions)
                    
                    # Provide health recommendations based on predicted AQI
                    st.write("#### Health Recommendations")
                    
                    if avg_pred_aqi <= 50:
                        st.success("Air quality is expected to be good. No special precautions needed.")
                    elif avg_pred_aqi <= 100:
                        st.info("Air quality is expected to be moderate. Sensitive individuals should consider limiting prolonged outdoor exertion.")
                    elif avg_pred_aqi <= 150:
                        st.warning("Air quality is expected to be unhealthy for sensitive groups. People with respiratory or heart disease, the elderly, and children should limit prolonged outdoor exertion.")
                    elif avg_pred_aqi <= 200:
                        st.error("Air quality is expected to be unhealthy. Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects.")
                    else:
                        st.error("Air quality is expected to be very unhealthy or hazardous. Everyone should avoid all outdoor exertion; sensitive groups should remain indoors and keep activity levels low.")
                    
                    # Provide environmental policy recommendations
                    st.write("#### Environmental Policy Recommendations")
                    
                    if avg_pred_aqi > 100:
                        st.markdown("""
                        - **Short-term measures**: Consider implementing odd-even vehicle restrictions
                        - **Public awareness**: Issue health advisories via SMS and local media
                        - **Industrial regulations**: Enforce stricter emission standards temporarily
                        - **Public transportation**: Increase frequency of buses and trains
                        """)
                    
                    if avg_pred_aqi > 150:
                        st.markdown("""
                        - **School closures**: Consider closing schools or moving classes online
                        - **Work from home**: Encourage businesses to allow remote work
                        - **Construction restrictions**: Temporarily halt major construction activities
                        - **Emergency response**: Activate air quality emergency response system
                        """)
        else:
            st.warning("Please select at least one city to view forecasts and recommendations.")
    
        # Footer with information about the system
        st.markdown("---")
        st.markdown("""
        ### About the Urban Air Quality Prediction System
        
        This advanced system uses a Graph Attention LSTM neural network to analyze and predict urban air quality metrics. 
        
        **Key features**:
        - Historical data analysis from multiple sources
        - Spatial-temporal modeling of pollutant dispersion
        - Integration of weather parameters for improved accuracy
        - Evaluation metrics to assess model performance
        - Actionable recommendations based on forecasts
        
        **Data sources**: Open-Meteo Air Quality API and Weather API
        
        **Citation**: If you use this system in your research, please cite: "Urban Air Quality Prediction System (2025)"
        """)

# Entry point for the Streamlit application
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"Application error: {e}", exc_info=True)