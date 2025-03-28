
# AERONET: Air Quality Index Prediction System

## Project Overview
AERONET is an advanced Graph Neural Network (GNN) based air quality prediction system that leverages spatial-temporal machine learning techniques to forecast air quality indices across multiple urban environments in Southern India.

## Key Features
- 🌍 Multi-city Air Quality Prediction
- 🧠 Graph Neural Network with Attention Mechanism
- 📊 Comprehensive Pollutant Analysis
- 🌐 Geographic Proximity Modeling
- 🚀 High-Performance Prediction Framework

## Technical Architecture
- **Model**: Graph Attention LSTM Neural Network
- **Input**: Multi-pollutant Historical Data
- **Pollutants Monitored**: 
  - PM10
  - PM2.5
  - NO₂
  - SO₂
  - CO
  - O₃

## Cities Covered
1. Hyderabad
2. Bangalore
3. Adilabad
4. Jogulamba Gadwal
5. Warangal
6. Chennai

## Installation Dependencies
- Python 3.8+
- Libraries:
  - torch
  - networkx
  - pandas
  - numpy
  - scikit-learn
  - streamlit
  - plotly
  - requests

## Key Components
- `AirQualityPredictor`: Core prediction engine
- `GraphAttentionLSTM`: Custom neural network architecture
- Interactive Streamlit Dashboard
- Geographic Connectivity Modeling
- Comprehensive Data Caching Mechanism

## Unique Methodological Innovations
- Haversine Distance-based City Connectivity
- Dynamic Multi-Pollutant AQI Calculation
- Graph-based Spatial Correlation Modeling

## Data Sources
- Open-Meteo Air Quality API
- Fallback Synthetic Data Generation
- Local Caching Mechanism

## Visualization Features
- Interactive City Network Graph
- AQI Trend Analysis
- Detailed City-level Insights

## Usage
```bash
streamlit run aeronet.py
```

## Research Contributions
- Enhanced spatial-temporal air quality prediction
- Novel graph neural network approach
- Comprehensive multi-pollutant modeling

## License
MIT License

## Research Citation
Suggested Citation Format:
```
@software{AERONET2025,
  title={AERONET: Air Quality Index Prediction using Graph Neural Networks},
  author={[Maram Shiva Vaishnavi]},
  year={2025}
}
```

## Future Work
- Expand to more cities
- Integrate additional environmental sensors
- Develop real-time prediction API

## Acknowledgements
- Open-Meteo Air Quality API
- Computational Resources Support
```
