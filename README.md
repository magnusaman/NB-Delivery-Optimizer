# New Brunswick Delivery VRP Demo

A real-time delivery route optimization system for New Brunswick, featuring EV charging stations and intelligent partner assignment.

## Features

- 🗺️ **Interactive Map** - Google Maps integration with real-time data
- 🏪 **Store Discovery** - Walmart, Dollarama, and Sobeys locations
- 🚗 **EV Charging Stations** - 200+ charging stations across NB
- 👥 **Delivery Partners** - 72 partners with different vehicle types
- ⚡ **EV Routing** - Battery-aware routing for electric vehicles
- 🛣️ **Real Routes** - Google Directions API for accurate distances

## Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Maps**: Google Maps JavaScript API
- **APIs**: Google Maps Directions API, Distance Matrix API, Places API
- **Deployment**: Render

## Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd delivery-optimization
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r demo/requirements.txt
   ```

4. **Set environment variables**
   ```bash
   export GOOGLE_MAPS_API_KEY="your-api-key-here"
   ```

5. **Run the application**
   ```bash
   cd demo
   python app.py
   ```

6. **Open in browser**
   ```
   http://localhost:5001
   ```

## Deployment on Render

1. **Connect your GitHub repository to Render**
2. **Set environment variables**:
   - `GOOGLE_MAPS_API_KEY`: Your Google Maps API key
3. **Deploy automatically** - Render will detect the `render.yaml` configuration

## API Endpoints

- `GET /` - Main application interface
- `GET /api/data` - Returns stores, partners, and charging stations
- `POST /api/place-order` - Places an order and returns route optimization

## Data Sources

- **Stores**: Real store locations from Google Places API
- **Charging Stations**: EV charging stations from Google Places API  
- **Partners**: Generated delivery partners across NB cities
- **Routes**: Real-time routing via Google Directions API

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_MAPS_API_KEY` | Google Maps API key with Directions, Distance Matrix, and Places APIs enabled | Yes |

## License

MIT License