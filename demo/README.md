# Interactive VRP Demo

Professional interactive demo for showcasing the New Brunswick Delivery Route Optimization system.

## Features

- 🗺️ **Interactive Map**: Click anywhere in New Brunswick to place an order
- 🏪 **Real Stores**: Displays actual Walmart, Dollarama, and Sobeys locations
- 🚚 **Live Partners**: Shows 20 delivery partners distributed around stores
- 📍 **Smart Assignment**: Automatically assigns nearest store and partner
- 🛣️ **Real Routing**: Uses Google Distance Matrix API for actual driving distances and times
- 📊 **Live Metrics**: Shows distance, time, and route breakdown in real-time

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure `.env` file exists in parent directory with:
```
GOOGLE_MAPS_API_KEY=your_key_here
```

3. Run the demo:
```bash
python app.py
```

4. Open browser to: http://localhost:5001

## How to Use (For Presentation)

1. **Start the demo**: Open http://localhost:5001
2. **Explain the interface**: 
   - Map shows 50 stores (blue=Walmart, yellow=Dollarama, red=Sobeys)
   - Green diamonds are delivery partners with different vehicles
3. **Place an order**: Click anywhere on the map in New Brunswick
4. **Show the assignment**:
   - System finds nearest store (highlighted and bounces)
   - Assigns closest available partner (highlighted and bounces)
   - Draws route: Partner → Store → Customer
5. **Explain metrics**:
   - Real Google Maps distances (not straight-line)
   - Estimated delivery times based on traffic
   - Total route cost and time

## Architecture

- **Backend**: Flask (Python)
- **Frontend**: Vanilla JavaScript + Google Maps JavaScript API
- **Data Source**: Real store locations from Places API
- **Routing**: Google Distance Matrix API for real-time distances

## Presentation Tips

- Click in **Fredericton, Moncton, or Saint John** for best demo (lots of stores)
- Mention the **18.5% improvement** from GA optimization (from main project)
- Highlight **real API integration** (not simulated data)
- Show how system handles **dynamic order placement** in real-time

