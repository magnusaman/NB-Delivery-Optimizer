#!/usr/bin/env python3
"""
Interactive VRP Demo - Flask Backend
Real-time order placement with nearest store + partner assignment
"""

from flask import Flask, render_template, request, jsonify
import json
import os
import sys
import math
import random
from typing import Tuple, List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
import googlemaps

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'demo-secret-key'

# Load Google Maps API key
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', '')

# Load stores data
def load_stores():
    stores = []
    for fname in ['stores_walmart.json', 'stores_dollarama.json', 'stores_sobeys.json']:
        path = os.path.join('..', 'nb_vrp_dataset', fname)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                items = json.load(f)
                for item in items:
                    loc = item.get('geometry', {}).get('location', {})
                    stores.append({
                        'lat': loc.get('lat'),
                        'lng': loc.get('lng'),
                        'name': item.get('name', 'Unknown'),
                        'chain': 'Walmart' if 'walmart' in fname else 'Dollarama' if 'dollarama' in fname else 'Sobeys'
                    })
    return stores

STORES = load_stores()

# Generate delivery partners around major cities
def generate_partners():
    partners = []
    random.seed(42)
    # Major NB cities with multiple partners each
    cities = [
        {'name': 'Fredericton', 'lat': 45.9636, 'lng': -66.6431, 'partners': 15},
        {'name': 'Moncton', 'lat': 46.0878, 'lng': -64.7782, 'partners': 15},
        {'name': 'Saint John', 'lat': 45.2733, 'lng': -66.0633, 'partners': 15},
        {'name': 'Bathurst', 'lat': 47.6189, 'lng': -65.6519, 'partners': 8},
        {'name': 'Miramichi', 'lat': 47.0280, 'lng': -65.5004, 'partners': 8},
        {'name': 'Edmundston', 'lat': 47.3734, 'lng': -68.3250, 'partners': 6},
        {'name': 'Campbellton', 'lat': 48.0055, 'lng': -66.6731, 'partners': 5},
    ]
    
    partner_id = 1
    for city in cities:
        for i in range(city['partners']):
            # Distribute partners within ~3km radius of city center
            partners.append({
                'id': f'P{partner_id}',
                'name': f'Partner {partner_id}',
                'city': city['name'],
                'lat': city['lat'] + random.uniform(-0.03, 0.03),
                'lng': city['lng'] + random.uniform(-0.03, 0.03),
                'status': 'available',
                'vehicle': random.choice(['EV', 'Van', 'Bike'])
            })
            partner_id += 1
    return partners

PARTNERS = generate_partners()

def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    lat1_r, lng1_r = math.radians(lat1), math.radians(lng1)
    lat2_r, lng2_r = math.radians(lat2), math.radians(lng2)
    dlat = lat2_r - lat1_r
    dlng = lng2_r - lng1_r
    a = math.sin(dlat/2)**2 + math.cos(lat1_r)*math.cos(lat2_r)*(math.sin(dlng/2)**2)
    return 2*R*math.asin(math.sqrt(a))


@app.route('/')
def index():
    return render_template('index.html', api_key=GOOGLE_MAPS_API_KEY)


@app.route('/api/data')
def get_data():
    """Return stores and partners"""
    return jsonify({
        'stores': STORES,  # Show all stores
        'partners': PARTNERS
    })


@app.route('/api/place-order', methods=['POST'])
def place_order():
    """Place order and assign nearest store + partner"""
    data = request.json
    lat = float(data['lat'])
    lng = float(data['lng'])
    
    # Find nearest store
    nearest_store = min(STORES, key=lambda s: haversine_km(lat, lng, s['lat'], s['lng']))
    store_distance = haversine_km(lat, lng, nearest_store['lat'], nearest_store['lng'])
    
    # Find nearest available partner to the store
    available_partners = [p for p in PARTNERS if p['status'] == 'available']
    if not available_partners:
        return jsonify({'error': 'No available partners'}), 400
    
    nearest_partner = min(available_partners, key=lambda p: haversine_km(
        nearest_store['lat'], nearest_store['lng'], p['lat'], p['lng']
    ))
    partner_to_store_distance = haversine_km(
        nearest_partner['lat'], nearest_partner['lng'],
        nearest_store['lat'], nearest_store['lng']
    )
    
    # Get real road distance and route from Google Directions API (not straight line)
    try:
        gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
        
        # Partner to store - use Directions API for actual road route
        directions1 = gmaps.directions(
            (nearest_partner['lat'], nearest_partner['lng']),
            (nearest_store['lat'], nearest_store['lng']),
            mode='driving',
            departure_time='now'
        )
        
        # Store to customer - use Directions API for actual road route
        directions2 = gmaps.directions(
            (nearest_store['lat'], nearest_store['lng']),
            (lat, lng),
            mode='driving',
            departure_time='now'
        )
        
        if directions1 and len(directions1) > 0:
            leg1 = directions1[0]['legs'][0]
            partner_to_store_real_km = leg1['distance']['value'] / 1000.0
            partner_to_store_time_min = leg1['duration']['value'] / 60.0
            route1_polyline = directions1[0]['overview_polyline']['points']
        else:
            partner_to_store_real_km = partner_to_store_distance * 1.3  # Road factor
            partner_to_store_time_min = partner_to_store_distance / 0.5
            route1_polyline = None
        
        if directions2 and len(directions2) > 0:
            leg2 = directions2[0]['legs'][0]
            store_to_customer_real_km = leg2['distance']['value'] / 1000.0
            store_to_customer_time_min = leg2['duration']['value'] / 60.0
            route2_polyline = directions2[0]['overview_polyline']['points']
        else:
            store_to_customer_real_km = store_distance * 1.3
            store_to_customer_time_min = store_distance / 0.5
            route2_polyline = None
    
    except Exception as e:
        print(f"API Error: {e}")
        partner_to_store_real_km = partner_to_store_distance * 1.3
        partner_to_store_time_min = partner_to_store_distance / 0.5
        store_to_customer_real_km = store_distance * 1.3
        store_to_customer_time_min = store_distance / 0.5
        route1_polyline = None
        route2_polyline = None
    
    total_distance = partner_to_store_real_km + store_to_customer_real_km
    total_time = partner_to_store_time_min + store_to_customer_time_min + 5  # +5 min pickup time
    
    return jsonify({
        'success': True,
        'order': {
            'customer_location': {'lat': lat, 'lng': lng}
        },
        'assigned_store': nearest_store,
        'assigned_partner': nearest_partner,
        'routes': {
            'partner_to_store_polyline': route1_polyline,
            'store_to_customer_polyline': route2_polyline
        },
        'metrics': {
            'partner_to_store_km': round(partner_to_store_real_km, 2),
            'partner_to_store_min': round(partner_to_store_time_min, 1),
            'store_to_customer_km': round(store_to_customer_real_km, 2),
            'store_to_customer_min': round(store_to_customer_time_min, 1),
            'total_distance_km': round(total_distance, 2),
            'estimated_time_min': round(total_time, 1)
        }
    })


if __name__ == '__main__':
    app.run(debug=True, port=5001)

