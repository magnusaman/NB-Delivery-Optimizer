#!/usr/bin/env python3
"""
Google Maps based VRP utilities for New Brunswick, Canada.

Provides:
- Store and charging station discovery via Places API
- Distance matrix retrieval via Distance Matrix API
- Route polylines via Directions API

Usage:
  1) pip install -r requirements.txt
  2) Create .env with GOOGLE_MAPS_API_KEY=...
  3) Run: python google_maps_vrp.py --bootstrap
"""

import os
import sys
import json
import time
import argparse
import math
from typing import List, Dict, Any, Tuple, Optional

from dotenv import load_dotenv
import googlemaps


NB_BOUNDS = {
    'northeast': {'lat': 48.074, 'lng': -63.687},
    'southwest': {'lat': 44.599, 'lng': -69.125},
}


def get_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_MAPS_API_KEY not set in environment or .env")
    return api_key


def create_gmaps_client() -> googlemaps.Client:
    return googlemaps.Client(key=get_api_key())


def _rectangular_bounds() -> str:
    ne = NB_BOUNDS['northeast']
    sw = NB_BOUNDS['southwest']
    # bounds string for Places API session bias in NE|SW format is not supported directly;
    # we will use textsearch with region bias via location/radius tiles.
    return json.dumps(NB_BOUNDS)


def tiles_for_nb(center_step_km: float = 50.0) -> List[Tuple[float, float]]:
    # Rough tiling over NB to paginate Places queries
    # Convert ~1 deg lat ~ 111km, lng scaled by cos(lat) ~ 0.7 at NB
    lat_deg_per_km = 1.0 / 111.0
    lng_deg_per_km = 1.0 / (111.0 * 0.7)
    lat_step = center_step_km * lat_deg_per_km
    lng_step = center_step_km * lng_deg_per_km
    lats = []
    lngs = []
    lat = NB_BOUNDS['southwest']['lat'] + lat_step
    while lat < NB_BOUNDS['northeast']['lat']:
        lats.append(lat)
        lat += lat_step
    lng = NB_BOUNDS['southwest']['lng'] + lng_step
    while lng < NB_BOUNDS['northeast']['lng']:
        lngs.append(lng)
        lng += lng_step
    centers = []
    for la in lats:
        for lo in lngs:
            centers.append((la, lo))
    return centers


def search_places_across_nb(
    gmaps_client: googlemaps.Client,
    keyword: Optional[str] = None,
    place_type: Optional[str] = None,
    radius_m: int = 30000
) -> List[Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for (lat, lng) in tiles_for_nb(60.0):
        kwargs: Dict[str, Any] = { 'location': (lat, lng), 'radius': radius_m }
        if keyword:
            kwargs['keyword'] = keyword
        if place_type:
            kwargs['type'] = place_type
        page = gmaps_client.places_nearby(**kwargs)
        for item in page.get('results', []):
            place_id = item.get('place_id')
            if place_id and place_id not in results:
                results[place_id] = item
        next_page_token = page.get('next_page_token')
        while next_page_token:
            time.sleep(2.0)
            page = gmaps_client.places_nearby(page_token=next_page_token)
            for item in page.get('results', []):
                place_id = item.get('place_id')
                if place_id and place_id not in results:
                    results[place_id] = item
            next_page_token = page.get('next_page_token')
    return list(results.values())


def fetch_nb_assets() -> Dict[str, List[Dict[str, Any]]]:
    gmaps_client = create_gmaps_client()
    walmart = search_places_across_nb(gmaps_client, keyword="Walmart", place_type="department_store")
    dollarama = search_places_across_nb(gmaps_client, keyword="Dollarama", place_type="store")
    sobeys = search_places_across_nb(gmaps_client, keyword="Sobeys", place_type="supermarket")
    # EV: prefer the official type and also merge results from a keyword search
    ev_type = search_places_across_nb(gmaps_client, keyword=None, place_type="electric_vehicle_charging_station")
    ev_kw = search_places_across_nb(gmaps_client, keyword="EV charging", place_type=None)
    # merge by place_id
    ev_map: Dict[str, Dict[str, Any]] = {}
    for item in ev_type + ev_kw:
        pid = item.get('place_id')
        if pid and pid not in ev_map:
            ev_map[pid] = item
    charging = list(ev_map.values())
    return {
        'walmart': walmart,
        'dollarama': dollarama,
        'sobeys': sobeys,
        'charging': charging,
    }


def extract_lat_lng(place: Dict[str, Any]) -> Tuple[float, float]:
    loc = place.get('geometry', {}).get('location', {})
    return float(loc.get('lat')), float(loc.get('lng'))


def build_distance_matrix(origins: List[Tuple[float, float]], destinations: List[Tuple[float, float]]) -> Dict[str, Any]:
    gmaps_client = create_gmaps_client()
    # Google supports up to 25x25 per call; chunk if larger
    def chunks(seq, size):
        for i in range(0, len(seq), size):
            yield seq[i:i+size]
    full_rows: List[List[Dict[str, Any]]] = []
    for o_chunk in chunks(origins, 25):
        row_block: List[List[Dict[str, Any]]] = []
        for d_chunk in chunks(destinations, 25):
            resp = gmaps_client.distance_matrix(o_chunk, d_chunk, mode="driving", departure_time="now")
            row_block.append(resp.get('rows', []))
            time.sleep(0.1)
        # Stitch row_block horizontally
        stitched_rows: List[List[Dict[str, Any]]] = []
        for row_idx in range(len(row_block[0])):
            combined: List[Dict[str, Any]] = []
            for block in row_block:
                combined.extend(block[row_idx].get('elements', []))
            stitched_rows.append(combined)
        full_rows.extend(stitched_rows)
    return {'rows': full_rows}


def fetch_route_polyline(origin: Tuple[float, float], destination: Tuple[float, float]) -> str:
    gmaps_client = create_gmaps_client()
    directions = gmaps_client.directions(origin, destination, mode="driving", departure_time="now")
    if not directions:
        return ""
    overview = directions[0].get('overview_polyline', {}).get('points', "")
    return overview


def save_json(path: str, data: Any) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def bootstrap_nb_assets(output_dir: str = "nb_vrp_dataset") -> None:
    os.makedirs(output_dir, exist_ok=True)
    assets = fetch_nb_assets()
    save_json(os.path.join(output_dir, 'stores_walmart.json'), assets['walmart'])
    save_json(os.path.join(output_dir, 'stores_dollarama.json'), assets['dollarama'])
    save_json(os.path.join(output_dir, 'stores_sobeys.json'), assets['sobeys'])
    save_json(os.path.join(output_dir, 'ev_charging.json'), assets['charging'])
    print(f"Saved Places results to {output_dir}")


# ===== Demo optimization utilities =====

def _demo_generate_drivers() -> List[Dict[str, Any]]:
    # Three drivers starting from major NB cities
    return [
        { 'id': 'D1', 'name': 'Driver 1', 'start': (45.9636, -66.6431), 'capacity': 8 },   # Fredericton
        { 'id': 'D2', 'name': 'Driver 2', 'start': (46.0878, -64.7782), 'capacity': 8 },   # Moncton
        { 'id': 'D3', 'name': 'Driver 3', 'start': (45.2733, -66.0633), 'capacity': 8 },   # Saint John
    ]


def _demo_generate_orders(n: int = 10) -> List[Dict[str, Any]]:
    # Ten simple orders near the three cities
    seeds = [
        (45.96, -66.65), (46.10, -64.78), (45.28, -66.06),
        (45.95, -66.65), (46.08, -64.80), (45.27, -66.07),
        (45.97, -66.64), (46.09, -64.76), (45.29, -66.05),
        (45.94, -66.63),
    ]
    orders: List[Dict[str, Any]] = []
    for i, (lat, lng) in enumerate(seeds[:n]):
        orders.append({ 'id': f'O{i+1}', 'location': (lat, lng), 'demand': 1, 'priority': 1 })
    return orders


def _load_any_store_coords(limit: int = 12) -> List[Tuple[float, float]]:
    paths = [
        os.path.join('nb_vrp_dataset', 'stores_walmart.json'),
        os.path.join('nb_vrp_dataset', 'stores_dollarama.json'),
        os.path.join('nb_vrp_dataset', 'stores_sobeys.json'),
    ]
    coords: List[Tuple[float, float]] = []
    for p in paths:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                items = json.load(f)
            coords.extend([extract_lat_lng(x) for x in items])
        if len(coords) >= limit:
            break
    return coords[:limit]


def _nearest_index(point: Tuple[float, float], candidates: List[Tuple[float, float]]) -> int:
    # Quick nearest by straight-line as a heuristic to pick store before DM call
    import math
    best_idx = 0
    best_d2 = float('inf')
    for i, (la, lo) in enumerate(candidates):
        d2 = (la - point[0])**2 + (lo - point[1])**2
        if d2 < best_d2:
            best_d2 = d2
            best_idx = i
    return best_idx


def run_demo_optimization() -> None:
    drivers = _demo_generate_drivers()
    orders = _demo_generate_orders(10)
    store_coords = _load_any_store_coords(15)
    if not store_coords:
        print('No store data found. Run --bootstrap first.')
        sys.exit(1)

    # Assign each order to nearest store (quick heuristic)
    order_to_store: Dict[str, Tuple[float, float]] = {}
    for o in orders:
        idx = _nearest_index(o['location'], store_coords)
        order_to_store[o['id']] = store_coords[idx]

    # Assign orders to drivers greedily by nearest start
    driver_loads: Dict[str, List[Dict[str, Any]]] = { d['id']: [] for d in drivers }
    for o in orders:
        nearest_driver = min(drivers, key=lambda d: (_nearest_index(d['start'], [o['location']])))
        if len(driver_loads[nearest_driver['id']]) < nearest_driver['capacity']:
            driver_loads[nearest_driver['id']].append(o)

    # Build simple sequences: driver -> store -> order per order
    routes: List[Dict[str, Any]] = []
    for d in drivers:
        for o in driver_loads[d['id']]:
            s_coord = order_to_store[o['id']]
            d_to_s = fetch_route_polyline(d['start'], s_coord)
            s_to_o = fetch_route_polyline(s_coord, o['location'])
            routes.append({
                'driver_id': d['id'],
                'order_id': o['id'],
                'driver_start': d['start'],
                'store': s_coord,
                'customer': o['location'],
                'polyline_driver_to_store': d_to_s,
                'polyline_store_to_customer': s_to_o,
            })

    # Export
    os.makedirs('outputs', exist_ok=True)
    save_json(os.path.join('outputs', 'assignments.json'), {
        'drivers': drivers,
        'orders': orders,
        'routes': routes,
    })
    print('Saved outputs/assignments.json')


def export_outputs_to_csv_geojson() -> None:
    # Requires outputs/assignments.json
    path = os.path.join('outputs', 'assignments.json')
    if not os.path.exists(path):
        print('No assignments found. Run --optimize first.')
        return
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    os.makedirs('outputs', exist_ok=True)

    # CSV export
    import csv
    csv_path = os.path.join('outputs', 'assignments.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(['driver_id','order_id','driver_lat','driver_lng','store_lat','store_lng','customer_lat','customer_lng'])
        for r in data.get('routes', []):
            dlat, dlng = r['driver_start']
            slat, slng = r['store']
            clat, clng = r['customer']
            writer.writerow([r['driver_id'], r['order_id'], dlat, dlng, slat, slng, clat, clng])

    # GeoJSON export: LineStrings for each leg
    gj = {
        'type': 'FeatureCollection',
        'features': []
    }
    for r in data.get('routes', []):
        dlat, dlng = r['driver_start']
        slat, slng = r['store']
        clat, clng = r['customer']
        gj['features'].append({
            'type': 'Feature',
            'properties': {'driver_id': r['driver_id'], 'order_id': r['order_id'], 'leg': 'driver_to_store'},
            'geometry': {'type': 'LineString', 'coordinates': [[dlng, dlat], [slng, slat]]}
        })
        gj['features'].append({
            'type': 'Feature',
            'properties': {'driver_id': r['driver_id'], 'order_id': r['order_id'], 'leg': 'store_to_customer'},
            'geometry': {'type': 'LineString', 'coordinates': [[slng, slat], [clng, clat]]}
        })
    with open(os.path.join('outputs', 'routes.geojson'), 'w', encoding='utf-8') as fg:
        json.dump(gj, fg)
    print('Saved outputs/assignments.csv and outputs/routes.geojson')


def render_google_map_html() -> None:
    # Build a self-contained HTML with markers and decoded polylines using Google Maps JS API
    path = os.path.join('outputs', 'assignments.json')
    if not os.path.exists(path):
        print('No assignments found. Run --optimize first.')
        return
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Gather markers: drivers, stores, customers
    drivers = data.get('drivers', [])
    orders = data.get('orders', [])
    routes = data.get('routes', [])

    # Create minimal HTML
    load_dotenv()
    api_key = os.getenv('GOOGLE_MAPS_API_KEY', '')
    if not api_key:
        print('Missing GOOGLE_MAPS_API_KEY')
        return
    html_path = os.path.join('outputs', 'map.html')
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>NB Demo Map</title>
  <style>
    html, body, #map {{ height: 100%; margin: 0; padding: 0; }}
    .legend {{ background: white; padding: 8px; margin: 8px; font: 12px Arial; }}
  </style>
</head>
<body>
  <div id=\"map\"></div>
  <div class=\"legend\">Drivers=blue, Stores=green, Customers=red</div>
  <script>
    const drivers = {json.dumps(drivers)};
    const routes = {json.dumps(routes)};
    function initMap() {{
      const center = {{ lat: 46.0, lng: -66.0 }};
      const map = new google.maps.Map(document.getElementById('map'), {{ zoom: 7, center }});
      // Mark drivers
      drivers.forEach(d => {{
        new google.maps.Marker({{ position: {{ lat: d.start[0], lng: d.start[1] }}, map, label: 'D', icon: 'http://maps.google.com/mapfiles/ms/icons/blue-dot.png' }});
      }});
      // Mark stores/customers and draw polylines
      routes.forEach(r => {{
        const store = {{ lat: r.store[0], lng: r.store[1] }};
        const cust = {{ lat: r.customer[0], lng: r.customer[1] }};
        new google.maps.Marker({{ position: store, map, label: 'S', icon: 'http://maps.google.com/mapfiles/ms/icons/green-dot.png' }});
        new google.maps.Marker({{ position: cust, map, label: 'C', icon: 'http://maps.google.com/mapfiles/ms/icons/red-dot.png' }});
        // Decode polylines
        if (r.polyline_driver_to_store) {{
          const path1 = google.maps.geometry.encoding.decodePath(r.polyline_driver_to_store);
          new google.maps.Polyline({{ map, path: path1, strokeColor: '#4285F4', strokeOpacity: 0.7, strokeWeight: 3 }});
        }}
        if (r.polyline_store_to_customer) {{
          const path2 = google.maps.geometry.encoding.decodePath(r.polyline_store_to_customer);
          new google.maps.Polyline({{ map, path: path2, strokeColor: '#34A853', strokeOpacity: 0.7, strokeWeight: 3 }});
        }}
      }});
    }}
  </script>
  <script src=\"https://maps.googleapis.com/maps/api/js?key={api_key}&libraries=geometry&callback=initMap\" async defer></script>
</body>
</html>
"""
    with open(html_path, 'w', encoding='utf-8') as fhtml:
        fhtml.write(html)
    print('Saved outputs/map.html')


def verify_stores_details() -> None:
    # Enrich store JSONs using Places Details to verify names/addresses/place_ids
    client = create_gmaps_client()
    targets = [
        ('stores_walmart.json', 'walmart'),
        ('stores_dollarama.json', 'dollarama'),
        ('stores_sobeys.json', 'sobeys'),
    ]
    out_dir = os.path.join('outputs', 'store_verification')
    os.makedirs(out_dir, exist_ok=True)
    for fname, label in targets:
        p = os.path.join('nb_vrp_dataset', fname)
        if not os.path.exists(p):
            print(f'Missing {p}, skip')
            continue
        with open(p, 'r', encoding='utf-8') as f:
            items = json.load(f)
        verified: List[Dict[str, Any]] = []
        for it in items[:100]:  # limit to 100 per chain for speed
            pid = it.get('place_id')
            if not pid:
                continue
            details = client.place(place_id=pid, fields=['place_id','name','formatted_address','geometry','rating','user_ratings_total','website'])
            verified.append(details.get('result', {}))
            time.sleep(0.05)
        save_json(os.path.join(out_dir, f'{label}_verified.json'), verified)
    print('Saved outputs/store_verification/*.json')


def render_all_stores_map() -> None:
    # Load all stores and render a single map with distinct marker colors
    paths = {
        'Walmart': os.path.join('nb_vrp_dataset', 'stores_walmart.json'),
        'Dollarama': os.path.join('nb_vrp_dataset', 'stores_dollarama.json'),
        'Sobeys': os.path.join('nb_vrp_dataset', 'stores_sobeys.json'),
    }
    stores: Dict[str, List[Tuple[float, float]]] = {}
    total = 0
    for label, p in paths.items():
        if not os.path.exists(p):
            print(f'Missing {p}. Run --bootstrap first.')
            return
        with open(p, 'r', encoding='utf-8') as f:
            items = json.load(f)
        coords = [extract_lat_lng(x) for x in items]
        stores[label] = coords
        total += len(coords)
    if total == 0:
        print('No store coordinates found.')
        return

    load_dotenv()
    api_key = os.getenv('GOOGLE_MAPS_API_KEY', '')
    if not api_key:
        print('Missing GOOGLE_MAPS_API_KEY')
        return

    os.makedirs('outputs', exist_ok=True)
    html_path = os.path.join('outputs', 'stores_map.html')

    # Prepare JS arrays
    js_data = {k: [{'lat': lat, 'lng': lng} for (lat, lng) in v] for k, v in stores.items()}

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>NB Stores Map</title>
  <style> html, body, #map {{ height: 100%; margin: 0; padding: 0; }} .legend {{ background: white; padding: 8px; margin: 8px; font: 12px Arial; }} </style>
</head>
<body>
  <div id=\"map\"></div>
  <div class=\"legend\">Walmart=blue, Dollarama=green, Sobeys=red</div>
  <script>
    const storeData = {json.dumps(js_data)};
    function initMap() {{
      const center = {{ lat: 46.2, lng: -66.0 }};
      const map = new google.maps.Map(document.getElementById('map'), {{ zoom: 7, center }});
      const icons = {{
        'Walmart': 'http://maps.google.com/mapfiles/ms/icons/blue-dot.png',
        'Dollarama': 'http://maps.google.com/mapfiles/ms/icons/green-dot.png',
        'Sobeys': 'http://maps.google.com/mapfiles/ms/icons/red-dot.png'
      }};
      for (const chain in storeData) {{
        const points = storeData[chain];
        points.forEach(pt => {{
          new google.maps.Marker({{ position: pt, map, icon: icons[chain], title: chain }});
        }});
      }}
    }}
  </script>
  <script src=\"https://maps.googleapis.com/maps/api/js?key={api_key}&callback=initMap\" async defer></script>
</body>
</html>
"""
    with open(html_path, 'w', encoding='utf-8') as fhtml:
        fhtml.write(html)
    print('Saved outputs/stores_map.html')


def render_representation_map() -> None:
    # Load stores and EV charging stations, render a combined map
    paths = {
        'Walmart': os.path.join('nb_vrp_dataset', 'stores_walmart.json'),
        'Dollarama': os.path.join('nb_vrp_dataset', 'stores_dollarama.json'),
        'Sobeys': os.path.join('nb_vrp_dataset', 'stores_sobeys.json'),
    }
    ev_path = os.path.join('nb_vrp_dataset', 'ev_charging.json')
    stores: Dict[str, List[Tuple[float, float]]] = {}
    for label, p in paths.items():
        if not os.path.exists(p):
            print(f'Missing {p}. Run --bootstrap first.')
            return
        with open(p, 'r', encoding='utf-8') as f:
            items = json.load(f)
        stores[label] = [extract_lat_lng(x) for x in items]
    if not os.path.exists(ev_path):
        print(f'Missing {ev_path}. Run --bootstrap first.')
        return
    with open(ev_path, 'r', encoding='utf-8') as f:
        ev_items = json.load(f)
    ev_coords = [extract_lat_lng(x) for x in ev_items]

    load_dotenv()
    api_key = os.getenv('GOOGLE_MAPS_API_KEY', '')
    if not api_key:
        print('Missing GOOGLE_MAPS_API_KEY')
        return

    os.makedirs('outputs', exist_ok=True)
    html_path = os.path.join('outputs', 'representation.html')

    js_stores = {k: [{'lat': lat, 'lng': lng} for (lat, lng) in v] for k, v in stores.items()}
    js_evs = [{'lat': lat, 'lng': lng} for (lat, lng) in ev_coords]

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>NB Representation Map</title>
  <style> html, body, #map {{ height: 100%; margin: 0; padding: 0; }} .legend {{ background: white; padding: 8px; margin: 8px; font: 12px Arial; }} </style>
</head>
<body>
  <div id=\"map\"></div>
  <div class=\"legend\">Walmart=blue, Dollarama=green, Sobeys=red, EV=purple</div>
  <script>
    const stores = {json.dumps(js_stores)};
    const evs = {json.dumps(js_evs)};
    function initMap() {{
      const center = {{ lat: 46.2, lng: -66.0 }};
      const map = new google.maps.Map(document.getElementById('map'), {{ zoom: 7, center }});
      const icons = {{
        'Walmart': 'http://maps.google.com/mapfiles/ms/icons/blue-dot.png',
        'Dollarama': 'http://maps.google.com/mapfiles/ms/icons/green-dot.png',
        'Sobeys': 'http://maps.google.com/mapfiles/ms/icons/red-dot.png',
        'EV': 'http://maps.google.com/mapfiles/ms/icons/purple-dot.png'
      }};
      for (const chain in stores) {{
        stores[chain].forEach(pt => {{
          new google.maps.Marker({{ position: pt, map, icon: icons[chain], title: chain }});
        }});
      }}
      evs.forEach(pt => {{
        new google.maps.Marker({{ position: pt, map, icon: icons['EV'], title: 'EV' }});
      }});
    }}
  </script>
  <script src=\"https://maps.googleapis.com/maps/api/js?key={api_key}&callback=initMap\" async defer></script>
</body>
</html>
"""
    with open(html_path, 'w', encoding='utf-8') as fhtml:
        fhtml.write(html)
    print('Saved outputs/representation.html')


# ===== Phase 1: Dataset generation and baselines =====

# Distance matrix cache system
DISTANCE_CACHE_FILE = os.path.join('outputs', 'distance_cache.json')

def _load_distance_cache() -> Dict[str, float]:
    if os.path.exists(DISTANCE_CACHE_FILE):
        with open(DISTANCE_CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def _save_distance_cache(cache: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(DISTANCE_CACHE_FILE), exist_ok=True)
    with open(DISTANCE_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)

def _cache_key(origin: Tuple[float, float], dest: Tuple[float, float]) -> str:
    return f"{origin[0]:.6f},{origin[1]:.6f}|{dest[0]:.6f},{dest[1]:.6f}"

def get_real_distance(origin: Tuple[float, float], dest: Tuple[float, float], cache: Dict[str, float]) -> float:
    """Get real driving distance from Google Distance Matrix API with caching"""
    key = _cache_key(origin, dest)
    if key in cache:
        return cache[key]
    
    # Fetch from Google Distance Matrix API
    try:
        gmaps = create_gmaps_client()
        result = gmaps.distance_matrix([origin], [dest], mode="driving", departure_time="now")
        if result and result.get('rows'):
            element = result['rows'][0]['elements'][0]
            if element['status'] == 'OK':
                distance_m = element['distance']['value']
                distance_km = distance_m / 1000.0
                cache[key] = distance_km
                return distance_km
    except Exception as e:
        print(f"Warning: API error for {key}, falling back to haversine: {e}")
    
    # Fallback to haversine
    dist = _haversine_km(origin, dest)
    cache[key] = dist
    return dist

def batch_fetch_distances(pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> Dict[str, float]:
    """Batch fetch distances from Google Distance Matrix API"""
    cache = _load_distance_cache()
    
    # Filter pairs not in cache
    to_fetch = [(o, d) for (o, d) in pairs if _cache_key(o, d) not in cache]
    
    if not to_fetch:
        print(f"All {len(pairs)} distances found in cache.")
        return cache
    
    print(f"Fetching {len(to_fetch)} new distances from Google Distance Matrix API...")
    
    # Fetch in chunks of 25x25 (API limit)
    gmaps = create_gmaps_client()
    for i in range(0, len(to_fetch), 25):
        chunk = to_fetch[i:i+25]
        origins = [o for (o, d) in chunk]
        dests = [d for (o, d) in chunk]
        
        try:
            result = gmaps.distance_matrix(origins, dests, mode="driving", departure_time="now")
            if result and result.get('rows'):
                for idx, row in enumerate(result['rows']):
                    for jdx, element in enumerate(row['elements']):
                        if element['status'] == 'OK':
                            origin = origins[idx]
                            dest = dests[jdx]
                            distance_km = element['distance']['value'] / 1000.0
                            cache[_cache_key(origin, dest)] = distance_km
            time.sleep(0.1)
        except Exception as e:
            print(f"Warning: API error in batch, falling back to haversine: {e}")
            for (o, d) in chunk:
                cache[_cache_key(o, d)] = _haversine_km(o, d)
    
    _save_distance_cache(cache)
    print(f"Distance cache updated. Total cached: {len(cache)} pairs.")
    return cache

def _load_all_store_coords() -> List[Tuple[float, float]]:
    coords: List[Tuple[float, float]] = []
    for fname in ['stores_walmart.json','stores_dollarama.json','stores_sobeys.json']:
        p = os.path.join('nb_vrp_dataset', fname)
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                items = json.load(f)
            coords.extend([extract_lat_lng(x) for x in items])
    return coords


def fetch_all_distances() -> None:
    """Smart sampling: fetch distances for nearest 5 stores per customer + top 50 store pairs"""
    dp = os.path.join('outputs', 'dataset', 'orders.json')
    if not os.path.exists(dp):
        print('Missing orders. Run --generate-demand first.')
        return
    with open(dp, 'r', encoding='utf-8') as f:
        orders = json.load(f)
    stores = _load_all_store_coords()
    if not stores:
        print('No stores found. Run --bootstrap first.')
        return
    
    print(f"Smart sampling: {len(orders)} orders, {len(stores)} stores")
    
    # Build list of needed pairs using smart sampling
    pairs = []
    
    # For each customer, only fetch distances to nearest 5 stores (by haversine)
    for o in orders:
        cust = (o['lat'], o['lng'])
        # Sort stores by haversine distance
        stores_with_dist = [(s, _haversine_km(cust, s)) for s in stores]
        stores_with_dist.sort(key=lambda x: x[1])
        # Take nearest 5
        for store, _ in stores_with_dist[:5]:
            pairs.append((store, cust))
            pairs.append((cust, store))  # bidirectional
    
    # Identify top 50 most-used stores (those assigned to most orders in greedy)
    store_usage = {}
    for o in orders:
        cust = (o['lat'], o['lng'])
        nearest = min(stores, key=lambda s: _haversine_km(cust, s))
        store_usage[nearest] = store_usage.get(nearest, 0) + 1
    
    top_stores = sorted(store_usage.keys(), key=lambda s: store_usage[s], reverse=True)[:50]
    
    # Store-to-store for top 50
    for i, s1 in enumerate(top_stores):
        for s2 in top_stores[i+1:]:
            pairs.append((s1, s2))
            pairs.append((s2, s1))
    
    # Deduplicate
    pairs = list(set(pairs))
    
    print(f"Smart sampling: {len(pairs)} distance pairs (nearest 5 stores per customer + top 50 store pairs)")
    batch_fetch_distances(pairs)
    print("Distance fetching complete.")

def generate_synthetic_demand(num_per_day: int = 150) -> None:
    import random
    random.seed(42)
    stores = _load_all_store_coords()
    if not stores:
        print('No stores found. Run --bootstrap first.')
        return
    os.makedirs(os.path.join('outputs', 'dataset'), exist_ok=True)
    orders: List[Dict[str, Any]] = []
    for i in range(num_per_day):
        # pick a random store as center and jitter ~2-5km
        s_lat, s_lng = random.choice(stores)
        d_km = random.uniform(0.5, 5.0)
        bearing = random.uniform(0, 360)
        # approx conversion
        dlat = (d_km / 111.0) * math.cos(math.radians(bearing))
        dlng = (d_km / (111.0 * 0.7)) * math.sin(math.radians(bearing))
        lat = s_lat + dlat
        lng = s_lng + dlng
        # time window: 30-120 minutes from start time; priority mix
        priority = 1 if random.random() < 0.3 else 0
        window_min = random.choice([30, 60, 90, 120])
        size = random.randint(1, 3)
        orders.append({
            'id': f'ORD{i+1}',
            'lat': round(lat, 6),
            'lng': round(lng, 6),
            'size': size,
            'priority': priority,
            'time_window_min': window_min,
        })
    with open(os.path.join('outputs', 'dataset', 'orders.json'), 'w', encoding='utf-8') as f:
        json.dump(orders, f, indent=2)
    print('Saved outputs/dataset/orders.json')


def _haversine_km(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    import math
    R = 6371.0
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    x = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*(math.sin(dlon/2)**2)
    return 2*R*math.asin(math.sqrt(x))


def solve_greedy_baseline(use_real_distances: bool = False) -> None:
    # Load orders
    p = os.path.join('outputs', 'dataset', 'orders.json')
    if not os.path.exists(p):
        print('Missing orders. Run --generate-demand first.')
        return
    with open(p, 'r', encoding='utf-8') as f:
        orders = json.load(f)
    stores = _load_all_store_coords()
    if not stores:
        print('No stores found. Run --bootstrap first.')
        return
    
    distance_cache = _load_distance_cache() if use_real_distances else {}
    
    def dist_func(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        if use_real_distances and distance_cache:
            key = _cache_key(a, b)
            if key in distance_cache:
                return distance_cache[key]
        return _haversine_km(a, b)
    
    # Assign each order to nearest store
    assignments: List[Dict[str, Any]] = []
    for o in orders:
        cust = (o['lat'], o['lng'])
        nearest = min(stores, key=lambda s: dist_func(cust, s))
        assignments.append({ 'order_id': o['id'], 'store_lat': nearest[0], 'store_lng': nearest[1], 'customer_lat': cust[0], 'customer_lng': cust[1] })
    
    suffix = '_real' if use_real_distances else ''
    outdir = os.path.join('outputs', 'baselines', f'greedy{suffix}')
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'assignments.json'), 'w', encoding='utf-8') as f:
        json.dump(assignments, f, indent=2)
    print(f'Saved {outdir}/assignments.json (real distances: {use_real_distances})')


def solve_ga_baseline(use_real_distances: bool = False) -> None:
    import random
    import copy
    random.seed(42)
    
    # Load dataset
    dp = os.path.join('outputs', 'dataset', 'orders.json')
    if not os.path.exists(dp):
        print('Missing orders. Run --generate-demand first.')
        return
    with open(dp, 'r', encoding='utf-8') as f:
        orders = json.load(f)
    stores = _load_all_store_coords()
    if not stores:
        print('No stores found. Run --bootstrap first.')
        return
    
    distance_cache = _load_distance_cache() if use_real_distances else {}
    
    def dist_func(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        if use_real_distances and distance_cache:
            key = _cache_key(a, b)
            if key in distance_cache:
                return distance_cache[key]
        return _haversine_km(a, b)
    
    # Cluster orders by nearest store (greedy assignment)
    clusters: Dict[int, List[int]] = {}
    order_list = []
    for i, o in enumerate(orders):
        cust = (o['lat'], o['lng'])
        nearest_idx = min(range(len(stores)), key=lambda si: dist_func(cust, stores[si]))
        if nearest_idx not in clusters:
            clusters[nearest_idx] = []
        clusters[nearest_idx].append(i)
        order_list.append((i, o['lat'], o['lng']))
    
    # GA parameters
    population_size = 50
    generations = 100
    mutation_rate = 0.15
    elite_size = 5
    
    # Chromosome: list of order indices; decode into routes per cluster
    # Fitness: total distance
    def fitness(chromosome: List[int]) -> float:
        total = 0.0
        for store_idx, order_indices in clusters.items():
            # extract order sequence for this cluster
            seq = [oi for oi in chromosome if oi in order_indices]
            if not seq:
                continue
            store = stores[store_idx]
            # tour: store -> orders in seq -> back to store
            prev = store
            for oi in seq:
                o = orders[oi]
                curr = (o['lat'], o['lng'])
                total += dist_func(prev, curr)
                prev = curr
            total += dist_func(prev, store)
        return total
    
    # Initialize population
    pop = []
    base_order = list(range(len(orders)))
    for _ in range(population_size):
        c = base_order[:]
        random.shuffle(c)
        pop.append(c)
    
    # PMX crossover
    def pmx_crossover(p1: List[int], p2: List[int]) -> List[int]:
        n = len(p1)
        if n < 2:
            return p1[:]
        cx1, cx2 = sorted(random.sample(range(n), 2))
        child = [-1]*n
        child[cx1:cx2] = p1[cx1:cx2]
        mapping = {p1[i]: p2[i] for i in range(cx1, cx2)}
        for i in range(n):
            if child[i] == -1:
                val = p2[i]
                while val in mapping:
                    val = mapping[val]
                child[i] = val
        return child
    
    # Swap mutation
    def swap_mutation(c: List[int]) -> List[int]:
        c = c[:]
        if len(c) < 2:
            return c
        i, j = random.sample(range(len(c)), 2)
        c[i], c[j] = c[j], c[i]
        return c
    
    # Evolution
    best_solution = None
    best_fitness = float('inf')
    
    for gen in range(generations):
        # Evaluate
        fits = [(fitness(c), c) for c in pop]
        fits.sort()
        if fits[0][0] < best_fitness:
            best_fitness = fits[0][0]
            best_solution = fits[0][1]
        
        # Selection: elitism + tournament
        new_pop = [c for (f, c) in fits[:elite_size]]
        while len(new_pop) < population_size:
            # Tournament
            t1, t2 = random.sample(fits, 2)
            p1 = t1[1] if t1[0] < t2[0] else t2[1]
            t3, t4 = random.sample(fits, 2)
            p2 = t3[1] if t3[0] < t4[0] else t4[1]
            child = pmx_crossover(p1, p2)
            if random.random() < mutation_rate:
                child = swap_mutation(child)
            new_pop.append(child)
        pop = new_pop
        if gen % 20 == 0:
            print(f'Gen {gen}: best fitness = {best_fitness:.2f} km')
    
    # Decode best solution into assignments
    assignments = []
    for store_idx, order_indices in clusters.items():
        seq = [oi for oi in best_solution if oi in order_indices]
        store = stores[store_idx]
        for oi in seq:
            o = orders[oi]
            assignments.append({
                'order_id': o['id'],
                'store_lat': store[0],
                'store_lng': store[1],
                'customer_lat': o['lat'],
                'customer_lng': o['lng']
            })
    
    # Export
    suffix = '_real' if use_real_distances else ''
    outdir = os.path.join('outputs', 'baselines', f'ga{suffix}')
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'assignments.json'), 'w', encoding='utf-8') as f:
        json.dump(assignments, f, indent=2)
    with open(os.path.join(outdir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump({'num_orders': len(assignments), 'best_fitness_km': round(best_fitness, 3), 'real_distances': use_real_distances}, f, indent=2)
    print(f'Saved GA baseline: best fitness = {best_fitness:.2f} km (real distances: {use_real_distances})')


def evaluate_and_export() -> None:
    import csv
    # Load dataset and greedy assignments
    dp = os.path.join('outputs', 'dataset', 'orders.json')
    gp = os.path.join('outputs', 'baselines', 'greedy', 'assignments.json')
    if not (os.path.exists(dp) and os.path.exists(gp)):
        print('Missing inputs. Run --generate-demand and --solve-greedy first.')
        return
    with open(dp, 'r', encoding='utf-8') as f:
        orders = {o['id']: o for o in json.load(f)}
    with open(gp, 'r', encoding='utf-8') as f:
        assigns = json.load(f)
    
    # Cluster assignments by store to compute full tours (store -> customer -> back to store)
    store_clusters: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}
    for a in assigns:
        store_key = (a['store_lat'], a['store_lng'])
        if store_key not in store_clusters:
            store_clusters[store_key] = []
        store_clusters[store_key].append(a)
    
    # Metrics with return trips
    total_dist = 0.0
    rows = []
    for store_coord, cluster in store_clusters.items():
        # Each customer: store -> customer -> back to store
        for a in cluster:
            store_to_cust = _haversine_km(store_coord, (a['customer_lat'], a['customer_lng']))
            cust_to_store = store_to_cust  # symmetric
            trip_dist = store_to_cust + cust_to_store
            total_dist += trip_dist
            rows.append([a['order_id'], a['store_lat'], a['store_lng'], a['customer_lat'], a['customer_lng'], round(store_to_cust, 3), round(trip_dist, 3)])
    avg_dist = (total_dist / max(1, len(assigns)))

    outdir = os.path.join('outputs', 'baselines', 'greedy')
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'assignments.csv'), 'w', newline='', encoding='utf-8') as fcsv:
        w = csv.writer(fcsv)
        w.writerow(['order_id','store_lat','store_lng','customer_lat','customer_lng','one_way_km','round_trip_km'])
        w.writerows(rows)
    with open(os.path.join(outdir, 'metrics.json'), 'w', encoding='utf-8') as fm:
        json.dump({'num_orders': len(assigns), 'total_distance_km': round(total_dist, 3), 'avg_distance_km': round(avg_dist, 3), 'note': 'includes round trips'}, fm, indent=2)
    print('Saved greedy baseline metrics and CSV under outputs/baselines/greedy/ (with round trips)')


def render_eval_map() -> None:
    # Visualize orders and their assigned stores as straight lines (evaluation view)
    dp = os.path.join('outputs', 'dataset', 'orders.json')
    gp = os.path.join('outputs', 'baselines', 'greedy', 'assignments.json')
    if not (os.path.exists(dp) and os.path.exists(gp)):
        print('Missing inputs. Run --generate-demand and --solve-greedy first.')
        return
    with open(dp, 'r', encoding='utf-8') as f:
        orders = json.load(f)
    with open(gp, 'r', encoding='utf-8') as f:
        assigns = json.load(f)

    load_dotenv()
    api_key = os.getenv('GOOGLE_MAPS_API_KEY', '')
    if not api_key:
        print('Missing GOOGLE_MAPS_API_KEY')
        return

    os.makedirs('outputs', exist_ok=True)
    html_path = os.path.join('outputs', 'evaluation_map.html')

    js_orders = [{'lat': o['lat'], 'lng': o['lng'], 'id': o['id']} for o in orders]
    js_lines = [{'s': {'lat': a['store_lat'], 'lng': a['store_lng']}, 'c': {'lat': a['customer_lat'], 'lng': a['customer_lng']}} for a in assigns]

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Evaluation Map</title>
  <style> html, body, #map {{ height: 100%; margin: 0; padding: 0; }} .legend {{ background: white; padding: 8px; margin: 8px; font: 12px Arial; }} </style>
</head>
<body>
  <div id=\"map\"></div>
  <div class=\"legend\">Orders=red dots, Stores=blue dots, Lines=assignments</div>
  <script>
    const orders = {json.dumps(js_orders)};
    const lines = {json.dumps(js_lines)};
    function initMap() {{
      const center = {{ lat: 46.2, lng: -66.0 }};
      const map = new google.maps.Map(document.getElementById('map'), {{ zoom: 7, center }});
      orders.forEach(o => {{
        new google.maps.Marker({{ position: {{ lat: o.lat, lng: o.lng }}, map, icon: 'http://maps.google.com/mapfiles/ms/icons/red-dot.png', title: o.id }});
      }});
      lines.forEach(l => {{
        new google.maps.Polyline({{ map, path: [l.s, l.c], strokeColor: '#999', strokeOpacity: 0.7, strokeWeight: 2 }});
        new google.maps.Marker({{ position: l.s, map, icon: 'http://maps.google.com/mapfiles/ms/icons/blue-dot.png' }});
      }});
    }}
  </script>
  <script src=\"https://maps.googleapis.com/maps/api/js?key={api_key}&callback=initMap\" async defer></script>
  </body>
  </html>
"""
    with open(html_path, 'w', encoding='utf-8') as fhtml:
        fhtml.write(html)
    print('Saved outputs/evaluation_map.html')


def main() -> None:
    parser = argparse.ArgumentParser(description="Google Maps VRP helpers for New Brunswick")
    parser.add_argument('--bootstrap', action='store_true', help='Fetch NB stores and EV stations into nb_vrp_dataset/')
    parser.add_argument('--distance-matrix', action='store_true', help='Build a demo distance matrix over fetched stores')
    parser.add_argument('--optimize', action='store_true', help='Run demo optimization with generated drivers and orders')
    parser.add_argument('--export', action='store_true', help='Export assignments to CSV and GeoJSON')
    parser.add_argument('--render-map', action='store_true', help='Render Google Maps HTML with markers and routes')
    parser.add_argument('--render-stores-map', action='store_true', help='Render Google Maps HTML with all store markers')
    parser.add_argument('--verify-stores', action='store_true', help='Verify store details via Places Details API')
    parser.add_argument('--render-representation', action='store_true', help='Render combined stores + EV map to representation.html')
    # Phase 1 research dataset + baselines
    parser.add_argument('--generate-demand', action='store_true', help='Generate synthetic demand near stores')
    parser.add_argument('--fetch-distances', action='store_true', help='Pre-fetch real driving distances from Google Distance Matrix API')
    parser.add_argument('--solve-greedy', action='store_true', help='Greedy baseline routing per store (nearest neighbor)')
    parser.add_argument('--solve-ga', action='store_true', help='Simple GA baseline routing per store')
    parser.add_argument('--use-real-distances', action='store_true', help='Use cached real distances instead of haversine')
    parser.add_argument('--evaluate', action='store_true', help='Compute metrics and export CSVs for greedy baseline')
    parser.add_argument('--render-eval-map', action='store_true', help='Render map of orders and greedy assignments')
    args = parser.parse_args()

    if args.bootstrap:
        bootstrap_nb_assets()
        return

    if args.distance_matrix:
        # Demo: use a few Walmart stores if present
        src_path = os.path.join('nb_vrp_dataset', 'stores_walmart.json')
        if not os.path.exists(src_path):
            print("Run with --bootstrap first to fetch stores.")
            sys.exit(1)
        with open(src_path, 'r', encoding='utf-8') as f:
            stores = json.load(f)
        coords = [extract_lat_lng(p) for p in stores[:8]]
        if len(coords) < 2:
            print("Not enough stores fetched.")
            sys.exit(1)
        dm = build_distance_matrix(coords, coords)
        save_json('distance_matrix_demo.json', dm)
        print("Saved distance_matrix_demo.json")

    if args.optimize:
        run_demo_optimization()
        
    if args.export:
        export_outputs_to_csv_geojson()

    if args.render_map:
        render_google_map_html()

    if args.render_stores_map:
        render_all_stores_map()

    if args.verify_stores:
        verify_stores_details()

    if args.render_representation:
        render_representation_map()

    if args.generate_demand:
        generate_synthetic_demand()

    if args.fetch_distances:
        fetch_all_distances()

    if args.solve_greedy:
        solve_greedy_baseline(use_real_distances=args.use_real_distances)

    if args.solve_ga:
        solve_ga_baseline(use_real_distances=args.use_real_distances)

    if args.evaluate:
        evaluate_and_export()

    if args.render_eval_map:
        render_eval_map()


if __name__ == '__main__':
    main()


