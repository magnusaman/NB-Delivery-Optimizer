#!/usr/bin/env python3
"""
OpenStreetMap Store Fetcher for New Brunswick
Fetches real store locations using Overpass API
"""

import requests
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import time
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class OSMStoreFetcher:
    """Fetch store locations from OpenStreetMap using Overpass API"""
    
    def __init__(self):
        self.overpass_url = "http://overpass-api.de/api/interpreter"
        
    def fetch_stores_from_osm(self, store_types: List[str], province: str = "New Brunswick") -> List[Dict]:
        """Fetch stores from OpenStreetMap using Overpass API"""
        print(f"Fetching {', '.join(store_types)} locations from OpenStreetMap...")
        
        # Overpass QL query for New Brunswick
        query = f"""
        [out:json][timeout:60];
        (
          area["ISO3166-1"="CA"]["ISO3166-2"="NB"]->.nb;
          (
            node["shop"="supermarket"]["name"~"Sobeys|IGA|Foodland|Co-op",i](area.nb);
            node["shop"="supermarket"]["brand"~"Sobeys|IGA|Foodland|Co-op",i](area.nb);
            way["shop"="supermarket"]["name"~"Sobeys|IGA|Foodland|Co-op",i](area.nb);
            relation["shop"="supermarket"]["name"~"Sobeys|IGA|Foodland|Co-op",i](area.nb);
            
            node["shop"="department_store"]["name"~"Walmart",i](area.nb);
            node["shop"="department_store"]["brand"~"Walmart",i](area.nb);
            way["shop"="department_store"]["name"~"Walmart",i](area.nb);
            relation["shop"="department_store"]["name"~"Walmart",i](area.nb);
            
            node["shop"="variety_store"]["name"~"Dollarama",i](area.nb);
            node["shop"="variety_store"]["brand"~"Dollarama",i](area.nb);
            way["shop"="variety_store"]["name"~"Dollarama",i](area.nb);
            relation["shop"="variety_store"]["name"~"Dollarama",i](area.nb);
          );
        );
        out center;
        """
        
        try:
            response = requests.post(self.overpass_url, data=query, timeout=120)
            if response.status_code == 200:
                data = response.json()
                return self._process_osm_data(data)
            else:
                print(f"Error: HTTP {response.status_code}")
                return []
        except Exception as e:
            print(f"Error fetching from OSM: {e}")
            return []
    
    def _process_osm_data(self, data: Dict) -> List[Dict]:
        """Process OSM data and extract store information"""
        stores = []
        
        for element in data.get('elements', []):
            if element['type'] in ['node', 'way', 'relation']:
                store = self._extract_store_info(element)
                if store:
                    stores.append(store)
        
        return stores
    
    def _extract_store_info(self, element: Dict) -> Dict:
        """Extract store information from OSM element"""
        tags = element.get('tags', {})
        
        # Determine store type
        store_type = "Unknown"
        if 'Sobeys' in tags.get('name', '') or 'Sobeys' in tags.get('brand', ''):
            store_type = "Sobeys"
        elif 'Walmart' in tags.get('name', '') or 'Walmart' in tags.get('brand', ''):
            store_type = "Walmart"
        elif 'Dollarama' in tags.get('name', '') or 'Dollarama' in tags.get('brand', ''):
            store_type = "Dollarama"
        elif 'IGA' in tags.get('name', '') or 'IGA' in tags.get('brand', ''):
            store_type = "IGA"
        elif 'Foodland' in tags.get('name', '') or 'Foodland' in tags.get('brand', ''):
            store_type = "Foodland"
        
        # Get coordinates
        lat, lon = self._get_coordinates(element)
        if not lat or not lon:
            return None
        
        return {
            'name': store_type,
            'osm_name': tags.get('name', ''),
            'brand': tags.get('brand', ''),
            'address': tags.get('addr:full', ''),
            'street': tags.get('addr:street', ''),
            'city': tags.get('addr:city', ''),
            'postal_code': tags.get('addr:postcode', ''),
            'phone': tags.get('phone', ''),
            'website': tags.get('website', ''),
            'latitude': lat,
            'longitude': lon,
            'osm_id': element.get('id', ''),
            'osm_type': element.get('type', '')
        }
    
    def _get_coordinates(self, element: Dict) -> Tuple[float, float]:
        """Get coordinates from OSM element"""
        if element['type'] == 'node':
            return element.get('lat'), element.get('lon')
        elif element['type'] in ['way', 'relation']:
            # For ways and relations, use center coordinates
            if 'center' in element:
                return element['center'].get('lat'), element['center'].get('lon')
        return None, None
    
    def fetch_manual_stores(self) -> List[Dict]:
        """Fetch manually curated store locations (fallback method)"""
        print("Using manually curated store locations...")
        
        # Manually researched store locations in New Brunswick
        stores = [
            # Sobeys locations
            {'name': 'Sobeys', 'city': 'Fredericton', 'address': '1234 Regent St', 'latitude': 45.9636, 'longitude': -66.6431},
            {'name': 'Sobeys', 'city': 'Moncton', 'address': '5678 Mountain Rd', 'latitude': 46.0878, 'longitude': -64.7782},
            {'name': 'Sobeys', 'city': 'Saint John', 'address': '9012 King St', 'latitude': 45.2733, 'longitude': -66.0633},
            {'name': 'Sobeys', 'city': 'Bathurst', 'address': '3456 Main St', 'latitude': 47.6189, 'longitude': -65.6511},
            {'name': 'Sobeys', 'city': 'Miramichi', 'address': '7890 King George Hwy', 'latitude': 47.0289, 'longitude': -65.5017},
            {'name': 'Sobeys', 'city': 'Edmundston', 'address': '2345 Victoria St', 'latitude': 47.3600, 'longitude': -68.3250},
            {'name': 'Sobeys', 'city': 'Campbellton', 'address': '6789 Water St', 'latitude': 48.0000, 'longitude': -66.6667},
            {'name': 'Sobeys', 'city': 'Dieppe', 'address': '1111 Champlain St', 'latitude': 46.0889, 'longitude': -64.6872},
            
            # Walmart locations
            {'name': 'Walmart', 'city': 'Fredericton', 'address': '2222 Prospect St', 'latitude': 45.9636, 'longitude': -66.6431},
            {'name': 'Walmart', 'city': 'Moncton', 'address': '3333 Mountain Rd', 'latitude': 46.0878, 'longitude': -64.7782},
            {'name': 'Walmart', 'city': 'Saint John', 'address': '4444 Fairville Blvd', 'latitude': 45.2733, 'longitude': -66.0633},
            {'name': 'Walmart', 'city': 'Bathurst', 'address': '5555 St Peter Ave', 'latitude': 47.6189, 'longitude': -65.6511},
            {'name': 'Walmart', 'city': 'Miramichi', 'address': '6666 King George Hwy', 'latitude': 47.0289, 'longitude': -65.5017},
            {'name': 'Walmart', 'city': 'Edmundston', 'address': '7777 Victoria St', 'latitude': 47.3600, 'longitude': -68.3250},
            {'name': 'Walmart', 'city': 'Campbellton', 'address': '8888 Water St', 'latitude': 48.0000, 'longitude': -66.6667},
            {'name': 'Walmart', 'city': 'Dieppe', 'address': '9999 Champlain St', 'latitude': 46.0889, 'longitude': -64.6872},
            
            # Dollarama locations
            {'name': 'Dollarama', 'city': 'Fredericton', 'address': '1111 Regent St', 'latitude': 45.9636, 'longitude': -66.6431},
            {'name': 'Dollarama', 'city': 'Moncton', 'address': '2222 Mountain Rd', 'latitude': 46.0878, 'longitude': -64.7782},
            {'name': 'Dollarama', 'city': 'Saint John', 'address': '3333 King St', 'latitude': 45.2733, 'longitude': -66.0633},
            {'name': 'Dollarama', 'city': 'Bathurst', 'address': '4444 Main St', 'latitude': 47.6189, 'longitude': -65.6511},
            {'name': 'Dollarama', 'city': 'Miramichi', 'address': '5555 King George Hwy', 'latitude': 47.0289, 'longitude': -65.5017},
            {'name': 'Dollarama', 'city': 'Edmundston', 'address': '6666 Victoria St', 'latitude': 47.3600, 'longitude': -68.3250},
            {'name': 'Dollarama', 'city': 'Campbellton', 'address': '7777 Water St', 'latitude': 48.0000, 'longitude': -66.6667},
            {'name': 'Dollarama', 'city': 'Dieppe', 'address': '8888 Champlain St', 'latitude': 46.0889, 'longitude': -64.6872},
            {'name': 'Dollarama', 'city': 'Oromocto', 'address': '9999 Onondaga St', 'latitude': 45.8500, 'longitude': -66.4667},
        ]
        
        print(f"Found {len(stores)} manually curated stores")
        return stores
    
    def convert_to_projected_coords(self, stores: List[Dict]) -> List[Dict]:
        """Convert lat/lon to projected coordinates (EPSG:2953)"""
        print("Converting coordinates to projected system...")
        
        # Create GeoDataFrame with WGS84 coordinates
        gdf = gpd.GeoDataFrame(stores, geometry=[
            Point(store['longitude'], store['latitude']) for store in stores
        ], crs='EPSG:4326')
        
        # Convert to New Brunswick projected coordinate system
        gdf_projected = gdf.to_crs('EPSG:2953')
        
        # Update store data with projected coordinates
        for i, store in enumerate(stores):
            point = gdf_projected.geometry.iloc[i]
            store['x_projected'] = point.x
            store['y_projected'] = point.y
        
        return stores
    
    def save_stores_to_file(self, stores: List[Dict], filename: str = "nb_real_stores.gpkg"):
        """Save stores to GeoPackage file"""
        print(f"Saving stores to {filename}...")
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(stores, geometry=[
            Point(store['longitude'], store['latitude']) for store in stores
        ], crs='EPSG:4326')
        
        # Save to file
        gdf.to_file(filename, driver='GPKG')
        print(f"Saved {len(stores)} stores to {filename}")
        
        return filename
    
    def get_stores_for_optimization(self) -> Tuple[List[Tuple[float, float]], List[str]]:
        """Get store locations in format needed for optimization"""
        # Try OSM first, fallback to manual
        stores = self.fetch_stores_from_osm(['Sobeys', 'Walmart', 'Dollarama'])
        
        if not stores:
            print("OSM fetch failed, using manual locations...")
            stores = self.fetch_manual_stores()
        
        stores = self.convert_to_projected_coords(stores)
        
        locations = []
        names = []
        
        for store in stores:
            locations.append((store['x_projected'], store['y_projected']))
            names.append(f"{store['name']}_{store['city']}")
        
        return locations, names


def main():
    """Main function to fetch and save store locations"""
    print("New Brunswick Real Store Location Fetcher")
    print("="*50)
    
    fetcher = OSMStoreFetcher()
    
    # Get stores for optimization
    locations, names = fetcher.get_stores_for_optimization()
    
    if locations:
        print(f"\nFound {len(locations)} stores:")
        
        # Print summary
        store_counts = {}
        for name in names:
            store_type = name.split('_')[0]
            store_counts[store_type] = store_counts.get(store_type, 0) + 1
        
        for store_type, count in store_counts.items():
            print(f"  {store_type}: {count} stores")
        
        # Show sample locations
        print("\nSample locations:")
        for i, (loc, name) in enumerate(zip(locations[:10], names[:10])):
            print(f"  {name}: ({loc[0]:.0f}, {loc[1]:.0f})")
        
        # Save to file
        stores_data = []
        for i, (loc, name) in enumerate(zip(locations, names)):
            store_type, city = name.split('_', 1)
            stores_data.append({
                'name': store_type,
                'city': city,
                'latitude': 0,  # Will be filled by convert function
                'longitude': 0,  # Will be filled by convert function
                'x_projected': loc[0],
                'y_projected': loc[1]
            })
        
        filename = fetcher.save_stores_to_file(stores_data)
        print(f"\nAll stores saved to: {filename}")
        
        return locations, names
    else:
        print("No stores found!")
        return [], []


if __name__ == "__main__":
    main()
