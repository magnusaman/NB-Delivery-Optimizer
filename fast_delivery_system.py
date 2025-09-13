#!/usr/bin/env python3
"""
Fast Real-World Delivery System - Like Blinkit/Uber Eats
Uses straight-line distances with realistic speed factors for fast computation
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
from typing import List, Dict, Tuple
import random
import time

class FastDeliverySystem:
    """Fast real-world delivery system with realistic routing"""
    
    def __init__(self):
        self.stores = []
        self.delivery_partners = []
        self.orders = []
        
    def load_stores(self, stores_file: str = "nb_verified_stores.gpkg"):
        """Load store locations"""
        print("Loading store locations...")
        stores_gdf = gpd.read_file(stores_file)
        
        for idx, store in stores_gdf.iterrows():
            self.stores.append({
                'id': f"STORE_{idx}",
                'name': store['name'],
                'city': store['city'],
                'address': store['address'],
                'location': store.geometry,
                'x': store.geometry.x,
                'y': store.geometry.y,
                'available': True
            })
        
        print(f"Loaded {len(self.stores)} stores")
    
    def create_delivery_partners(self, num_partners: int = 30):
        """Create delivery partners near stores"""
        print(f"Creating {num_partners} delivery partners...")
        
        # Get bounds from stores with buffer
        store_x = [store['x'] for store in self.stores]
        store_y = [store['y'] for store in self.stores]
        
        minx, maxx = min(store_x), max(store_x)
        miny, maxy = min(store_y), max(store_y)
        
        # Add buffer around stores (50km)
        buffer = 50000
        minx -= buffer
        maxx += buffer
        miny -= buffer
        maxy += buffer
        
        for i in range(num_partners):
            # Random location near stores
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            location = Point(x, y)
            
            self.delivery_partners.append({
                'id': f"PARTNER_{i+1}",
                'name': f"Delivery Partner {i+1}",
                'location': location,
                'x': x,
                'y': y,
                'available': True,
                'current_order': None,
                'rating': random.uniform(4.0, 5.0)
            })
        
        print(f"Created {len(self.delivery_partners)} delivery partners")
    
    def create_demo_orders(self, num_orders: int = 10):
        """Create realistic demo orders"""
        print(f"Creating {num_orders} demo orders...")
        
        # Get bounds from stores with smaller buffer for customers
        store_x = [store['x'] for store in self.stores]
        store_y = [store['y'] for store in self.stores]
        
        minx, maxx = min(store_x), max(store_x)
        miny, maxy = min(store_y), max(store_y)
        
        # Smaller buffer for customers (20km)
        buffer = 20000
        minx -= buffer
        maxx += buffer
        miny -= buffer
        maxy += buffer
        
        # Create realistic order scenarios
        order_scenarios = [
            {"value_range": (25, 50), "priority": "low", "count": 4},
            {"value_range": (50, 100), "priority": "medium", "count": 4},
            {"value_range": (100, 200), "priority": "high", "count": 2}
        ]
        
        order_id = 1
        for scenario in order_scenarios:
            for _ in range(scenario["count"]):
                if order_id > num_orders:
                    break
                    
                # Random customer location
                x = random.uniform(minx, maxx)
                y = random.uniform(miny, maxy)
                customer_location = Point(x, y)
                
                self.orders.append({
                    'id': f"ORDER_{order_id}",
                    'customer_name': f"Customer {order_id}",
                    'customer_location': customer_location,
                    'customer_x': x,
                    'customer_y': y,
                    'order_value': random.uniform(*scenario["value_range"]),
                    'priority': scenario["priority"],
                    'order_time': time.time(),
                    'status': 'pending',
                    'assigned_partner': None,
                    'estimated_delivery_time': None
                })
                order_id += 1
        
        print(f"Created {len(self.orders)} demo orders")
    
    def calculate_realistic_distance(self, start: Point, end: Point) -> float:
        """Calculate realistic road distance from straight-line distance"""
        straight_distance = start.distance(end)
        
        # Apply realistic road factor (roads are 1.3-1.5x longer than straight line)
        road_factor = random.uniform(1.3, 1.5)
        realistic_distance = straight_distance * road_factor
        
        return realistic_distance
    
    def find_nearest_store(self, customer_location: Point) -> Dict:
        """Find nearest store to customer"""
        min_distance = float('inf')
        nearest_store = None
        
        for store in self.stores:
            if store['available']:
                distance = self.calculate_realistic_distance(customer_location, store['location'])
                if distance < min_distance:
                    min_distance = distance
                    nearest_store = store
        
        return nearest_store
    
    def find_nearest_available_partner(self, store_location: Point) -> Dict:
        """Find nearest available delivery partner to store"""
        min_distance = float('inf')
        nearest_partner = None
        
        for partner in self.delivery_partners:
            if partner['available']:
                distance = self.calculate_realistic_distance(store_location, partner['location'])
                if distance < min_distance:
                    min_distance = distance
                    nearest_partner = partner
        
        return nearest_partner
    
    def process_order(self, order: Dict) -> Dict:
        """Process a single order: Customer → Store → Partner → Route"""
        print(f"\nProcessing {order['id']} ({order['priority']} priority, ${order['order_value']:.0f})...")
        
        # Step 1: Find nearest store
        nearest_store = self.find_nearest_store(order['customer_location'])
        print(f"  Nearest store: {nearest_store['name']} in {nearest_store['city']}")
        
        # Step 2: Find nearest available delivery partner
        nearest_partner = self.find_nearest_available_partner(nearest_store['location'])
        print(f"  Nearest partner: {nearest_partner['name']} (Rating: {nearest_partner['rating']:.1f})")
        
        # Step 3: Calculate realistic road distances
        partner_to_store = self.calculate_realistic_distance(
            nearest_partner['location'], nearest_store['location']
        )
        store_to_customer = self.calculate_realistic_distance(
            nearest_store['location'], order['customer_location']
        )
        total_distance = partner_to_store + store_to_customer
        
        # Step 4: Estimate delivery time (30 km/h average speed in city)
        total_distance_km = total_distance / 1000
        estimated_time = (total_distance_km / 30) * 60  # minutes
        
        # Step 5: Create route
        route = {
            'order_id': order['id'],
            'partner_id': nearest_partner['id'],
            'store_id': nearest_store['id'],
            'customer_location': order['customer_location'],
            'partner_to_store_distance': partner_to_store,
            'store_to_customer_distance': store_to_customer,
            'total_distance': total_distance,
            'estimated_delivery_time': estimated_time,
            'order_value': order['order_value'],
            'priority': order['priority'],
            'route_waypoints': [
                {'type': 'partner', 'location': nearest_partner['location']},
                {'type': 'store', 'location': nearest_store['location']},
                {'type': 'customer', 'location': order['customer_location']}
            ]
        }
        
        # Update order status
        order['assigned_partner'] = nearest_partner['id']
        order['estimated_delivery_time'] = estimated_time
        order['status'] = 'assigned'
        
        # Mark partner as busy
        nearest_partner['available'] = False
        nearest_partner['current_order'] = order['id']
        
        print(f"  Route: Partner → Store → Customer")
        print(f"  Partner to Store: {partner_to_store/1000:.1f} km")
        print(f"  Store to Customer: {store_to_customer/1000:.1f} km")
        print(f"  Total distance: {total_distance/1000:.1f} km")
        print(f"  Estimated time: {estimated_time:.1f} minutes")
        
        return route
    
    def process_all_orders(self) -> List[Dict]:
        """Process all orders and create routes"""
        print(f"\nProcessing {len(self.orders)} orders...")
        
        # Sort orders by priority (high first)
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        sorted_orders = sorted(self.orders, key=lambda x: priority_order[x['priority']], reverse=True)
        
        routes = []
        for order in sorted_orders:
            if order['status'] == 'pending':
                route = self.process_order(order)
                routes.append(route)
        
        return routes
    
    def export_results(self, routes: List[Dict], output_file: str = "fast_delivery_routes.gpkg"):
        """Export routes and data to GPKG file"""
        print(f"Exporting results to {output_file}...")
        
        try:
            # Create route polylines
            route_data = []
            for i, route in enumerate(routes):
                waypoints = route['route_waypoints']
                coords = [(wp['location'].x, wp['location'].y) for wp in waypoints]
                polyline = LineString(coords)
                
                route_data.append({
                    'route_id': f"ROUTE_{i+1}",
                    'order_id': route['order_id'],
                    'partner_id': route['partner_id'],
                    'store_id': route['store_id'],
                    'total_distance_km': route['total_distance'] / 1000,
                    'estimated_time_min': route['estimated_delivery_time'],
                    'order_value': route['order_value'],
                    'priority': route['priority'],
                    'geometry': polyline
                })
            
            # Create GeoDataFrames
            routes_gdf = gpd.GeoDataFrame(route_data, crs='EPSG:2953')
            
            # Export routes
            routes_gdf.to_file(output_file, driver='GPKG', layer='routes')
            print(f"  Exported {len(routes)} routes")
            
            # Export stores
            stores_data = []
            for store in self.stores:
                stores_data.append({
                    'store_id': store['id'],
                    'name': store['name'],
                    'city': store['city'],
                    'address': store['address'],
                    'geometry': store['location']
                })
            stores_gdf = gpd.GeoDataFrame(stores_data, crs='EPSG:2953')
            stores_gdf.to_file(output_file, driver='GPKG', layer='stores', mode='a')
            print(f"  Exported {len(self.stores)} stores")
            
            # Export delivery partners
            partners_data = []
            for partner in self.delivery_partners:
                partners_data.append({
                    'partner_id': partner['id'],
                    'name': partner['name'],
                    'rating': partner['rating'],
                    'available': partner['available'],
                    'geometry': partner['location']
                })
            partners_gdf = gpd.GeoDataFrame(partners_data, crs='EPSG:2953')
            partners_gdf.to_file(output_file, driver='GPKG', layer='partners', mode='a')
            print(f"  Exported {len(self.delivery_partners)} partners")
            
            # Export orders
            orders_data = []
            for order in self.orders:
                orders_data.append({
                    'order_id': order['id'],
                    'customer_name': order['customer_name'],
                    'order_value': order['order_value'],
                    'priority': order['priority'],
                    'status': order['status'],
                    'geometry': order['customer_location']
                })
            orders_gdf = gpd.GeoDataFrame(orders_data, crs='EPSG:2953')
            orders_gdf.to_file(output_file, driver='GPKG', layer='orders', mode='a')
            print(f"  Exported {len(self.orders)} orders")
            
            print(f"Successfully exported to {output_file}")
            return output_file
            
        except Exception as e:
            print(f"Export error: {e}")
            # Fallback: export routes only
            print("Trying fallback export...")
            
            route_data = []
            for i, route in enumerate(routes):
                waypoints = route['route_waypoints']
                coords = [(wp['location'].x, wp['location'].y) for wp in waypoints]
                polyline = LineString(coords)
                
                route_data.append({
                    'route_id': f"ROUTE_{i+1}",
                    'order_id': route['order_id'],
                    'partner_id': route['partner_id'],
                    'store_id': route['store_id'],
                    'total_distance_km': route['total_distance'] / 1000,
                    'estimated_time_min': route['estimated_delivery_time'],
                    'order_value': route['order_value'],
                    'priority': route['priority'],
                    'geometry': polyline
                })
            
            routes_gdf = gpd.GeoDataFrame(route_data, crs='EPSG:2953')
            routes_gdf.to_file("routes_only.gpkg", driver='GPKG')
            print(f"  Exported routes to routes_only.gpkg")
            
            return "routes_only.gpkg"
    
    def print_summary(self, routes: List[Dict]):
        """Print delivery system summary"""
        print(f"\n{'='*70}")
        print("FAST DELIVERY SYSTEM SUMMARY")
        print(f"{'='*70}")
        
        print(f"Stores: {len(self.stores)}")
        print(f"Delivery Partners: {len(self.delivery_partners)}")
        print(f"Orders: {len(self.orders)}")
        print(f"Routes Generated: {len(routes)}")
        
        total_distance = sum(route['total_distance'] for route in routes)
        total_time = sum(route['estimated_delivery_time'] for route in routes)
        total_value = sum(route['order_value'] for route in routes)
        
        print(f"\nTotal Distance: {total_distance/1000:.1f} km")
        print(f"Total Time: {total_time:.1f} minutes ({total_time/60:.1f} hours)")
        print(f"Total Order Value: ${total_value:.0f}")
        print(f"Average Distance per Route: {total_distance/len(routes)/1000:.1f} km")
        print(f"Average Time per Route: {total_time/len(routes):.1f} minutes")
        print(f"Average Order Value: ${total_value/len(routes):.0f}")
        
        # Priority breakdown
        priority_stats = {}
        for route in routes:
            priority = route['priority']
            if priority not in priority_stats:
                priority_stats[priority] = {'count': 0, 'distance': 0, 'time': 0}
            priority_stats[priority]['count'] += 1
            priority_stats[priority]['distance'] += route['total_distance']
            priority_stats[priority]['time'] += route['estimated_delivery_time']
        
        print(f"\nPriority Breakdown:")
        for priority, stats in priority_stats.items():
            print(f"  {priority.upper()}: {stats['count']} orders, "
                  f"{stats['distance']/1000:.1f} km, {stats['time']:.1f} min")
        
        print(f"\nSample Routes:")
        for i, route in enumerate(routes[:5]):
            print(f"  Route {i+1}: {route['partner_id']} → {route['store_id']} → Customer")
            print(f"    Distance: {route['total_distance']/1000:.1f} km, "
                  f"Time: {route['estimated_delivery_time']:.1f} min, "
                  f"Value: ${route['order_value']:.0f} ({route['priority']})")
        
        if len(routes) > 5:
            print(f"  ... and {len(routes) - 5} more routes")


def main():
    """Main function to run fast delivery system"""
    print("Fast Real-World Delivery System - Like Blinkit/Uber Eats")
    print("Customer Orders → Nearest Store → Nearest Partner → Route to Customer")
    print("Uses realistic road factors for fast computation")
    print("="*70)
    
    # Initialize system
    system = FastDeliverySystem()
    
    # Load data
    system.load_stores()
    system.create_delivery_partners(num_partners=30)
    system.create_demo_orders(num_orders=10)
    
    # Process orders
    routes = system.process_all_orders()
    
    # Export results
    output_file = system.export_results(routes)
    
    # Print summary
    system.print_summary(routes)
    
    print(f"\n✅ Fast delivery system complete!")
    print(f"✅ Results exported to: {output_file}")
    print(f"✅ Ready for QGIS visualization!")
    print(f"\nTo visualize in QGIS:")
    print(f"  1. Open QGIS")
    print(f"  2. Load {output_file}")
    print(f"  3. View layers: routes, stores, partners, orders")


if __name__ == "__main__":
    main()
