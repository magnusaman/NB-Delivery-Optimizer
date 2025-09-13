#!/usr/bin/env python3
"""
Final New Brunswick Delivery System
- Multiple delivery partners per store (2-3 within 1km)
- Order → Closest Store → Closest Partner → Customer
- Comprehensive map visualization with all elements
"""

import math
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString, MultiLineString
import fiona
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
from copy import deepcopy
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

warnings.filterwarnings('ignore')

@dataclass
class DeliveryAssignment:
    """Represents a delivery assignment"""
    order_id: str
    store_id: str
    partner_id: str
    route: List[Tuple[float, float]]
    total_time: float
    total_distance: float

class FinalDeliverySystem:
    """Final delivery system with multiple partners per store and comprehensive visualization"""
    
    def __init__(self, roads_file: str = "roads_clean.gpkg"):
        self.roads_file = roads_file
        self.G = nx.Graph()
        self.roads_gdf = None
        self.stores = []
        self.partners = []
        self.orders = []
        self.assignments = []
        
        # Load road network
        self.load_road_network()
        
    def load_road_network(self):
        """Load and process the road network from GeoPackage"""
        print("🛣️  Loading road network...")
        
        try:
            self.roads_gdf = gpd.read_file(self.roads_file)
            print(f"   ✅ Loaded {len(self.roads_gdf)} road segments")
        except Exception as e:
            print(f"   ⚠️  Could not load roads: {e}")
            self.roads_gdf = None
    
    def add_stores(self, store_locations: List[Tuple[float, float]], store_names: List[str] = None):
        """Add store locations"""
        if store_names is None:
            store_names = [f"Store_{i+1}" for i in range(len(store_locations))]
        
        self.stores = []
        for i, (x, y) in enumerate(store_locations):
            store = {
                'id': store_names[i],
                'location': (x, y),
                'name': store_names[i]
            }
            self.stores.append(store)
        
        print(f"🏪 Added {len(self.stores)} stores")
    
    def add_delivery_partners(self, partners_per_store: int = 3):
        """Add multiple delivery partners per store (2-3 within 1km)"""
        print(f"🚚 Adding {partners_per_store} delivery partners per store...")
        
        self.partners = []
        partner_id = 1
        
        for store in self.stores:
            store_coords = store['location']
            
            # Generate partners around each store (within 1km)
            for i in range(partners_per_store):
                # Generate random position within 1km of store
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(200, 1000)  # 200m to 1km from store
                
                # Calculate partner position
                partner_x = store_coords[0] + distance * math.cos(angle)
                partner_y = store_coords[1] + distance * math.sin(angle)
                
                partner = {
                    'id': f"Partner_{partner_id}",
                    'store_id': store['id'],
                    'location': (partner_x, partner_y),
                    'distance_from_store': distance,
                    'vehicle_type': random.choice(['bike', 'scooter', 'car']),
                    'capacity': random.choice([5, 8, 10]),
                    'status': 'available'
                }
                self.partners.append(partner)
                partner_id += 1
        
        print(f"   ✅ Generated {len(self.partners)} delivery partners")
        for store in self.stores:
            store_partners = [p for p in self.partners if p['store_id'] == store['id']]
            print(f"   📍 {store['id']}: {len(store_partners)} partners")
    
    def generate_orders(self, num_orders: int = 5):
        """Generate sample orders"""
        print(f"📦 Generating {num_orders} orders...")
        
        # Generate random order locations
        all_coords = []
        for store in self.stores:
            all_coords.append(store['location'])
        for partner in self.partners:
            all_coords.append(partner['location'])
        
        if not all_coords:
            print("   ⚠️  No locations available for order generation")
            return
        
        # Find bounding box
        min_x = min(coord[0] for coord in all_coords)
        max_x = max(coord[0] for coord in all_coords)
        min_y = min(coord[1] for coord in all_coords)
        max_y = max(coord[1] for coord in all_coords)
        
        self.orders = []
        for i in range(num_orders):
            # Generate random order location within bounding box
            order_x = random.uniform(min_x - 2000, max_x + 2000)
            order_y = random.uniform(min_y - 2000, max_y + 2000)
            
            order = {
                'id': f"Order_{i+1}",
                'location': (order_x, order_y),
                'demand': random.randint(1, 3),
                'priority': random.choice(['normal', 'urgent']),
                'created_at': datetime.now()
            }
            self.orders.append(order)
        
        print(f"   ✅ Generated {len(self.orders)} orders")
    
    def find_closest_store(self, order_location: Tuple[float, float]) -> str:
        """Find closest store to order location"""
        min_distance = float('inf')
        closest_store_id = None
        
        for store in self.stores:
            distance = math.hypot(
                order_location[0] - store['location'][0],
                order_location[1] - store['location'][1]
            )
            if distance < min_distance:
                min_distance = distance
                closest_store_id = store['id']
        
        return closest_store_id
    
    def find_closest_partner(self, store_id: str, order_location: Tuple[float, float]) -> str:
        """Find closest available partner to the store for the order"""
        store_partners = [p for p in self.partners if p['store_id'] == store_id and p['status'] == 'available']
        
        if not store_partners:
            return None
        
        min_distance = float('inf')
        closest_partner_id = None
        
        for partner in store_partners:
            # Distance from partner to order location
            distance = math.hypot(
                order_location[0] - partner['location'][0],
                order_location[1] - partner['location'][1]
            )
            if distance < min_distance:
                min_distance = distance
                closest_partner_id = partner['id']
        
        return closest_partner_id
    
    def calculate_route(self, start: Tuple[float, float], end: Tuple[float, float]) -> Tuple[List[Tuple[float, float]], float, float]:
        """Calculate route between two points (simplified with road factor)"""
        # Straight-line distance
        straight_distance = math.hypot(end[0] - start[0], end[1] - start[1])
        
        # Apply road factor (1.3x for urban areas)
        road_distance = straight_distance * 1.3
        
        # Calculate travel time (assume 30 km/h average)
        travel_time = (road_distance / 1000.0) / 30.0 * 60.0  # minutes
        
        # Create simple route (straight line for visualization)
        route = [start, end]
        
        return route, road_distance, travel_time
    
    def process_orders(self):
        """Process all orders: Order → Closest Store → Closest Partner → Customer"""
        print("🔄 Processing orders...")
        
        self.assignments = []
        
        for order in self.orders:
            print(f"   📦 Processing {order['id']}...")
            
            # Step 1: Find closest store
            closest_store_id = self.find_closest_store(order['location'])
            if not closest_store_id:
                print(f"      ❌ No store found for {order['id']}")
                continue
            
            # Step 2: Find closest partner to that store
            closest_partner_id = self.find_closest_partner(closest_store_id, order['location'])
            if not closest_partner_id:
                print(f"      ❌ No available partner found for {order['id']}")
                continue
            
            # Get partner and store details
            partner = next(p for p in self.partners if p['id'] == closest_partner_id)
            store = next(s for s in self.stores if s['id'] == closest_store_id)
            
            # Step 3: Calculate route: Partner → Store → Customer
            partner_to_store_route, partner_to_store_dist, partner_to_store_time = self.calculate_route(
                partner['location'], store['location']
            )
            
            store_to_customer_route, store_to_customer_dist, store_to_customer_time = self.calculate_route(
                store['location'], order['location']
            )
            
            # Combine routes
            full_route = partner_to_store_route[:-1] + store_to_customer_route
            total_distance = partner_to_store_dist + store_to_customer_dist
            total_time = partner_to_store_time + store_to_customer_time
            
            # Create assignment
            assignment = DeliveryAssignment(
                order_id=order['id'],
                store_id=closest_store_id,
                partner_id=closest_partner_id,
                route=full_route,
                total_time=total_time,
                total_distance=total_distance
            )
            
            self.assignments.append(assignment)
            
            # Mark partner as busy
            partner['status'] = 'busy'
            
            print(f"      ✅ {order['id']} → {closest_store_id} → {closest_partner_id}")
            print(f"         🕐 Time: {total_time:.1f} min, 📏 Distance: {total_distance/1000:.1f} km")
        
        print(f"   ✅ Processed {len(self.assignments)} orders successfully")
    
    def create_comprehensive_map(self, output_file: str = "delivery_map.png"):
        """Create comprehensive map visualization showing all elements"""
        print(f"🗺️  Creating comprehensive delivery map: {output_file}")
        
        # Set up the plot
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Plot roads if available
        if self.roads_gdf is not None:
            self.roads_gdf.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.7, label='Roads')
        
        # Plot stores
        store_x = [store['location'][0] for store in self.stores]
        store_y = [store['location'][1] for store in self.stores]
        ax.scatter(store_x, store_y, c='red', s=200, marker='s', label='Stores', 
                  edgecolors='darkred', linewidth=2, zorder=5)
        
        # Add store labels
        for store in self.stores:
            ax.annotate(store['id'], (store['location'][0], store['location'][1]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=10, 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Plot all delivery partners
        all_partner_x = [partner['location'][0] for partner in self.partners]
        all_partner_y = [partner['location'][1] for partner in self.partners]
        ax.scatter(all_partner_x, all_partner_y, c='blue', s=100, marker='o', 
                  label='All Partners', alpha=0.6, zorder=3)
        
        # Plot selected delivery partners (those assigned to orders)
        selected_partners = [assignment.partner_id for assignment in self.assignments]
        selected_partner_coords = []
        for partner in self.partners:
            if partner['id'] in selected_partners:
                selected_partner_coords.append(partner['location'])
        
        if selected_partner_coords:
            sel_x = [coord[0] for coord in selected_partner_coords]
            sel_y = [coord[1] for coord in selected_partner_coords]
            ax.scatter(sel_x, sel_y, c='green', s=150, marker='o', 
                      label='Selected Partners', edgecolors='darkgreen', linewidth=2, zorder=4)
        
        # Plot orders
        order_x = [order['location'][0] for order in self.orders]
        order_y = [order['location'][1] for order in self.orders]
        ax.scatter(order_x, order_y, c='orange', s=120, marker='^', 
                  label='Orders', edgecolors='darkorange', linewidth=2, zorder=4)
        
        # Add order labels
        for order in self.orders:
            ax.annotate(order['id'], (order['location'][0], order['location'][1]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9, 
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8))
        
        # Plot delivery routes
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.assignments)))
        for i, assignment in enumerate(self.assignments):
            route_x = [point[0] for point in assignment.route]
            route_y = [point[1] for point in assignment.route]
            
            # Plot route line
            ax.plot(route_x, route_y, color=colors[i], linewidth=3, alpha=0.8, 
                   label=f"Route {assignment.order_id}" if i < 5 else "")
            
            # Add arrows to show direction
            for j in range(len(route_x) - 1):
                dx = route_x[j+1] - route_x[j]
                dy = route_y[j+1] - route_y[j]
                ax.arrow(route_x[j], route_y[j], dx*0.3, dy*0.3, 
                        head_width=50, head_length=50, fc=colors[i], ec=colors[i], alpha=0.8)
        
        # Add partner-store connections (show which partners belong to which stores)
        for store in self.stores:
            store_partners = [p for p in self.partners if p['store_id'] == store['id']]
            for partner in store_partners:
                ax.plot([store['location'][0], partner['location'][0]], 
                       [store['location'][1], partner['location'][1]], 
                       'b--', alpha=0.3, linewidth=1)
        
        # Customize the plot
        ax.set_title('🚚 Final Delivery System - New Brunswick\n' + 
                    'Stores, Partners, Orders & Routes', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X Coordinate (meters)', fontsize=12)
        ax.set_ylabel('Y Coordinate (meters)', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Map saved as {output_file}")
        
        return fig, ax
    
    def generate_results_report(self, output_file: str = "results.py"):
        """Generate comprehensive results report with map output"""
        print(f"📊 Generating results report: {output_file}")
        
        # Create the results.py file
        results_content = f'''#!/usr/bin/env python3
"""
Delivery System Results Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def display_results():
    """Display comprehensive delivery system results"""
    
    print("🚚 FINAL DELIVERY SYSTEM RESULTS")
    print("=" * 50)
    
    # System Overview
    print(f"📊 SYSTEM OVERVIEW:")
    print(f"   🏪 Total Stores: 4")
    print(f"   🚚 Total Partners: 12")
    print(f"   📦 Total Orders: 8")
    print(f"   ✅ Successful Assignments: {len(self.assignments)}")
    print()
    
    # Store Details
    print(f"🏪 STORE DETAILS:")
    print(f"   Fredericton_Store: 3 partners")
    print(f"   Moncton_Store: 3 partners")
    print(f"   SaintJohn_Store: 3 partners")
    print(f"   Bathurst_Store: 3 partners")
    print()
    
    # Assignment Details
    print(f"📦 DELIVERY ASSIGNMENTS:")
    total_time = 0
    total_distance = 0
    
    for assignment in self.assignments:
        print(f"   {{assignment.order_id}}:")
        print(f"      🏪 Store: {{assignment.store_id}}")
        print(f"      🚚 Partner: {{assignment.partner_id}}")
        print(f"      🕐 Time: {{assignment.total_time:.1f}} minutes")
        print(f"      📏 Distance: {{assignment.total_distance/1000:.1f}} km")
        total_time += assignment.total_time
        total_distance += assignment.total_distance
        print()
    
    # Summary Statistics
    print(f"📈 SUMMARY STATISTICS:")
    print(f"   ⏱️  Total Delivery Time: {{total_time:.1f}} minutes")
    print(f"   📏 Total Distance: {{total_distance/1000:.1f}} km")
    print(f"   📊 Average Time per Order: {{total_time/len(self.assignments):.1f}} minutes")
    print(f"   📊 Average Distance per Order: {{total_distance/len(self.assignments)/1000:.1f}} km")
    print()
    
    # Partner Utilization
    print(f"🚚 PARTNER UTILIZATION:")
    selected_partners = set(assignment.partner_id for assignment in self.assignments)
    utilization_rate = len(selected_partners) / 12 * 100
    print(f"   📊 Utilization Rate: {{utilization_rate:.1f}}%")
    print(f"   ✅ Active Partners: {{len(selected_partners)}}")
    print(f"   😴 Available Partners: {{12 - len(selected_partners)}}")
    print()
    
    # Create visualization
    print("🗺️  Creating delivery map visualization...")
    create_delivery_map()
    
    print("✅ Results report completed!")

def create_delivery_map():
    """Create and display the delivery map"""
    # This function will be called to create the map
    # The actual map creation is handled by the main system
    pass

if __name__ == "__main__":
    display_results()
'''
        
        # Write the results file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(results_content)
        
        print(f"   ✅ Results report saved as {output_file}")
    
    def export_to_gpkg(self, output_file: str = "final_delivery_results.gpkg"):
        """Export all data to GeoPackage for GIS analysis"""
        print(f"💾 Exporting results to GeoPackage: {output_file}")
        
        try:
            # Prepare data for export
            store_records = []
            partner_records = []
            order_records = []
            route_records = []
            
            # Stores
            for store in self.stores:
                store_records.append({
                    'store_id': store['id'],
                    'name': store['name'],
                    'geometry': Point(store['location'])
                })
            
            # Partners
            for partner in self.partners:
                partner_records.append({
                    'partner_id': partner['id'],
                    'store_id': partner['store_id'],
                    'vehicle_type': partner['vehicle_type'],
                    'capacity': partner['capacity'],
                    'distance_from_store': partner['distance_from_store'],
                    'status': partner['status'],
                    'geometry': Point(partner['location'])
                })
            
            # Orders
            for order in self.orders:
                order_records.append({
                    'order_id': order['id'],
                    'demand': order['demand'],
                    'priority': order['priority'],
                    'created_at': order['created_at'].isoformat(),
                    'geometry': Point(order['location'])
                })
            
            # Routes
            for assignment in self.assignments:
                route_records.append({
                    'order_id': assignment.order_id,
                    'store_id': assignment.store_id,
                    'partner_id': assignment.partner_id,
                    'total_time': assignment.total_time,
                    'total_distance': assignment.total_distance,
                    'geometry': LineString(assignment.route)
                })
            
            # Create GeoDataFrames
            stores_gdf = gpd.GeoDataFrame(store_records, geometry='geometry', crs='EPSG:2953')
            partners_gdf = gpd.GeoDataFrame(partner_records, geometry='geometry', crs='EPSG:2953')
            orders_gdf = gpd.GeoDataFrame(order_records, geometry='geometry', crs='EPSG:2953')
            routes_gdf = gpd.GeoDataFrame(route_records, geometry='geometry', crs='EPSG:2953')
            
            # Export to GeoPackage
            stores_gdf.to_file(output_file, layer='stores', driver='GPKG')
            partners_gdf.to_file(output_file, layer='partners', driver='GPKG')
            orders_gdf.to_file(output_file, layer='orders', driver='GPKG')
            routes_gdf.to_file(output_file, layer='routes', driver='GPKG')
            
            print(f"   ✅ Successfully exported to {output_file}")
            
        except Exception as e:
            print(f"   ⚠️  Export failed: {e}")
            print("   📁 Exporting to separate files...")
            
            # Fallback: export to separate files
            stores_gdf.to_file("stores_only.gpkg", driver='GPKG')
            partners_gdf.to_file("partners_only.gpkg", driver='GPKG')
            orders_gdf.to_file("orders_only.gpkg", driver='GPKG')
            routes_gdf.to_file("routes_only.gpkg", driver='GPKG')
    
    def run_complete_system(self):
        """Run the complete delivery system"""
        print("🚀 FINAL DELIVERY SYSTEM - NEW BRUNSWICK")
        print("=" * 60)
        
        # Step 1: Add stores (using sample locations)
        sample_stores = [
            (2540000, 7385000),  # Fredericton area
            (2550000, 7370000),  # Moncton area  
            (2530000, 7360000),  # Saint John area
            (2560000, 7390000),  # Bathurst area
        ]
        store_names = ["Fredericton_Store", "Moncton_Store", "SaintJohn_Store", "Bathurst_Store"]
        
        self.add_stores(sample_stores, store_names)
        
        # Step 2: Add delivery partners (2-3 per store)
        self.add_delivery_partners(partners_per_store=3)
        
        # Step 3: Generate orders
        self.generate_orders(num_orders=8)
        
        # Step 4: Process orders
        self.process_orders()
        
        # Step 5: Create comprehensive map
        self.create_comprehensive_map("final_delivery_map.png")
        
        # Step 6: Generate results report
        self.generate_results_report("results.py")
        
        # Step 7: Export to GeoPackage
        self.export_to_gpkg("final_delivery_results.gpkg")
        
        print("\n✅ FINAL DELIVERY SYSTEM COMPLETED!")
        print("📁 Output files created:")
        print("   🗺️  final_delivery_map.png - Comprehensive map visualization")
        print("   📊 results.py - Results report and analysis")
        print("   💾 final_delivery_results.gpkg - GIS data export")
        
        return self.assignments

def main():
    """Main function to run the final delivery system"""
    system = FinalDeliverySystem()
    assignments = system.run_complete_system()
    
    # Display summary
    print(f"\n📈 FINAL SUMMARY:")
    print(f"   ✅ {len(assignments)} orders successfully processed")
    print(f"   🏪 {len(system.stores)} stores with {len(system.partners)} total partners")
    print(f"   🚚 {len(set(a.partner_id for a in assignments))} partners utilized")
    
    return system

if __name__ == "__main__":
    main()
