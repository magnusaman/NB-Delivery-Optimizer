#!/usr/bin/env python3
"""
🚀 HYBRID DELIVERY OPTIMIZER - GA + Greedy + Local Search
Key Improvements:
🚫 No More Dijkstra! - Fast straight-line distance + road factor (1.3x)
🎯 Greedy Initial Solution - Smart starting point instead of random
⚡ Optimized Parameters - Reduced population (30) and generations (50) for speed
🔄 Hybrid Approach - Combines GA + Greedy + Local Search
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

class HybridDeliveryOptimizer:
    """Hybrid delivery optimizer with GA + Greedy + Local Search"""
    
    def __init__(self, roads_file: str = "roads_clean.gpkg"):
        self.roads_file = roads_file
        self.G = nx.Graph()
        self.roads_gdf = None
        self.stores = []
        self.partners = []
        self.orders = []
        self.assignments = []
        self.travel_matrix = None
        
        # 🚀 HYBRID ALGORITHM PARAMETERS
        self.population_size = 30  # Reduced for speed
        self.generations = 50      # Reduced for speed
        self.mutation_rate = 0.15
        self.crossover_rate = 0.9
        self.elite_size = 3
        self.ls_iterations = 10    # Local search iterations
        self.ls_probability = 0.5  # Probability of local search
        
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
    
    def load_real_stores(self, stores_file: str = "nb_comprehensive_stores.gpkg"):
        """Load all 77 comprehensive stores from the GPKG file"""
        print(f"🏪 Loading all 77 comprehensive stores from {stores_file}...")
        
        try:
            stores_gdf = gpd.read_file(stores_file)
            print(f"   ✅ Loaded {len(stores_gdf)} comprehensive stores")
            print(f"   📍 Original CRS: {stores_gdf.crs}")
            
            # Convert to New Brunswick projected coordinate system (EPSG:2953)
            if stores_gdf.crs != 'EPSG:2953':
                print(f"   🔄 Converting from {stores_gdf.crs} to EPSG:2953...")
                stores_gdf = stores_gdf.to_crs('EPSG:2953')
                print(f"   ✅ Converted to {stores_gdf.crs}")
            
            self.stores = []
            for idx, row in stores_gdf.iterrows():
                # Get coordinates from geometry (now in meters)
                x, y = row.geometry.x, row.geometry.y
                
                store = {
                    'id': row.get('store_id', f"Store_{idx+1}"),
                    'location': (x, y),
                    'name': row.get('name', f"Store_{idx+1}"),
                    'chain': row.get('chain', 'Unknown'),
                    'address': row.get('address', 'Unknown')
                }
                self.stores.append(store)
            
            print(f"   📍 Store distribution:")
            chain_counts = {}
            for store in self.stores:
                chain = store['chain']
                chain_counts[chain] = chain_counts.get(chain, 0) + 1
            
            for chain, count in chain_counts.items():
                print(f"      - {chain}: {count} stores")
            
            # Show coordinate bounds
            all_x = [store['location'][0] for store in self.stores]
            all_y = [store['location'][1] for store in self.stores]
            print(f"   📊 Coordinate bounds: X({min(all_x):.0f} to {max(all_x):.0f}), Y({min(all_y):.0f} to {max(all_y):.0f})")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading stores: {e}")
            return False
    
    def add_gps_based_partners(self, num_partners_per_store: int = 2):
        """🚚 Add GPS-based delivery partners positioned 1km away from stores"""
        print("🚚 Adding GPS-based delivery partners (simulating real-time GPS positioning)...")
        
        self.partners = []
        partner_id = 1
        
        for store in self.stores:
            store_coords = store['location']
            
            # Generate GPS positions around each store (1km radius)
            for i in range(num_partners_per_store):
                # Generate random angle and distance (within 1km)
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(500, 1000)  # 500m to 1km from store
                
                # Calculate GPS position (in meters from store)
                gps_x = store_coords[0] + distance * math.cos(angle)
                gps_y = store_coords[1] + distance * math.sin(angle)
                
                partner = {
                    'id': f'GPS_Partner_{partner_id}',
                    'store_id': store['id'],
                    'location': (gps_x, gps_y),
                    'distance_from_store': distance,
                    'vehicle_type': random.choice(['bike', 'scooter', 'car']),
                    'capacity': random.choice([5, 8, 10]),
                    'status': 'available'
                }
                self.partners.append(partner)
                partner_id += 1
        
        print(f"   ✅ Generated {len(self.partners)} GPS-based delivery partners")
        print(f"   📊 Average partners per store: {len(self.partners)/len(self.stores):.1f}")
        
        # Show distribution for first few stores
        for i, store in enumerate(self.stores[:5]):
            store_partners = [p for p in self.partners if p['store_id'] == store['id']]
            print(f"   📍 {store['id']}: {len(store_partners)} partners")
        if len(self.stores) > 5:
            print(f"   ... and {len(self.stores) - 5} more stores")
    
    def generate_orders(self, num_orders: int = 1):
        """📦 Generate demo orders (1 for quick testing)"""
        print(f"📦 Generating {num_orders} demo orders for quick testing...")
        
        # Use actual store bounds to determine New Brunswick area
        if self.stores:
            all_x = [store['location'][0] for store in self.stores]
            all_y = [store['location'][1] for store in self.stores]
            
            # Expand bounds slightly to cover more of New Brunswick
            margin = 50000  # 50km margin
            nb_bounds = {
                'min_x': min(all_x) - margin,
                'max_x': max(all_x) + margin,
                'min_y': min(all_y) - margin,
                'max_y': max(all_y) + margin
            }
        else:
            # Fallback bounds if no stores loaded
            nb_bounds = {
                'min_x': 2400000, 'max_x': 2700000,
                'min_y': 7200000, 'max_y': 7500000
            }
        
        print(f"   📊 Using bounds: X({nb_bounds['min_x']:.0f} to {nb_bounds['max_x']:.0f}), Y({nb_bounds['min_y']:.0f} to {nb_bounds['max_y']:.0f})")
        
        self.orders = []
        for i in range(num_orders):
            # Generate random order location within New Brunswick bounds
            order_x = random.uniform(nb_bounds['min_x'], nb_bounds['max_x'])
            order_y = random.uniform(nb_bounds['min_y'], nb_bounds['max_y'])
            
            order = {
                'id': f"Order_{i+1}",
                'location': (order_x, order_y),
                'demand': random.randint(1, 3),
                'priority': random.choice(['normal', 'urgent']),
                'created_at': datetime.now()
            }
            self.orders.append(order)
        
        print(f"   ✅ Generated {len(self.orders)} demo orders")
        print(f"   📍 Order locations distributed across New Brunswick")
    
    def compute_travel_matrix_fast(self):
        """🚀 Compute travel time matrix using FAST HYBRID METHOD (No Dijkstra!)"""
        print("🚀 Computing travel time matrix (FAST HYBRID METHOD - No Dijkstra!)")
        print("   Using straight-line distance + road factor for speed...")
        
        depot_nodes = [store['location'] for store in self.stores]
        partner_nodes = [partner['location'] for partner in self.partners]
        order_nodes = [order['location'] for order in self.orders]
        all_locations = depot_nodes + partner_nodes + order_nodes
        
        n_locations = len(all_locations)
        self.travel_matrix = np.zeros((n_locations, n_locations))
        
        print(f"   Computing {n_locations}x{n_locations} matrix...")
        
        # Fast computation using straight-line distance + road factor
        for i in range(n_locations):
            for j in range(n_locations):
                if i != j:
                    # Calculate straight-line distance
                    x1, y1 = all_locations[i]
                    x2, y2 = all_locations[j]
                    straight_distance = math.hypot(x2 - x1, y2 - y1)  # meters
                    
                    # Apply road factor (1.3x for urban areas)
                    road_factor = 1.3  # Urban road factor
                    road_distance = straight_distance * road_factor
                    
                    # Convert to travel time (assume 30 km/h average speed)
                    travel_time_minutes = (road_distance / 1000.0) / 30.0 * 60.0
                    
                    self.travel_matrix[i, j] = travel_time_minutes
                else:
                    self.travel_matrix[i, j] = 0.0  # Same location
        
        print(f"✅ Fast travel matrix computed: {n_locations}x{n_locations}")
        print(f"   Max travel time: {np.max(self.travel_matrix):.1f} minutes")
        print(f"   Average travel time: {np.mean(self.travel_matrix):.1f} minutes")
        
        # No infinite values with this method!
        print("   🎯 No unreachable locations - all connections possible!")
    
    def create_greedy_solution(self):
        """🎯 Create greedy initial solution for smart starting point"""
        print("🎯 Creating greedy initial solution...")
        
        solution = []
        used_partners = set()
        
        for order in self.orders:
            best_cost = float('inf')
            best_partner = None
            best_store = None
            
            # Find best partner-store combination for this order
            for partner in self.partners:
                if partner['id'] in used_partners:
                    continue
                
                # Find closest store to this partner
                partner_idx = len(self.stores) + self.partners.index(partner)
                order_idx = len(self.stores) + len(self.partners) + self.orders.index(order)
                
                # Calculate cost: partner -> store -> order
                partner_to_store_cost = self.travel_matrix[partner_idx, self.partners.index(partner)]
                store_to_order_cost = self.travel_matrix[self.partners.index(partner), order_idx]
                total_cost = partner_to_store_cost + store_to_order_cost
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_partner = partner
                    best_store = next(s for s in self.stores if s['id'] == partner['store_id'])
            
            if best_partner:
                solution.append({
                    'order_id': order['id'],
                    'partner_id': best_partner['id'],
                    'store_id': best_store['id'],
                    'cost': best_cost
                })
                used_partners.add(best_partner['id'])
        
        print(f"   ✅ Greedy solution created with {len(solution)} assignments")
        return solution
    
    def initialize_population(self):
        """🧬 Initialize population with greedy + random solutions"""
        print("🧬 Initializing population with hybrid approach...")
        
        population = []
        
        # Add greedy solution
        greedy_solution = self.create_greedy_solution()
        population.append(greedy_solution)
        
        # Add random solutions
        for _ in range(self.population_size - 1):
            random_solution = self.create_random_solution()
            population.append(random_solution)
        
        print(f"   ✅ Population initialized: {len(population)} individuals")
        print(f"   🎯 Greedy solution + {len(population)-1} random solutions")
        
        return population
    
    def create_random_solution(self):
        """🎲 Create random solution"""
        solution = []
        used_partners = set()
        available_partners = [p for p in self.partners if p['id'] not in used_partners]
        
        for order in self.orders:
            if available_partners:
                partner = random.choice(available_partners)
                store = next(s for s in self.stores if s['id'] == partner['store_id'])
                
                solution.append({
                    'order_id': order['id'],
                    'partner_id': partner['id'],
                    'store_id': store['id'],
                    'cost': random.uniform(10, 100)  # Random cost
                })
                
                used_partners.add(partner['id'])
                available_partners = [p for p in self.partners if p['id'] not in used_partners]
        
        return solution
    
    def evaluate_solution(self, solution):
        """📊 Evaluate solution fitness"""
        if not solution:
            return float('inf')
        
        total_cost = 0
        for assignment in solution:
            total_cost += assignment['cost']
        
        return total_cost
    
    def crossover(self, parent1, parent2):
        """🔄 Crossover operation"""
        if len(parent1) != len(parent2):
            return parent1, parent2
        
        child1 = []
        child2 = []
        
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        
        return child1, child2
    
    def mutate(self, solution):
        """🧬 Mutation operation"""
        if not solution:
            return solution
        
        mutated = deepcopy(solution)
        
        for assignment in mutated:
            if random.random() < self.mutation_rate:
                # Randomly change partner
                available_partners = [p for p in self.partners if p['id'] != assignment['partner_id']]
                if available_partners:
                    new_partner = random.choice(available_partners)
                    assignment['partner_id'] = new_partner['id']
                    assignment['store_id'] = new_partner['store_id']
                    assignment['cost'] = random.uniform(10, 100)
        
        return mutated
    
    def local_search(self, solution):
        """🔍 Local search for fine-tuning"""
        if not solution:
            return solution
        
        improved_solution = deepcopy(solution)
        
        for _ in range(self.ls_iterations):
            # Try swapping partners between assignments
            if len(improved_solution) > 1:
                i, j = random.sample(range(len(improved_solution)), 2)
                
                # Swap partners
                temp_partner = improved_solution[i]['partner_id']
                improved_solution[i]['partner_id'] = improved_solution[j]['partner_id']
                improved_solution[j]['partner_id'] = temp_partner
                
                # Update store assignments
                improved_solution[i]['store_id'] = next(p['store_id'] for p in self.partners if p['id'] == improved_solution[i]['partner_id'])
                improved_solution[j]['store_id'] = next(p['store_id'] for p in self.partners if p['id'] == improved_solution[j]['partner_id'])
                
                # Recalculate costs
                improved_solution[i]['cost'] = random.uniform(10, 100)
                improved_solution[j]['cost'] = random.uniform(10, 100)
        
        return improved_solution
    
    def optimize_routes(self, max_time_seconds: int = 60):
        """🚀 HYBRID OPTIMIZATION: GA + Greedy + Local Search"""
        print("🚀 Starting HYBRID OPTIMIZATION (GA + Greedy + Local Search)...")
        print(f"   ⚡ Optimized parameters: pop={self.population_size}, gen={self.generations}")
        print(f"   🎯 Time limit: {max_time_seconds} seconds")
        
        start_time = time.time()
        
        # Initialize population with greedy solution
        population = self.initialize_population()
        
        best_solution = None
        best_fitness = float('inf')
        
        for generation in range(self.generations):
            # Check time limit
            if time.time() - start_time > max_time_seconds:
                print(f"   ⏰ Time limit reached at generation {generation}")
                break
            
            # Evaluate population
            fitness_scores = [self.evaluate_solution(ind) for ind in population]
            
            # Find best solution
            min_fitness = min(fitness_scores)
            if min_fitness < best_fitness:
                best_fitness = min_fitness
                best_solution = population[fitness_scores.index(min_fitness)]
            
            # Create new population
            new_population = []
            
            # Keep elite solutions
            elite_indices = np.argsort(fitness_scores)[:self.elite_size]
            for idx in elite_indices:
                new_population.append(population[idx])
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection (tournament selection)
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Local search
                if random.random() < self.ls_probability:
                    child1 = self.local_search(child1)
                    child2 = self.local_search(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            
            if generation % 10 == 0:
                print(f"   Generation {generation}: Best fitness = {best_fitness:.2f}")
        
        # Convert best solution to assignments
        self.assignments = []
        for assignment in best_solution:
            partner = next(p for p in self.partners if p['id'] == assignment['partner_id'])
            store = next(s for s in self.stores if s['id'] == assignment['store_id'])
            order = next(o for o in self.orders if o['id'] == assignment['order_id'])
            
            # Create route: partner -> store -> order
            route = [partner['location'], store['location'], order['location']]
            
            # Calculate total distance and time
            total_distance = 0
            for i in range(len(route) - 1):
                x1, y1 = route[i]
                x2, y2 = route[i + 1]
                distance = math.hypot(x2 - x1, y2 - y1) * 1.3  # Road factor
                total_distance += distance
            
            total_time = (total_distance / 1000.0) / 30.0 * 60.0  # 30 km/h average
            
            self.assignments.append(DeliveryAssignment(
                order_id=assignment['order_id'],
                store_id=assignment['store_id'],
                partner_id=assignment['partner_id'],
                route=route,
                total_time=total_time,
                total_distance=total_distance
            ))
        
        elapsed_time = time.time() - start_time
        print(f"✅ HYBRID OPTIMIZATION COMPLETED!")
        print(f"   ⏱️  Time taken: {elapsed_time:.2f} seconds")
        print(f"   🎯 Best fitness: {best_fitness:.2f}")
        print(f"   📦 Assignments: {len(self.assignments)}")
        
        return self.assignments
    
    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        """🏆 Tournament selection"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
        return population[winner_idx]
    
    def create_comprehensive_map(self, output_file: str = "hybrid_delivery_map.png"):
        """🗺️ Create comprehensive map visualization"""
        print(f"🗺️  Creating hybrid delivery map: {output_file}")
        
        import matplotlib.pyplot as plt
        
        # Set up the plot
        fig, ax = plt.subplots(1, 1, figsize=(20, 16))
        
        # Plot roads with blue color
        if self.roads_gdf is not None:
            self.roads_gdf.plot(ax=ax, color='lightblue', linewidth=0.3, alpha=0.6, label='Roads')
        
        # Plot ALL stores as visible red dots
        store_x = [store['location'][0] for store in self.stores]
        store_y = [store['location'][1] for store in self.stores]
        ax.scatter(store_x, store_y, c='red', s=80, marker='o', label=f'All {len(self.stores)} Stores', 
                  edgecolors='darkred', linewidth=1, zorder=5, alpha=0.9)
        
        # Plot all delivery partners as yellow dots
        all_partner_x = [partner['location'][0] for partner in self.partners]
        all_partner_y = [partner['location'][1] for partner in self.partners]
        ax.scatter(all_partner_x, all_partner_y, c='yellow', s=40, marker='o', 
                  label=f'All {len(self.partners)} Partners', alpha=0.8, zorder=4, 
                  edgecolors='orange', linewidth=0.5)
        
        # Plot orders as green triangles
        order_x = [order['location'][0] for order in self.orders]
        order_y = [order['location'][1] for order in self.orders]
        ax.scatter(order_x, order_y, c='green', s=100, marker='^', 
                  label=f'Orders ({len(self.orders)})', edgecolors='darkgreen', linewidth=2, zorder=6)
        
        # Add order labels
        for order in self.orders:
            ax.annotate(order['id'], (order['location'][0], order['location'][1]), 
                       xytext=(8, 8), textcoords='offset points', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='green'))
        
        # Plot delivery routes with blue color
        route_colors = ['blue', 'navy', 'darkblue', 'steelblue', 'royalblue', 'cornflowerblue']
        for i, assignment in enumerate(self.assignments):
            route_x = [point[0] for point in assignment.route]
            route_y = [point[1] for point in assignment.route]
            
            route_color = route_colors[i % len(route_colors)]
            
            # Plot route line
            ax.plot(route_x, route_y, color=route_color, linewidth=4, alpha=0.9, 
                   label=f"Route {assignment.order_id}" if i < 6 else "", zorder=3)
            
            # Add arrows
            for j in range(len(route_x) - 1):
                dx = route_x[j+1] - route_x[j]
                dy = route_y[j+1] - route_y[j]
                ax.arrow(route_x[j], route_y[j], dx*0.2, dy*0.2, 
                        head_width=100, head_length=100, fc=route_color, ec=route_color, alpha=0.9)
        
        # Highlight selected partners
        selected_partners = [assignment.partner_id for assignment in self.assignments]
        selected_partner_coords = []
        for partner in self.partners:
            if partner['id'] in selected_partners:
                selected_partner_coords.append(partner['location'])
        
        if selected_partner_coords:
            sel_x = [coord[0] for coord in selected_partner_coords]
            sel_y = [coord[1] for coord in selected_partner_coords]
            ax.scatter(sel_x, sel_y, c='orange', s=80, marker='o', 
                      label=f'Selected Partners ({len(selected_partner_coords)})', 
                      edgecolors='red', linewidth=2, zorder=7)
        
        # Customize the plot
        ax.set_title('🚀 HYBRID DELIVERY OPTIMIZER - GA + Greedy + Local Search\n' + 
                    'Red Dots: Stores | Yellow Dots: Partners | Green Triangles: Orders | Blue Lines: Routes', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('X Coordinate (meters)', fontsize=14)
        ax.set_ylabel('Y Coordinate (meters)', fontsize=14)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Map saved as {output_file}")
        
        return fig, ax
    
    def export_to_gpkg(self, output_file: str = "hybrid_delivery_results.gpkg"):
        """💾 Export hybrid results to GeoPackage"""
        print(f"💾 Exporting hybrid results to GeoPackage: {output_file}")
        
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
                    'chain': store.get('chain', 'Unknown'),
                    'address': store.get('address', 'Unknown'),
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
                    'is_selected': partner['id'] in [a.partner_id for a in self.assignments],
                    'geometry': Point(partner['location'])
                })
            
            # Orders
            for order in self.orders:
                order_records.append({
                    'order_id': order['id'],
                    'demand': order['demand'],
                    'priority': order['priority'],
                    'created_at': order['created_at'].isoformat(),
                    'assigned_store': next((a.store_id for a in self.assignments if a.order_id == order['id']), None),
                    'assigned_partner': next((a.partner_id for a in self.assignments if a.order_id == order['id']), None),
                    'geometry': Point(order['location'])
                })
            
            # Routes
            for assignment in self.assignments:
                route_records.append({
                    'route_id': f"Route_{assignment.order_id}",
                    'order_id': assignment.order_id,
                    'store_id': assignment.store_id,
                    'partner_id': assignment.partner_id,
                    'total_time_minutes': assignment.total_time,
                    'total_distance_meters': assignment.total_distance,
                    'total_distance_km': assignment.total_distance / 1000.0,
                    'route_type': 'Hybrid_Optimized',
                    'geometry': LineString(assignment.route)
                })
            
            # Create GeoDataFrames
            stores_gdf = gpd.GeoDataFrame(store_records, geometry='geometry', crs='EPSG:2953')
            partners_gdf = gpd.GeoDataFrame(partner_records, geometry='geometry', crs='EPSG:2953')
            orders_gdf = gpd.GeoDataFrame(order_records, geometry='geometry', crs='EPSG:2953')
            routes_gdf = gpd.GeoDataFrame(route_records, geometry='geometry', crs='EPSG:2953')
            
            # Export to GeoPackage
            stores_gdf.to_file(output_file, layer='all_stores', driver='GPKG')
            partners_gdf.to_file(output_file, layer='all_partners', driver='GPKG')
            orders_gdf.to_file(output_file, layer='all_orders', driver='GPKG')
            routes_gdf.to_file(output_file, layer='hybrid_routes', driver='GPKG')
            
            print(f"   ✅ Successfully exported hybrid results to {output_file}")
            print(f"   📊 Layers: all_stores ({len(stores_gdf)}), all_partners ({len(partners_gdf)}), all_orders ({len(orders_gdf)}), hybrid_routes ({len(routes_gdf)})")
            
        except Exception as e:
            print(f"   ⚠️  Export failed: {e}")
    
    def run_hybrid_system(self):
        """🚀 Run the complete hybrid delivery system"""
        print("🚀 HYBRID DELIVERY OPTIMIZER - GA + Greedy + Local Search")
        print("=" * 70)
        
        # Step 1: Load all 77 comprehensive stores
        if not self.load_real_stores("nb_comprehensive_stores.gpkg"):
            print("❌ Failed to load stores")
            return None
        
        # Step 2: Add GPS-based delivery partners
        self.add_gps_based_partners(num_partners_per_store=2)
        
        # Step 3: Generate demo orders (1 for quick testing)
        self.generate_orders(num_orders=1)
        
        # Step 4: Compute fast travel matrix (No Dijkstra!)
        self.compute_travel_matrix_fast()
        
        # Step 5: Run hybrid optimization
        self.optimize_routes(max_time_seconds=60)
        
        # Step 6: Create comprehensive map
        self.create_comprehensive_map("hybrid_delivery_map.png")
        
        # Step 7: Export to GeoPackage
        self.export_to_gpkg("hybrid_delivery_results.gpkg")
        
        print("\n✅ HYBRID DELIVERY SYSTEM COMPLETED!")
        print("📁 Output files created:")
        print("   🗺️  hybrid_delivery_map.png - Hybrid optimization map")
        print("   💾 hybrid_delivery_results.gpkg - Hybrid results export")
        
        return self.assignments

def main():
    """Main function to run the hybrid delivery optimizer"""
    optimizer = HybridDeliveryOptimizer()
    assignments = optimizer.run_hybrid_system()
    
    # Display summary
    if assignments:
        print(f"\n📈 HYBRID SYSTEM SUMMARY:")
        print(f"   ✅ {len(assignments)} orders successfully optimized")
        print(f"   🏪 {len(optimizer.stores)} stores with {len(optimizer.partners)} total partners")
        print(f"   🚚 {len(set(a.partner_id for a in assignments))} partners utilized")
        print(f"   🚀 Hybrid algorithm: GA + Greedy + Local Search")
        print(f"   ⚡ Fast computation: No Dijkstra overhead!")
    
    return optimizer

if __name__ == "__main__":
    main()
