#!/usr/bin/env python3
"""
New Brunswick Delivery Optimization System - Genetic Algorithm + Local Search Hybrid
A Blinkit-like delivery optimization model using GA + LS for New Brunswick, Canada
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
class Individual:
    """Represents a single solution (individual) in the genetic algorithm"""
    routes: List[List[int]]  # List of routes, each route is a list of order indices
    fitness: float = float('inf')
    total_distance: float = 0.0
    total_orders: int = 0
    is_feasible: bool = False

class GeneticDeliveryOptimizer:
    """Genetic Algorithm + Local Search hybrid for delivery optimization"""
    
    def __init__(self, roads_file: str = "roads_clean.gpkg"):
        self.roads_file = roads_file
        self.G = nx.Graph()
        self.roads_gdf = None
        self.depots = []
        self.couriers = []
        self.orders = []
        self.travel_matrix = None
        
        # GA Parameters (optimized for speed)
        self.population_size = 30  # Reduced for faster computation
        self.generations = 50      # Reduced for faster computation
        self.mutation_rate = 0.15  # Increased for more exploration
        self.crossover_rate = 0.9  # Increased for better mixing
        self.elite_size = 3        # Reduced for faster computation
        self.tournament_size = 3
        
        # Local Search Parameters (optimized)
        self.ls_iterations = 10    # Reduced for faster computation
        self.ls_probability = 0.5  # Increased for more local optimization
        
        # Results
        self.best_solution = None
        self.fitness_history = []
        
    def load_road_network(self):
        """Load and process the road network from GeoPackage"""
        print("Loading road network...")
        
        # Load roads
        self.roads_gdf = gpd.read_file(self.roads_file)
        print(f"Loaded {len(self.roads_gdf)} road segments")
        
        # Build graph from road segments
        print("Building road graph...")
        for idx, row in self.roads_gdf.iterrows():
            geom = row.geometry
            
            # Handle both LineString and MultiLineString
            if isinstance(geom, LineString):
                self._add_road_segment(geom, row)
            elif isinstance(geom, MultiLineString):
                for line in geom.geoms:
                    self._add_road_segment(line, row)
        
        print(f"Graph built with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        
    def _add_road_segment(self, line: LineString, road_data):
        """Add a road segment to the graph"""
        coords = list(line.coords)
        
        # Determine speed based on road type
        speed_kmh = self._get_road_speed(road_data)
        
        for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
            # Create node IDs (rounded coordinates)
            n1 = (round(x1, 2), round(y1, 2))
            n2 = (round(x2, 2), round(y2, 2))
            
            # Calculate distance
            dist = math.hypot(x2 - x1, y2 - y1)
            
            # Add nodes if they don't exist
            if n1 not in self.G:
                self.G.add_node(n1, x=x1, y=y1)
            if n2 not in self.G:
                self.G.add_node(n2, x=x2, y=y2)
            
            # Add edge (keep shortest if multiple edges exist)
            if self.G.has_edge(n1, n2):
                if dist < self.G[n1][n2]["length"]:
                    self.G[n1][n2]["length"] = dist
                    self.G[n1][n2]["speed_kmh"] = speed_kmh
            else:
                self.G.add_edge(n1, n2, length=dist, speed_kmh=speed_kmh)
    
    def _get_road_speed(self, road_data) -> float:
        """Determine speed limit based on road type"""
        speed_map = {
            'P1': 50,  # Primary roads
            'P2': 40,  # Secondary roads
            'P3': 30,  # Local roads
        }
        code = road_data.get('CODE', 'P2')
        return speed_map.get(code, 30)
    
    def add_depots(self, depot_locations: List[Tuple[float, float]], depot_names: List[str] = None):
        """Add Sobeys-like store locations (depots)"""
        if depot_names is None:
            depot_names = [f"Depot_{i+1}" for i in range(len(depot_locations))]
        
        self.depots = []
        for i, (x, y) in enumerate(depot_locations):
            snapped_node = self._snap_to_road(Point(x, y))
            depot = {
                'id': depot_names[i],
                'location': snapped_node,
                'original_coords': (x, y),
                'snapped_coords': snapped_node
            }
            self.depots.append(depot)
        
        print(f"Added {len(self.depots)} depots")
    
    def add_couriers(self, courier_configs: List[Dict]):
        """Add delivery couriers"""
        self.couriers = []
        for config in courier_configs:
            courier = {
                'id': config.get('id', f"Courier_{len(self.couriers)+1}"),
                'depot_id': config.get('depot_id'),
                'capacity': config.get('capacity', 10),
                'shift_start': config.get('shift_start', 0),
                'shift_end': config.get('shift_end', 480),
                'vehicle_type': config.get('vehicle_type', 'bike'),
                'gps_location': config.get('gps_location')  # GPS position of courier
            }
            self.couriers.append(courier)
        
        print(f"Added {len(self.couriers)} couriers")
    
    def add_gps_based_couriers(self, num_couriers_per_store: int = 1):
        """Add GPS-based couriers positioned 1km away from stores (simulating real GPS tracking)"""
        print("🚚 Adding GPS-based delivery partners (simulating real-time GPS positioning)...")
        
        courier_configs = []
        courier_id = 1
        
        for depot in self.depots:
            depot_coords = depot['snapped_coords']
            
            # Generate GPS positions around each store (1km radius)
            for i in range(num_couriers_per_store):
                # Generate random angle and distance (within 1km)
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(500, 1000)  # 500m to 1km from store
                
                # Calculate GPS position (in meters from depot)
                gps_x = depot_coords[0] + distance * math.cos(angle)
                gps_y = depot_coords[1] + distance * math.sin(angle)
                gps_location = self._snap_to_road(Point(gps_x, gps_y))
                
                # Create courier config with GPS position
                courier_configs.append({
                    'id': f'GPS_Partner_{courier_id}',
                    'depot_id': depot['id'],
                    'capacity': random.choice([8, 10, 12]),  # Variable capacity
                    'vehicle_type': random.choice(['bike', 'scooter', 'car']),
                    'gps_location': gps_location,
                    'distance_from_store_km': distance / 1000.0
                })
                courier_id += 1
        
        print(f"📍 Generated {len(courier_configs)} GPS-based delivery partners:")
        for config in courier_configs:
            print(f"  {config['id']}: {config['distance_from_store_km']:.1f}km from {config['depot_id']} ({config['vehicle_type']})")
        
        self.add_couriers(courier_configs)
    
    def generate_sample_orders(self, num_orders: int = 20, seed: int = 42):
        """Generate sample orders for testing"""
        random.seed(seed)
        np.random.seed(seed)
        
        all_nodes = list(self.G.nodes())
        self.orders = []
        
        for i in range(num_orders):
            random_node = random.choice(all_nodes)
            ready_time = random.randint(0, 300)
            service_time = random.randint(2, 10)
            demand = random.randint(1, 5)
            
            order = {
                'id': f"Order_{i+1}",
                'location': random_node,
                'ready_time': ready_time,
                'service_time': service_time,
                'demand': demand,
                'priority': random.choice(['normal', 'urgent']),
                'created_at': datetime.now()
            }
            self.orders.append(order)
        
        print(f"Generated {len(self.orders)} sample orders")
    
    def _snap_to_road(self, point: Point) -> Tuple[float, float]:
        """Snap a point to the nearest road node"""
        nodes = np.array(list(self.G.nodes()))
        if len(nodes) == 0:
            return (point.x, point.y)
        
        xs, ys = nodes[:, 0], nodes[:, 1]
        dx = xs - point.x
        dy = ys - point.y
        distances = dx * dx + dy * dy
        idx = int(np.argmin(distances))
        return tuple(nodes[idx])
    
    def compute_travel_matrix_fast(self):
        """Compute travel time matrix using fast straight-line + road factor (no Dijkstra!)"""
        print("🚀 Computing travel time matrix (FAST HYBRID METHOD - No Dijkstra!)")
        print("   Using straight-line distance + road factor for speed...")

        depot_nodes = [depot['snapped_coords'] for depot in self.depots]
        courier_nodes = [courier.get('gps_location', depot_nodes[0]) for courier in self.couriers]  # Use GPS locations
        order_nodes = [order['location'] for order in self.orders]
        all_locations = depot_nodes + courier_nodes + order_nodes

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
                    
                    # Apply road factor (1.3x for urban areas, 1.1x for highways)
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
    
    def _create_subgraph_around_locations(self, locations, buffer_km=20):
        """Create a subgraph around our locations for faster computation"""
        # Convert buffer from km to meters
        buffer_m = buffer_km * 1000
        
        # Find bounding box of all locations
        min_x = min(loc[0] for loc in locations) - buffer_m
        max_x = max(loc[0] for loc in locations) + buffer_m
        min_y = min(loc[1] for loc in locations) - buffer_m
        max_y = max(loc[1] for loc in locations) + buffer_m
        
        # Create subgraph with nodes in bounding box
        subgraph_nodes = []
        for node in self.G.nodes():
            x, y = node
            if min_x <= x <= max_x and min_y <= y <= max_y:
                subgraph_nodes.append(node)
        
        # Create subgraph
        subgraph = self.G.subgraph(subgraph_nodes).copy()
        
        # Ensure all our locations are in the subgraph
        for loc in locations:
            if loc not in subgraph.nodes:
                # Find nearest node and add it
                nearest = self._find_nearest_node(loc, subgraph_nodes)
                if nearest:
                    subgraph.add_node(loc, x=loc[0], y=loc[1])
                    # Connect to nearest node
                    if nearest in subgraph.nodes:
                        distance = math.hypot(loc[0] - nearest[0], loc[1] - nearest[1])
                        subgraph.add_edge(loc, nearest, length=distance, speed_kmh=30)
        
        return subgraph
    
    def _find_nearest_node(self, target_loc, candidate_nodes):
        """Find nearest node to target location"""
        if not candidate_nodes:
            return None
        
        min_dist = float('inf')
        nearest = None
        
        for node in candidate_nodes:
            dist = math.hypot(target_loc[0] - node[0], target_loc[1] - node[1])
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest

    
    def create_initial_population(self) -> List[Individual]:
        """Create initial population of solutions"""
        print("Creating initial population...")
        population = []
        
        for _ in range(self.population_size):
            individual = self._create_random_individual()
            self._evaluate_individual(individual)
            population.append(individual)
        
        return population
    
    def _create_random_individual(self) -> Individual:
        """Create a random individual (solution)"""
        # Shuffle all orders
        order_indices = list(range(len(self.orders)))
        random.shuffle(order_indices)
        
        # Distribute orders among couriers
        routes = []
        orders_per_courier = len(order_indices) // len(self.couriers)
        remainder = len(order_indices) % len(self.couriers)
        
        start_idx = 0
        for i, courier in enumerate(self.couriers):
            # Calculate how many orders this courier gets
            num_orders = orders_per_courier + (1 if i < remainder else 0)
            
            # Assign orders to this courier
            courier_orders = order_indices[start_idx:start_idx + num_orders]
            routes.append(courier_orders)
            start_idx += num_orders
        
        return Individual(routes=routes)
    
    def _evaluate_individual(self, individual: Individual):
        """Evaluate fitness of an individual"""
        total_distance = 0.0
        total_orders = 0
        penalty = 0.0
        
        for i, route in enumerate(individual.routes):
            if not route:  # Empty route
                continue
                
            courier = self.couriers[i]
            total_orders += len(route)
            
            # Calculate route distance
            route_distance = 0.0
            current_capacity = 0
            
            # Start from courier's GPS location (not depot)
            courier_gps_idx = len(self.depots) + i  # Courier GPS locations come after depots in matrix
            prev_location = courier_gps_idx
            
            for order_idx in route:
                order = self.orders[order_idx]
                current_capacity += order['demand']
                
                # Check capacity constraint
                if current_capacity > courier['capacity']:
                    penalty += 10000  # Heavy penalty for capacity violation
                
                # Add travel time (with bounds checking)
                order_location_idx = order_idx + len(self.depots) + len(self.couriers)  # Orders come after depots and couriers
                if (prev_location < self.travel_matrix.shape[0] and 
                    order_location_idx < self.travel_matrix.shape[1]):
                    travel_time = self.travel_matrix[prev_location, order_location_idx]
                    if np.isfinite(travel_time) and travel_time < 10000:  # Reasonable travel time
                        route_distance += travel_time
                    else:
                        penalty += 10000  # Penalty for unreachable locations
                else:
                    penalty += 10000  # Penalty for invalid indices
                
                prev_location = order_location_idx
            
            # Return to depot (with bounds checking)
            depot_idx = self._get_depot_index(courier['depot_id'])
            if (prev_location < self.travel_matrix.shape[0] and 
                depot_idx < self.travel_matrix.shape[1]):
                return_time = self.travel_matrix[prev_location, depot_idx]
                if np.isfinite(return_time) and return_time < 10000:  # Reasonable travel time
                    route_distance += return_time
                else:
                    penalty += 10000  # Penalty for unreachable depot
            
            total_distance += route_distance
        
        # Fitness = total distance + penalties
        individual.fitness = total_distance + penalty
        individual.total_distance = total_distance
        individual.total_orders = total_orders
        individual.is_feasible = penalty == 0
        
        return individual
    
    def _get_depot_index(self, depot_id: str) -> int:
        """Get depot index by ID"""
        for i, depot in enumerate(self.depots):
            if depot['id'] == depot_id:
                return i
        return 0
    
    def tournament_selection(self, population: List[Individual]) -> Individual:
        """Tournament selection for parent selection"""
        tournament = random.sample(population, self.tournament_size)
        return min(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Order crossover (OX) for route optimization"""
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
        
        # Flatten routes to get all orders
        orders1 = [order for route in parent1.routes for order in route]
        orders2 = [order for route in parent2.routes for order in route]
        
        # Ensure both parents have the same orders (handle missing orders)
        all_orders = set(orders1 + orders2)
        orders1 = [o for o in orders1 if o in all_orders]
        orders2 = [o for o in orders2 if o in all_orders]
        
        # If no orders, return parents
        if not orders1 and not orders2:
            return deepcopy(parent1), deepcopy(parent2)
        
        # Apply order crossover
        child1_orders = self._order_crossover(orders1, orders2) if orders1 else orders2.copy()
        child2_orders = self._order_crossover(orders2, orders1) if orders2 else orders1.copy()
        
        # Redistribute orders among couriers
        child1 = self._redistribute_orders(child1_orders)
        child2 = self._redistribute_orders(child2_orders)
        
        return child1, child2
    
    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Order crossover operation"""
        if len(parent1) < 2:
            return parent1.copy()
        
        # Select random segment from parent1
        start = random.randint(0, len(parent1) - 1)
        end = random.randint(start + 1, len(parent1))
        segment = parent1[start:end]
        
        # Create child by keeping segment and filling from parent2
        child = [-1] * len(parent1)
        child[start:end] = segment
        
        # Fill remaining positions from parent2
        parent2_idx = 0
        for i in range(len(child)):
            if child[i] == -1:
                # Find next element from parent2 that's not in segment
                while parent2_idx < len(parent2) and parent2[parent2_idx] in segment:
                    parent2_idx += 1
                
                # If we've exhausted parent2, fill with remaining elements from parent1
                if parent2_idx >= len(parent2):
                    # Find remaining elements from parent1 that aren't in segment
                    remaining = [x for x in parent1 if x not in segment]
                    if remaining:
                        child[i] = remaining[0]
                        segment.append(remaining[0])  # Add to segment to avoid duplicates
                else:
                    child[i] = parent2[parent2_idx]
                    parent2_idx += 1
        
        return child
    
    def _redistribute_orders(self, orders: List[int]) -> Individual:
        """Redistribute orders among couriers"""
        routes = []
        
        if not orders or not self.couriers:
            return Individual(routes=[[] for _ in self.couriers])
        
        orders_per_courier = len(orders) // len(self.couriers)
        remainder = len(orders) % len(self.couriers)
        
        start_idx = 0
        for i in range(len(self.couriers)):
            num_orders = orders_per_courier + (1 if i < remainder else 0)
            courier_orders = orders[start_idx:start_idx + num_orders] if start_idx < len(orders) else []
            routes.append(courier_orders)
            start_idx += num_orders
        
        return Individual(routes=routes)
    
    def mutate(self, individual: Individual) -> Individual:
        """Mutate individual using swap and relocate operators"""
        if random.random() > self.mutation_rate:
            return individual
        
        mutated = deepcopy(individual)
        
        # Choose mutation operator
        mutation_type = random.choice(['swap', 'relocate', 'reverse'])
        
        if mutation_type == 'swap':
            self._swap_mutation(mutated)
        elif mutation_type == 'relocate':
            self._relocate_mutation(mutated)
        elif mutation_type == 'reverse':
            self._reverse_mutation(mutated)
        
        return mutated
    
    def _swap_mutation(self, individual: Individual):
        """Swap two orders between routes or within a route"""
        if len(individual.routes) < 2:
            return
        
        # Find routes with orders
        non_empty_routes = [i for i, route in enumerate(individual.routes) if route]
        if len(non_empty_routes) < 2:
            return
        
        # Choose two different routes with orders
        route1_idx, route2_idx = random.sample(non_empty_routes, 2)
        route1, route2 = individual.routes[route1_idx], individual.routes[route2_idx]
        
        # Swap random orders between routes
        order1 = random.choice(route1)
        order2 = random.choice(route2)
        
        route1.remove(order1)
        route2.remove(order2)
        route1.append(order2)
        route2.append(order1)
    
    def _relocate_mutation(self, individual: Individual):
        """Move an order from one route to another"""
        if len(individual.routes) < 2:
            return
        
        # Find a route with orders
        source_routes = [i for i, route in enumerate(individual.routes) if route]
        if not source_routes:
            return
        
        source_idx = random.choice(source_routes)
        source_route = individual.routes[source_idx]
        
        # Choose order to move
        order_to_move = random.choice(source_route)
        source_route.remove(order_to_move)
        
        # Choose destination route (can be any route, including empty ones)
        dest_idx = random.choice([i for i in range(len(individual.routes)) if i != source_idx])
        individual.routes[dest_idx].append(order_to_move)
    
    def _reverse_mutation(self, individual: Individual):
        """Reverse a segment of a route"""
        # Choose a route with at least 2 orders
        valid_routes = [i for i, route in enumerate(individual.routes) if len(route) >= 2]
        if not valid_routes:
            return
        
        route_idx = random.choice(valid_routes)
        route = individual.routes[route_idx]
        
        # Choose segment to reverse
        start = random.randint(0, len(route) - 2)
        end = random.randint(start + 1, len(route))
        
        route[start:end] = route[start:end][::-1]
    
    def local_search(self, individual: Individual) -> Individual:
        """Local search improvement using 2-opt and relocate"""
        improved = deepcopy(individual)
        
        for _ in range(self.ls_iterations):
            # Choose improvement operator
            operator = random.choice(['2opt', 'relocate', 'swap'])
            
            if operator == '2opt':
                self._two_opt_improvement(improved)
            elif operator == 'relocate':
                self._relocate_improvement(improved)
            elif operator == 'swap':
                self._swap_improvement(improved)
            
            # Re-evaluate
            self._evaluate_individual(improved)
        
        return improved
    
    def _two_opt_improvement(self, individual: Individual):
        """2-opt improvement for a route"""
        for route in individual.routes:
            if len(route) < 4:
                continue
            
            # Try 2-opt moves
            for i in range(len(route) - 1):
                for j in range(i + 2, len(route)):
                    # Calculate current cost
                    current_cost = self._calculate_route_cost(route, 0)  # Use first courier as default
                    
                    # Try 2-opt swap
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    new_cost = self._calculate_route_cost(new_route, 0)
                    
                    if new_cost < current_cost:
                        route[:] = new_route
    
    def _relocate_improvement(self, individual: Individual):
        """Relocate improvement"""
        for i, route in enumerate(individual.routes):
            if len(route) < 2:
                continue
            
            for j, order in enumerate(route):
                # Try moving order to different position
                for k in range(len(route)):
                    if k != j:
                        # Calculate cost before move
                        current_cost = self._calculate_route_cost(route, 0)
                        
                        # Make move
                        route.pop(j)
                        route.insert(k, order)
                        
                        # Calculate cost after move
                        new_cost = self._calculate_route_cost(route, 0)
                        
                        # Revert if worse
                        if new_cost >= current_cost:
                            route.pop(k)
                            route.insert(j, order)
    
    def _swap_improvement(self, individual: Individual):
        """Swap improvement"""
        for route in individual.routes:
            if len(route) < 2:
                continue
            
            for i in range(len(route)):
                for j in range(i + 1, len(route)):
                    # Calculate current cost
                    current_cost = self._calculate_route_cost(route, 0)
                    
                    # Try swap
                    route[i], route[j] = route[j], route[i]
                    new_cost = self._calculate_route_cost(route, 0)
                    
                    # Revert if worse
                    if new_cost >= current_cost:
                        route[i], route[j] = route[j], route[i]
    
    def _calculate_route_cost(self, route: List[int], courier_idx: int = 0) -> float:
        """Calculate cost of a route"""
        if not route:
            return 0.0
        
        # Start from courier's GPS location
        courier_gps_idx = len(self.depots) + courier_idx
        cost = 0.0
        prev_location = courier_gps_idx
        
        for order_idx in route:
            order_location_idx = order_idx + len(self.depots) + len(self.couriers)
            if (prev_location < self.travel_matrix.shape[0] and 
                order_location_idx < self.travel_matrix.shape[1]):
                travel_cost = self.travel_matrix[prev_location, order_location_idx]
                if np.isfinite(travel_cost) and travel_cost < 10000:
                    cost += travel_cost
                else:
                    cost += 10000  # Penalty for unreachable
            prev_location = order_location_idx
        
        # Return to depot
        depot_idx = 0  # Simplified - use first depot
        if (prev_location < self.travel_matrix.shape[0] and 
            depot_idx < self.travel_matrix.shape[1]):
            return_cost = self.travel_matrix[prev_location, depot_idx]
            if np.isfinite(return_cost) and return_cost < 10000:
                cost += return_cost
            else:
                cost += 10000
        
        return cost
    
    def optimize_routes(self, max_time_seconds: int = 300):
        """Main optimization using HYBRID GA + Greedy Heuristic + Local Search"""
        print("🚀 Starting HYBRID ALGORITHM optimization...")
        print("   🔬 Algorithm: GA + Greedy Heuristic + Local Search")
        print(f"   📊 Population size: {self.population_size}")
        print(f"   🔄 Generations: {self.generations}")
        print(f"   🎯 Local search probability: {self.ls_probability}")
        
        start_time = time.time()
        
        # Step 1: Create greedy initial solution
        print("   🎯 Creating greedy initial solution...")
        greedy_solution = self._create_greedy_solution()
        self._evaluate_individual(greedy_solution)
        print(f"   📈 Greedy solution fitness: {greedy_solution.fitness:.2f}")
        
        # Step 2: Create initial population (mix of greedy and random)
        print("   🧬 Creating hybrid initial population...")
        population = self.create_initial_population()
        population.append(greedy_solution)  # Add greedy solution
        population.sort(key=lambda x: x.fitness)
        
        self.best_solution = population[0]
        self.fitness_history = [self.best_solution.fitness]
        
        print(f"   🏆 Initial best fitness: {self.best_solution.fitness:.2f}")
        
        # Step 3: Main GA loop with hybrid improvements
        for generation in range(self.generations):
            if time.time() - start_time > max_time_seconds:
                print(f"   ⏰ Time limit reached at generation {generation}")
                break
            
            # Create new population
            new_population = []
            
            # Elitism - keep best individuals
            new_population.extend(population[:self.elite_size])
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Local search (with probability)
                if random.random() < self.ls_probability:
                    child1 = self.local_search(child1)
                if random.random() < self.ls_probability:
                    child2 = self.local_search(child2)
                
                # Evaluate
                self._evaluate_individual(child1)
                self._evaluate_individual(child2)
                
                new_population.extend([child1, child2])
            
            # Update population
            population = new_population[:self.population_size]
            population.sort(key=lambda x: x.fitness)
            
            # Update best solution
            if population[0].fitness < self.best_solution.fitness:
                self.best_solution = population[0]
            
            self.fitness_history.append(self.best_solution.fitness)
            
            # Progress reporting
            if generation % 10 == 0:
                elapsed = time.time() - start_time
                print(f"   🔄 Generation {generation}: Best = {self.best_solution.fitness:.2f}, "
                      f"Orders = {self.best_solution.total_orders}, Time = {elapsed:.1f}s")
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ HYBRID optimization completed in {elapsed_time:.1f} seconds")
        print(f"   🏆 Final best fitness: {self.best_solution.fitness:.2f}")
        print(f"   📦 Total orders delivered: {self.best_solution.total_orders}")
        
        return self.best_solution
    
    def _create_greedy_solution(self) -> Individual:
        """Create a greedy initial solution for better starting point"""
        print("   🎯 Building greedy solution...")
        
        # Simple greedy: assign each order to the nearest available courier
        routes = [[] for _ in self.couriers]
        courier_capacities = [courier['capacity'] for courier in self.couriers]
        
        for order_idx in range(len(self.orders)):
            best_courier = -1
            best_cost = float('inf')
            
            # Find best courier for this order
            for courier_idx in range(len(self.couriers)):
                if courier_capacities[courier_idx] >= self.orders[order_idx]['demand']:
                    # Calculate cost to add this order to this courier
                    cost = self._calculate_order_assignment_cost(order_idx, courier_idx, routes[courier_idx])
                    if cost < best_cost:
                        best_cost = cost
                        best_courier = courier_idx
            
            # Assign order to best courier
            if best_courier != -1:
                routes[best_courier].append(order_idx)
                courier_capacities[best_courier] -= self.orders[order_idx]['demand']
        
        return Individual(routes=routes)
    
    def _calculate_order_assignment_cost(self, order_idx: int, courier_idx: int, existing_route: List[int]) -> float:
        """Calculate cost of assigning an order to a courier's route"""
        if not existing_route:
            # Empty route: cost = GPS to order + order to depot
            courier_gps_idx = len(self.depots) + courier_idx
            order_location_idx = order_idx + len(self.depots) + len(self.couriers)
            depot_idx = self._get_depot_index(self.couriers[courier_idx]['depot_id'])
            
            gps_to_order = self.travel_matrix[courier_gps_idx, order_location_idx]
            order_to_depot = self.travel_matrix[order_location_idx, depot_idx]
            return gps_to_order + order_to_depot
        else:
            # Non-empty route: cost = last order to new order + new order to depot
            last_order_idx = existing_route[-1]
            last_order_location_idx = last_order_idx + len(self.depots) + len(self.couriers)
            order_location_idx = order_idx + len(self.depots) + len(self.couriers)
            depot_idx = self._get_depot_index(self.couriers[courier_idx]['depot_id'])
            
            last_to_new = self.travel_matrix[last_order_location_idx, order_location_idx]
            new_to_depot = self.travel_matrix[order_location_idx, depot_idx]
            return last_to_new + new_to_depot
    
    def export_results(self, output_file: str = "genetic_delivery_routes.gpkg"):
        """Export results to GeoPackage"""
        print(f"Exporting results to {output_file}...")
        
        if not self.best_solution or not np.isfinite(self.best_solution.fitness):
            print("No valid solution to export!")
            return
        
        # Prepare data for export
        route_records = []
        depot_records = []
        courier_records = []
        order_records = []
        
        # Routes
        print("  Processing routes...")
        for i, route in enumerate(self.best_solution.routes):
            if route:  # Non-empty route
                print(f"    Processing route {i+1}/{len(self.best_solution.routes)}")
                try:
                    geometry = self._get_route_geometry(route, i)
                    if geometry:
                        route_records.append({
                            'courier_id': self.couriers[i]['id'],
                            'depot_id': self.couriers[i]['depot_id'],
                            'num_stops': len(route),
                            'route_cost': self._calculate_route_cost(route, i),
                            'geometry': geometry
                        })
                except Exception as e:
                    print(f"    Warning: Could not generate geometry for route {i+1}: {e}")
                    # Create a simple straight-line geometry as fallback
                    depot_coords = self.depots[self._get_depot_index(self.couriers[i]['depot_id'])]['snapped_coords']
                    coords = [depot_coords]
                    for order_idx in route:
                        coords.append(self.orders[order_idx]['location'])
                    coords.append(depot_coords)
                    route_records.append({
                        'courier_id': self.couriers[i]['id'],
                        'depot_id': self.couriers[i]['depot_id'],
                        'num_stops': len(route),
                        'route_cost': self._calculate_route_cost(route, i),
                        'geometry': LineString(coords)
                    })
        
        # Depots
        print("  Processing depots...")
        for depot in self.depots:
            depot_records.append({
                'depot_id': depot['id'],
                'geometry': Point(depot['snapped_coords'])
            })
        
        # Couriers (GPS positions)
        print("  Processing courier GPS positions...")
        for courier in self.couriers:
            gps_location = courier.get('gps_location', (0, 0))
            courier_records.append({
                'courier_id': courier['id'],
                'depot_id': courier['depot_id'],
                'vehicle_type': courier['vehicle_type'],
                'capacity': courier['capacity'],
                'distance_from_store': courier.get('distance_from_store_km', 0),
                'geometry': Point(gps_location)
            })
        
        # Orders
        print("  Processing orders...")
        for order in self.orders:
            order_records.append({
                'order_id': order['id'],
                'ready_time': order['ready_time'],
                'service_time': order['service_time'],
                'demand': order['demand'],
                'priority': order['priority'],
                'geometry': Point(order['location'])
            })
        
        # Create GeoDataFrames
        print("  Creating GeoDataFrames...")
        routes_gdf = gpd.GeoDataFrame(route_records, geometry='geometry', crs=self.roads_gdf.crs)
        depots_gdf = gpd.GeoDataFrame(depot_records, geometry='geometry', crs=self.roads_gdf.crs)
        couriers_gdf = gpd.GeoDataFrame(courier_records, geometry='geometry', crs=self.roads_gdf.crs)
        orders_gdf = gpd.GeoDataFrame(order_records, geometry='geometry', crs=self.roads_gdf.crs)
        
        # Export with fallback mechanism
        print("  Writing to file...")
        try:
            routes_gdf.to_file(output_file, layer='routes', driver='GPKG')
            depots_gdf.to_file(output_file, layer='depots', driver='GPKG')
            couriers_gdf.to_file(output_file, layer='couriers', driver='GPKG')
            orders_gdf.to_file(output_file, layer='orders', driver='GPKG')
            print(f"✅ Successfully exported all layers to {output_file}")
        except Exception as e:
            print(f"⚠️  Multi-layer export failed: {e}")
            print("📁 Exporting to separate files...")
            routes_gdf.to_file("gps_routes_only.gpkg", driver='GPKG')
            depots_gdf.to_file("gps_depots_only.gpkg", driver='GPKG')
            couriers_gdf.to_file("gps_couriers_only.gpkg", driver='GPKG')
            orders_gdf.to_file("gps_orders_only.gpkg", driver='GPKG')
        
        print(f"Results exported to {output_file}")
        self.print_summary()
    
    def _get_route_geometry(self, route, courier_idx):
        """Convert route to actual road geometry using shortest paths"""
        if not route:
            return None

        # Start from courier's GPS location, not depot
        courier_gps_coords = self.couriers[courier_idx].get('gps_location', (0, 0))
        depot_idx = self._get_depot_index(self.couriers[courier_idx]['depot_id'])
        depot_coords = self.depots[depot_idx]['snapped_coords']

        # Build path following actual roads
        path_coords = []
        prev = courier_gps_coords

        for order_idx in route:
            order_coords = self.orders[order_idx]['location']
            try:
                # Get shortest path on road network
                sp = nx.shortest_path(self.G, prev, order_coords, weight="length")
                # Convert node coordinates to actual coordinates
                for node in sp:
                    if node in self.G.nodes:
                        x, y = self.G.nodes[node]['x'], self.G.nodes[node]['y']
                        path_coords.append((x, y))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # Fallback: direct connection if no path found
                path_coords.append(order_coords)
            prev = order_coords

        # Return to depot
        try:
            sp = nx.shortest_path(self.G, prev, depot_coords, weight="length")
            for node in sp:
                if node in self.G.nodes:
                    x, y = self.G.nodes[node]['x'], self.G.nodes[node]['y']
                    path_coords.append((x, y))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path_coords.append(depot_coords)

        if len(path_coords) >= 2:
            return LineString(path_coords)
        return None

    
    def print_summary(self):
        """Print optimization summary"""
        if not self.best_solution:
            print("No solution found!")
            return
        
        print("\n" + "="*60)
        print("GENETIC ALGORITHM + LOCAL SEARCH OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"Algorithm: Hybrid GA + LS")
        print(f"Best fitness: {self.best_solution.fitness:.2f}")
        print(f"Total distance: {self.best_solution.total_distance:.2f} minutes")
        print(f"Total orders: {self.best_solution.total_orders}/{len(self.orders)}")
        print(f"Feasible solution: {self.best_solution.is_feasible}")
        
        print(f"\nRoute Details:")
        for i, route in enumerate(self.best_solution.routes):
            if route:
                courier = self.couriers[i]
                route_cost = self._calculate_route_cost(route, i)
                print(f"  {courier['id']} (from {courier['depot_id']}): "
                      f"{len(route)} orders, {route_cost:.1f} min")
        
        print(f"\nConvergence:")
        print(f"  Initial fitness: {self.fitness_history[0]:.2f}")
        print(f"  Final fitness: {self.fitness_history[-1]:.2f}")
        print(f"  Improvement: {((self.fitness_history[0] - self.fitness_history[-1]) / self.fitness_history[0] * 100):.1f}%")


def main():
    """Main function to run the GPS-based genetic delivery optimization"""
    print("🚚 GPS-BASED DELIVERY OPTIMIZATION SYSTEM - New Brunswick")
    print("📍 Simulating Real-Time GPS Tracking of Delivery Partners")
    print("="*80)
    
    # Initialize optimizer
    optimizer = GeneticDeliveryOptimizer("roads_clean.gpkg")
    
    # Load road network
    optimizer.load_road_network()
    
    # Fetch real store locations
    print("Fetching real store locations...")
    try:
        from accurate_store_fetcher import AccurateStoreFetcher
        fetcher = AccurateStoreFetcher()
        depot_locations, depot_names = fetcher.get_stores_for_optimization()
        
        print(f"Using {len(depot_locations)} real store locations")
        print("Sample locations:")
        for i, (loc, name) in enumerate(zip(depot_locations[:5], depot_names[:5])):
            print(f"  {name}: ({loc[0]:.0f}, {loc[1]:.0f})")
        
    except ImportError:
        print("Accurate fetcher not available, using sample locations...")
        depot_locations = [
            (2540000, 7385000),  # Fredericton area
            (2550000, 7370000),  # Moncton area  
            (2530000, 7360000),  # Saint John area
            (2560000, 7390000),  # Bathurst area
        ]
        depot_names = ["Fredericton_Sobeys", "Moncton_Sobeys", "SaintJohn_Sobeys", "Bathurst_Sobeys"]
    
    optimizer.add_depots(depot_locations, depot_names)
    
    # Add GPS-based couriers (simulating real GPS tracking)
    print("🌟 GPS-BASED DELIVERY SYSTEM DEMO")
    print("Simulating delivery partners positioned around stores based on GPS tracking...")
    optimizer.add_gps_based_couriers(num_couriers_per_store=1)  # 1 courier per store
    
    # Generate sample orders (just 1 for demo)
    print("📦 Generating demo order...")
    optimizer.generate_sample_orders(num_orders=1, seed=42)
    
    # Compute travel matrix (FAST HYBRID METHOD - No Dijkstra!)
    optimizer.compute_travel_matrix_fast()
    
    # Optimize routes using HYBRID GA + Greedy + LS (FAST!)
    best_solution = optimizer.optimize_routes(max_time_seconds=60)  # 1 minute (much faster now!)
    
    if best_solution:
        # Export results
        optimizer.export_results("New_routes.gpkg")
    else:
        print("Optimization failed!")


if __name__ == "__main__":
    main()
