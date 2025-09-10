#!/usr/bin/env python3
"""
New Brunswick Delivery Optimization System
A Blinkit-like delivery optimization model for New Brunswick, Canada
"""

import math
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString, MultiLineString
import fiona
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DeliveryOptimizer:
    """Main class for delivery optimization in New Brunswick"""
    
    def __init__(self, roads_file: str = "roads_clean.gpkg"):
        self.roads_file = roads_file
        self.G = nx.Graph()
        self.roads_gdf = None
        self.depots = []
        self.couriers = []
        self.orders = []
        self.travel_matrix = None
        self.solution = None
        
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
        # Map road types to speeds (km/h)
        speed_map = {
            'P1': 50,  # Primary roads
            'P2': 40,  # Secondary roads
            'P3': 30,  # Local roads
        }
        
        code = road_data.get('CODE', 'P2')
        return speed_map.get(code, 30)  # Default to 30 km/h
    
    def add_depots(self, depot_locations: List[Tuple[float, float]], depot_names: List[str] = None):
        """Add Sobeys-like store locations (depots)"""
        if depot_names is None:
            depot_names = [f"Depot_{i+1}" for i in range(len(depot_locations))]
        
        self.depots = []
        for i, (x, y) in enumerate(depot_locations):
            # Snap to nearest road node
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
                'shift_end': config.get('shift_end', 480),  # 8 hours in minutes
                'vehicle_type': config.get('vehicle_type', 'bike')
            }
            self.couriers.append(courier)
        
        print(f"Added {len(self.couriers)} couriers")
    
    def generate_sample_orders(self, num_orders: int = 50, seed: int = 42):
        """Generate sample orders for testing"""
        random.seed(seed)
        np.random.seed(seed)
        
        # Get all road nodes for random order placement
        all_nodes = list(self.G.nodes())
        
        self.orders = []
        for i in range(num_orders):
            # Random location
            random_node = random.choice(all_nodes)
            
            # Random order parameters
            ready_time = random.randint(0, 300)  # Ready within 5 hours
            service_time = random.randint(2, 10)  # 2-10 minutes service time
            demand = random.randint(1, 5)  # 1-5 units
            
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
    
    def compute_travel_matrix(self):
        """Compute travel time matrix between all depots and orders (optimized)"""
        print("Computing travel time matrix...")
        
        # Get all relevant locations
        depot_nodes = [depot['snapped_coords'] for depot in self.depots]
        order_nodes = [order['location'] for order in self.orders]
        all_locations = depot_nodes + order_nodes
        
        # Create index mapping
        self.location_index = {loc: i for i, loc in enumerate(all_locations)}
        n_locations = len(all_locations)
        
        # Initialize travel time matrix
        self.travel_matrix = np.full((n_locations, n_locations), np.inf)
        
        print(f"Computing {n_locations}x{n_locations} travel matrix...")
        
        # OPTIMIZATION 1: Use a smaller subgraph around our locations
        subgraph = self._create_subgraph_around_locations(all_locations)
        print(f"Using subgraph with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")
        
        # OPTIMIZATION 2: Compute only what we need
        for i, source in enumerate(all_locations):
            if i % 5 == 0:  # Progress indicator
                print(f"  Processing location {i+1}/{n_locations}")
            
            try:
                # Use subgraph for faster computation
                if source in subgraph.nodes:
                    path_lengths = nx.single_source_dijkstra_path_length(
                        subgraph, source, weight='length', cutoff=50000  # 50km cutoff
                    )
                else:
                    # Fallback to full graph if not in subgraph
                    path_lengths = nx.single_source_dijkstra_path_length(
                        self.G, source, weight='length', cutoff=50000
                    )
                
                # Fill matrix only for our locations
                for target, distance in path_lengths.items():
                    if target in self.location_index:
                        j = self.location_index[target]
                        # Use simple distance-to-time conversion for speed
                        time_minutes = self._simple_distance_to_time(distance)
                        self.travel_matrix[i, j] = time_minutes
                        
            except nx.NetworkXNoPath:
                continue
        
        # Fill diagonal with zeros
        np.fill_diagonal(self.travel_matrix, 0)
        
        print(f"Travel matrix computed: {n_locations}x{n_locations}")
    
    def _create_subgraph_around_locations(self, locations, buffer_km=10):
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
    
    def _simple_distance_to_time(self, distance_meters):
        """Simple distance to time conversion (faster than path-based)"""
        # Use average speed of 30 km/h
        speed_kmh = 30.0
        time_minutes = (distance_meters / 1000.0) / speed_kmh * 60.0
        return time_minutes
    
    def _distance_to_time(self, distance: float, source: Tuple, target: Tuple) -> float:
        """Convert distance to travel time in minutes"""
        # Try to get actual path for better speed estimation
        try:
            path = nx.shortest_path(self.G, source, target, weight='length')
            total_time = 0
            for i in range(len(path) - 1):
                edge_data = self.G[path[i]][path[i+1]]
                edge_length = edge_data['length']
                speed_kmh = edge_data.get('speed_kmh', 30)
                time_minutes = (edge_length / 1000.0) / speed_kmh * 60.0
                total_time += time_minutes
            return total_time
        except:
            # Fallback: use average speed
            avg_speed = 30  # km/h
            return (distance / 1000.0) / avg_speed * 60.0
    
    def optimize_routes(self, time_limit_seconds: int = 30):
        """Solve the multi-depot vehicle routing problem"""
        print("Optimizing delivery routes...")
        
        if self.travel_matrix is None:
            self.compute_travel_matrix()
        
        # Setup OR-Tools
        n_depots = len(self.depots)
        n_orders = len(self.orders)
        n_vehicles = len(self.couriers)
        
        # Create routing index manager
        # Nodes: depots first (0 to n_depots-1), then orders (n_depots to n_depots+n_orders-1)
        manager = pywrapcp.RoutingIndexManager(
            n_depots + n_orders, n_vehicles, 
            list(range(n_depots)),  # Start nodes (depots)
            list(range(n_depots))   # End nodes (same depots)
        )
        
        routing = pywrapcp.RoutingModel(manager)
        
        # Define transit callback
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(max(1, round(self.travel_matrix[from_node, to_node])))
        
        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Add time dimension
        routing.AddDimension(
            transit_callback_index,
            30,      # slack time
            8 * 60,  # maximum time per vehicle (8 hours)
            False,   # don't force start cumul to zero
            'Time'
        )
        
        time_dimension = routing.GetDimensionOrDie('Time')
        
        # Add capacity dimension
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            if from_node < n_depots:
                return 0  # Depots have no demand
            else:
                order_idx = from_node - n_depots
                return self.orders[order_idx]['demand']
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [courier['capacity'] for courier in self.couriers],  # vehicle capacities
            True,  # start cumul to zero
            'Capacity'
        )
        
        # Set time windows
        for depot_idx in range(n_depots):
            index = manager.NodeToIndex(depot_idx)
            time_dimension.CumulVar(index).SetRange(0, 8 * 60)  # 8-hour window
        
        for order_idx in range(n_orders):
            node = n_depots + order_idx
            index = manager.NodeToIndex(node)
            order = self.orders[order_idx]
            ready_time = order['ready_time']
            due_time = ready_time + 120  # 2-hour delivery window
            time_dimension.CumulVar(index).SetRange(ready_time, due_time)
        
        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(time_limit_seconds)
        
        # Solve
        self.solution = routing.SolveWithParameters(search_parameters)
        
        if self.solution:
            print("Solution found!")
            self._extract_routes(manager, routing)
        else:
            print("No solution found!")
        
        return self.solution is not None
    
    def _extract_routes(self, manager, routing):
        """Extract routes from the solution"""
        self.courier_routes = []
        
        for vehicle_id in range(len(self.couriers)):
            route = []
            index = routing.Start(vehicle_id)
            
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                index = self.solution.Value(routing.NextVar(index))
            
            # Add end node
            node = manager.IndexToNode(index)
            route.append(node)
            
            self.courier_routes.append(route)
    
    def get_route_geometry(self, route: List[int]) -> LineString:
        """Convert route to actual road geometry"""
        if len(route) < 2:
            return None
        
        # Get all locations
        depot_nodes = [depot['snapped_coords'] for depot in self.depots]
        order_nodes = [order['location'] for order in self.orders]
        all_locations = depot_nodes + order_nodes
        
        # Build path through road network
        path_coords = []
        
        for i in range(len(route) - 1):
            start_node = all_locations[route[i]]
            end_node = all_locations[route[i + 1]]
            
            try:
                # Get shortest path between nodes
                shortest_path = nx.shortest_path(
                    self.G, start_node, end_node, weight='length'
                )
                
                # Add coordinates
                for node in shortest_path:
                    if node in self.G.nodes:
                        x, y = self.G.nodes[node]['x'], self.G.nodes[node]['y']
                        path_coords.append((x, y))
                        
            except nx.NetworkXNoPath:
                # Direct connection if no path found
                if start_node in self.G.nodes and end_node in self.G.nodes:
                    x1, y1 = self.G.nodes[start_node]['x'], self.G.nodes[start_node]['y']
                    x2, y2 = self.G.nodes[end_node]['x'], self.G.nodes[end_node]['y']
                    path_coords.extend([(x1, y1), (x2, y2)])
        
        if len(path_coords) >= 2:
            return LineString(path_coords)
        return None
    
    def export_results(self, output_file: str = "delivery_routes.gpkg"):
        """Export results to GeoPackage"""
        print(f"Exporting results to {output_file}...")
        
        # Prepare data for export
        route_records = []
        depot_records = []
        order_records = []
        
        # Routes
        for i, route in enumerate(self.courier_routes):
            if len(route) > 2:  # Valid route
                geometry = self.get_route_geometry(route)
                if geometry:
                    route_records.append({
                        'courier_id': self.couriers[i]['id'],
                        'depot_id': self.couriers[i]['depot_id'],
                        'num_stops': len(route) - 2,  # Exclude start/end depot
                        'geometry': geometry
                    })
        
        # Depots
        for depot in self.depots:
            depot_records.append({
                'depot_id': depot['id'],
                'geometry': Point(depot['snapped_coords'])
            })
        
        # Orders
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
        routes_gdf = gpd.GeoDataFrame(route_records, geometry='geometry', crs=self.roads_gdf.crs)
        depots_gdf = gpd.GeoDataFrame(depot_records, geometry='geometry', crs=self.roads_gdf.crs)
        orders_gdf = gpd.GeoDataFrame(order_records, geometry='geometry', crs=self.roads_gdf.crs)
        
        # Export
        routes_gdf.to_file(output_file, layer='routes', driver='GPKG')
        depots_gdf.to_file(output_file, layer='depots', driver='GPKG')
        orders_gdf.to_file(output_file, layer='orders', driver='GPKG')
        self.roads_gdf.to_file(output_file, layer='roads', driver='GPKG')
        
        print(f"Results exported to {output_file}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print optimization summary"""
        if not self.solution:
            print("No solution found!")
            return
        
        print("\n" + "="*50)
        print("DELIVERY OPTIMIZATION SUMMARY")
        print("="*50)
        
        total_distance = 0
        total_orders = 0
        
        for i, route in enumerate(self.courier_routes):
            if len(route) > 2:
                courier = self.couriers[i]
                num_orders = len(route) - 2
                total_orders += num_orders
                
                # Calculate route distance
                route_distance = 0
                for j in range(len(route) - 1):
                    route_distance += self.travel_matrix[route[j], route[j + 1]]
                
                total_distance += route_distance
                
                print(f"\n{courier['id']} (from {courier['depot_id']}):")
                print(f"  Orders delivered: {num_orders}")
                print(f"  Route time: {route_distance:.1f} minutes")
                print(f"  Route: {' -> '.join([str(route[k]) for k in range(min(5, len(route)))])}{'...' if len(route) > 5 else ''}")
        
        print(f"\nTOTAL:")
        print(f"  Orders delivered: {total_orders}/{len(self.orders)}")
        print(f"  Total distance: {total_distance:.1f} minutes")
        print(f"  Average time per order: {total_distance/max(total_orders, 1):.1f} minutes")


def main():
    """Main function to run the delivery optimization"""
    print("New Brunswick Delivery Optimization System")
    print("="*50)
    
    # Initialize optimizer
    optimizer = DeliveryOptimizer("roads_clean.gpkg")
    
    # Load road network
    optimizer.load_road_network()
    
    # Add sample depots (Sobeys-like stores in major NB cities)
    # These are approximate coordinates for major cities in New Brunswick
    depot_locations = [
        (2540000, 7385000),  # Fredericton area
        (2550000, 7370000),  # Moncton area  
        (2530000, 7360000),  # Saint John area
        (2560000, 7390000),  # Bathurst area
    ]
    depot_names = ["Fredericton_Sobeys", "Moncton_Sobeys", "SaintJohn_Sobeys", "Bathurst_Sobeys"]
    
    optimizer.add_depots(depot_locations, depot_names)
    
    # Add couriers
    courier_configs = [
        {'id': 'C1', 'depot_id': 'Fredericton_Sobeys', 'capacity': 8, 'vehicle_type': 'bike'},
        {'id': 'C2', 'depot_id': 'Fredericton_Sobeys', 'capacity': 10, 'vehicle_type': 'car'},
        {'id': 'C3', 'depot_id': 'Moncton_Sobeys', 'capacity': 8, 'vehicle_type': 'bike'},
        {'id': 'C4', 'depot_id': 'Moncton_Sobeys', 'capacity': 12, 'vehicle_type': 'car'},
        {'id': 'C5', 'depot_id': 'SaintJohn_Sobeys', 'capacity': 6, 'vehicle_type': 'bike'},
        {'id': 'C6', 'depot_id': 'Bathurst_Sobeys', 'capacity': 8, 'vehicle_type': 'bike'},
    ]
    
    optimizer.add_couriers(courier_configs)
    
    # Generate sample orders (start with fewer for faster testing)
    optimizer.generate_sample_orders(num_orders=15, seed=42)
    
    # Optimize routes
    success = optimizer.optimize_routes(time_limit_seconds=60)
    
    if success:
        # Export results
        optimizer.export_results("new_brunswick_delivery_routes.gpkg")
    else:
        print("Optimization failed!")


if __name__ == "__main__":
    main()
