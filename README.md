# New Brunswick Delivery Route Optimization System

A comprehensive delivery route optimization system for New Brunswick, Canada, similar to Blinkit, Uber Eats, and DoorDash. This system uses real store locations, delivery partners, and customer orders to create optimized delivery routes.

## 🎯 Project Overview

This system implements a **Multi-Depot Vehicle Routing Problem (MDVRP)** solution that:

1. **Maps real store locations** (Walmart, Dollarama, Sobeys) across New Brunswick
2. **Creates delivery partner networks** distributed across the province
3. **Generates realistic customer orders** with priority levels
4. **Optimizes delivery routes** using advanced algorithms
5. **Exports results** to QGIS-ready format for visualization

## 🏗️ System Architecture

### Phase 1: Data Foundation
- **Input**: Real store locations + Road network
- **Process**: Verification, geocoding, coordinate transformation
- **Output**: 46 verified stores + 60,738 road segments

### Phase 2: Network Construction
- **Input**: Road segments (MultiLineString geometries)
- **Process**: NetworkX graph construction with distance weights
- **Output**: 658,004 nodes, 662,819 edges (massive connected road network)

### Phase 3: Optimization Engine
- **Input**: Store locations + synthetic orders + distance matrix
- **Process**: OR-Tools constraint solver + Genetic Algorithm
- **Output**: Optimized delivery routes

### Phase 4: Visualization & Export
- **Input**: Optimized route sequences
- **Process**: Polyline reconstruction + GPKG export
- **Output**: QGIS-ready files for analysis

## 📁 Project Structure

```
├── fast_delivery_system.py          # Main delivery optimization system
├── genetic_delivery_optimizer.py    # Alternative genetic algorithm approach
├── nb_verified_stores.gpkg          # 46 verified store locations
├── roads_clean.gpkg                 # New Brunswick road network
├── routes_only.gpkg                 # Optimized delivery routes
├── New_routes.gpkg                  # Genetic algorithm results
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Required packages (see requirements.txt)

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd new-brunswick-delivery-optimization

# Install dependencies
pip install -r requirements.txt
```

### Running the System

#### Option 1: Fast Delivery System (Recommended)
```bash
python fast_delivery_system.py
```

#### Option 2: Genetic Algorithm System
```bash
python genetic_delivery_optimizer.py
```

## 📊 Results

### Fast Delivery System Results:
- **10 demo orders** processed successfully
- **Realistic delivery times**: 78-170 minutes (1.3-2.8 hours)
- **Realistic distances**: 39-85 km per route
- **Priority-based processing**: High priority orders processed first
- **Total system**: 511.7 km, 17.1 hours, $733 in orders

### Example Routes:
```
Route 1: PARTNER_22 → STORE_24 (Dollarama Miramichi) → Customer
- Distance: 41.0 km, Time: 82.0 min, Value: $148 (high priority)

Route 2: PARTNER_28 → STORE_4 (Walmart Fredericton) → Customer  
- Distance: 42.5 km, Time: 85.0 min, Value: $180 (high priority)
```

## 🗺️ Visualization

### In QGIS:
1. **Load `routes_only.gpkg`** → Shows delivery routes (colored lines)
2. **Load `nb_verified_stores.gpkg`** → Shows store locations (colored dots)
3. **Each route shows**: Partner → Store → Customer path
4. **Realistic distances and times** for each delivery

### What the Map Shows:
- **46 colored dots** = Store locations (Walmart=blue, Dollarama=green, Sobeys=red)
- **Colored lines** = Optimized delivery routes connecting stores to customers
- **Road network** = Real New Brunswick roads (background)

## 🔧 Technical Details

### Algorithms Used:
1. **Dijkstra's Algorithm** - Shortest path computation
2. **OR-Tools Constraint Solver** - Exact optimization
3. **Genetic Algorithm** - Heuristic optimization
4. **Local Search** - Route improvement

### Data Structures:
1. **NetworkX Graph** - Road network representation
2. **NumPy Arrays** - Distance matrices
3. **GeoPandas DataFrames** - Spatial data handling
4. **Shapely Geometries** - Spatial operations

### Coordinate Systems:
- **Input**: WGS84 (EPSG:4326) - Latitude/Longitude
- **Processing**: New Brunswick (EPSG:2953) - Meters
- **Output**: New Brunswick (EPSG:2953) - Meters

## 📈 Performance Metrics

- **Graph Construction**: ~30 seconds
- **Distance Matrix**: ~60 seconds (50x50)
- **OR-Tools Optimization**: <1 second
- **Genetic Algorithm**: ~1 second (100 generations)
- **Total Runtime**: ~2 minutes for complete optimization

## 🎯 Key Features

1. **Real-World Data**: Verified store locations from Google Maps
2. **Scalable Network**: 658K+ node road network
3. **Multiple Algorithms**: OR-Tools + Genetic Algorithm comparison
4. **Production Ready**: QGIS-compatible output
5. **Extensible**: Easy to add time windows, capacities, etc.

## 🚀 Future Enhancements

1. **Time Windows**: Add delivery time constraints
2. **Vehicle Capacities**: Add load capacity limits
3. **Dynamic Dispatch**: Real-time order assignment
4. **Traffic Data**: Incorporate real-time traffic conditions
5. **Cost Optimization**: Include fuel costs, driver wages
6. **Multi-Objective**: Balance distance vs. time vs. cost

## 📋 Requirements

```
geopandas>=0.13.0
shapely>=2.0.0
networkx>=3.0
ortools>=9.6.0
numpy>=1.24.0
pandas>=2.0.0
fiona>=1.9.0
pyproj>=3.5.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- New Brunswick road network data
- Google Maps for store location verification
- OpenStreetMap contributors
- OR-Tools team for optimization algorithms

## 📞 Contact

For questions or support, please open an issue in the GitHub repository.

---

**This system provides a solid foundation for real-world delivery route optimization in New Brunswick, with the flexibility to iterate and improve based on specific business requirements.**