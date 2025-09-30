# New Brunswick Delivery Routing (Google Maps Simplified)

This simplified system leverages Google Maps Platform (Places, Distance Matrix, Directions) to fetch real store and EV charging locations across New Brunswick and compute distances and routes without maintaining a local road network.

## 🎯 Project Overview

This system implements a **Multi-Depot Vehicle Routing Problem (MDVRP)** solution that:

1. **Maps real store locations** (Walmart, Dollarama, Sobeys) across New Brunswick
2. **Creates delivery partner networks** distributed across the province
3. **Generates realistic customer orders** with priority levels
4. **Optimizes delivery routes** using advanced algorithms
5. **Exports results** to QGIS-ready format for visualization

## 🏗️ System Overview

- **Data Source**: Google Maps Places API (Walmart, Dollarama, Sobeys, EV charging)
- **Routing**: Distance Matrix API for times/distances; Directions API for polylines
- **Focus**: Your business logic (vehicles, partners, orders), not road graphs

## 📁 Project Structure

```
├── google_maps_vrp.py              # Google Maps integration utilities
├── requirements.txt                # Minimal dependencies
├── .env                            # GOOGLE_MAPS_API_KEY=...
├── nb_vrp_dataset/                 # Output JSONs (created on bootstrap)
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Google Maps Platform key with: Places, Distance Matrix, Directions enabled

### Installation
```bash
pip install -r requirements.txt
```

### Setup API Key
Create a `.env` file in the project root:
```
GOOGLE_MAPS_API_KEY=YOUR_API_KEY
```

### Usage
```bash
# 1) Fetch stores and EV charging locations across NB (JSON output)
python google_maps_vrp.py --bootstrap

# 2) Build a demo distance matrix over fetched stores
python google_maps_vrp.py --distance-matrix
```

## 📊 Outputs
- `nb_vrp_dataset/stores_walmart.json`
- `nb_vrp_dataset/stores_dollarama.json`
- `nb_vrp_dataset/stores_sobeys.json`
- `nb_vrp_dataset/ev_charging.json`
- `distance_matrix_demo.json`

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

## 🔧 Notes
- Respect Google Maps Places query quotas; the bootstrap process uses paging and tiling.
- Distance Matrix is chunked to comply with 25×25 limits per request.

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
googlemaps>=4.10.0
python-dotenv>=1.0.0
requests>=2.31.0
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