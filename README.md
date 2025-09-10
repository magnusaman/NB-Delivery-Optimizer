# NB-Delivery-Optimizer
# New Brunswick Delivery Optimizer (Hybrid GA + Local Search)

🚴 A Blinkit-style delivery route optimizer for New Brunswick, Canada.  
Solves a **Multi-Depot Vehicle Routing Problem (MDVRP)** using a hybrid Genetic Algorithm + Local Search.

## How it works
- Input: Road network (`roads_clean.gpkg`), depots, couriers, orders.
- Builds road graph (~60k segments, 650k nodes).
- Generates orders (synthetic).
- Computes travel-time matrix (Euclidean approx).
- Runs GA+LS to assign orders to couriers & optimize routes.
- Outputs results as a GeoPackage (`genetic_new_brunswick_delivery_routes.gpkg`) with layers:
  - `roads` (base network)
  - `depots` (Sobeys stores)
  - `orders` (customer points)
  - `routes` (optimized courier paths)

## Results (Sample Run)
- Initial fitness: 9424.58
- Final fitness: 2045.32 (**78.3% improvement**)
- Orders delivered: 20/20
- Runtime: ~1.7s
- Exported routes visible in QGIS.

## Next Steps
- Replace Euclidean travel with shortest-path on road graph.
- Add load-balancing penalties (fairer distribution).
- Compare with Google OR-Tools (exact solver).
- Scale to hundreds of orders.

## Run
```bash
pip install -r requirements.txt
python genetic_delivery_optimizer.py
