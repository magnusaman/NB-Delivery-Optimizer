import geopandas as gpd
import fiona

# list layers
layers = fiona.listlayers("New_routes.gpkg")
print(layers)

# read specific layer
routes = gpd.read_file("New_routes.gpkg", layer="routes")
print(routes.head())

# plot quick map
routes.plot()
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
roads = gpd.read_file("genetic_new_brunswick_delivery_routes.gpkg", layer="roads")
depots = gpd.read_file("genetic_new_brunswick_delivery_routes.gpkg", layer="depots")
orders = gpd.read_file("genetic_new_brunswick_delivery_routes.gpkg", layer="orders")

roads.plot(ax=ax, color="lightgray", linewidth=0.2)
routes.plot(ax=ax, color="red")
orders.plot(ax=ax, color="blue", markersize=5)
depots.plot(ax=ax, color="green", markersize=50, marker="*")
plt.show()
