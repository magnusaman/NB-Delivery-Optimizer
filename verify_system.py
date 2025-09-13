#!/usr/bin/env python3
"""
Verify the delivery system results
"""

import geopandas as gpd
import fiona

def verify_system():
    """Verify the delivery system results"""
    print("🔍 VERIFYING DELIVERY SYSTEM RESULTS")
    print("=" * 50)
    
    # Check GPKG file
    try:
        layers = fiona.listlayers('final_delivery_results.gpkg')
        print(f"📁 GPKG Layers: {layers}")
        
        # Check stores
        stores = gpd.read_file('final_delivery_results.gpkg', layer='all_stores')
        print(f"\n🏪 STORES VERIFICATION:")
        print(f"   Total stores: {len(stores)}")
        print(f"   Store IDs: {stores['store_id'].tolist()}")
        
        # Check partners
        partners = gpd.read_file('final_delivery_results.gpkg', layer='all_partners')
        print(f"\n🚚 PARTNERS VERIFICATION:")
        print(f"   Total partners: {len(partners)}")
        selected_partners = partners[partners['is_selected'] == True]
        print(f"   Selected partners: {len(selected_partners)}")
        print(f"   Selected partner IDs: {selected_partners['partner_id'].tolist()}")
        
        # Check orders
        orders = gpd.read_file('final_delivery_results.gpkg', layer='all_orders')
        print(f"\n📦 ORDERS VERIFICATION:")
        print(f"   Total orders: {len(orders)}")
        print(f"   Order IDs: {orders['order_id'].tolist()}")
        
        # Check routes
        routes = gpd.read_file('final_delivery_results.gpkg', layer='delivery_routes')
        print(f"\n🛣️ ROUTES VERIFICATION:")
        print(f"   Total routes: {len(routes)}")
        for idx, route in routes.iterrows():
            print(f"   {route['route_id']}: {route['partner_id']} → {route['store_id']} → {route['order_id']} ({route['total_distance_km']:.1f} km)")
        
        # Check road network
        if 'road_network' in layers:
            roads = gpd.read_file('final_delivery_results.gpkg', layer='road_network')
            print(f"\n🛣️ ROAD NETWORK:")
            print(f"   Road segments: {len(roads)}")
        
        print(f"\n✅ VERIFICATION COMPLETE!")
        print(f"   📊 All {len(stores)} stores are present in the GPKG file")
        print(f"   🚚 {len(selected_partners)} out of {len(partners)} partners are utilized")
        print(f"   📦 {len(orders)} orders processed")
        print(f"   🛣️ {len(routes)} delivery routes created")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    verify_system()
