let map;
let stores = [];
let partners = [];
let chargingStations = [];
let markers = {
    stores: [],
    partners: [],
    chargingStations: [],
    customer: null,
    assignedStore: null,
    assignedPartner: null
};
let polylines = [];

const MARKER_ICONS = {
    walmart: 'http://maps.google.com/mapfiles/ms/icons/blue-dot.png',
    dollarama: 'http://maps.google.com/mapfiles/ms/icons/yellow-dot.png',
    sobeys: 'http://maps.google.com/mapfiles/ms/icons/red-dot.png',
    partner: 'http://maps.google.com/mapfiles/ms/icons/green-dot.png',
    charging: 'http://maps.google.com/mapfiles/ms/icons/ltblue-dot.png',
    customer: 'http://maps.google.com/mapfiles/ms/icons/pink-dot.png'
};

function initMap() {
    // Center on New Brunswick
    map = new google.maps.Map(document.getElementById('map'), {
        zoom: 7,
        center: { lat: 46.5, lng: -66.0 },
        mapTypeControl: true,
        fullscreenControl: true,
        streetViewControl: false,
        styles: [
            {
                featureType: 'poi',
                elementType: 'labels',
                stylers: [{ visibility: 'off' }]
            }
        ]
    });

    // Click listener for placing orders
    map.addListener('click', (event) => {
        placeOrder(event.latLng.lat(), event.latLng.lng());
    });

    // Load initial data
    loadData();
}

async function loadData() {
    try {
        const response = await fetch('/api/data');
        const data = await response.json();
        
        stores = data.stores;
        partners = data.partners;
        chargingStations = data.charging_stations || [];
        
        renderStores();
        renderPartners();
        renderChargingStations();
    } catch (error) {
        console.error('Error loading data:', error);
    }
}

function renderStores() {
    // Clear existing store markers
    markers.stores.forEach(m => m.setMap(null));
    markers.stores = [];
    
    stores.forEach(store => {
        const marker = new google.maps.Marker({
            position: { lat: store.lat, lng: store.lng },
            map: map,
            icon: MARKER_ICONS[store.chain.toLowerCase()],
            title: store.name,
            opacity: 0.7,
            optimized: true  // Optimize for performance
        });
        
        const infoWindow = new google.maps.InfoWindow({
            content: `<div style="padding:10px">
                <h3 style="margin:0 0 5px 0">${store.chain}</h3>
                <p style="margin:0">${store.name}</p>
            </div>`
        });
        
        marker.addListener('click', () => {
            infoWindow.open(map, marker);
        });
        
        markers.stores.push(marker);
    });
}

function renderPartners() {
    // Clear existing partner markers
    markers.partners.forEach(m => m.setMap(null));
    markers.partners = [];
    
    partners.forEach(partner => {
        const marker = new google.maps.Marker({
            position: { lat: partner.lat, lng: partner.lng },
            map: map,
            icon: {
                path: google.maps.SymbolPath.CIRCLE,
                scale: 8,
                fillColor: '#34a853',
                fillOpacity: 0.8,
                strokeColor: 'white',
                strokeWeight: 2
            },
            title: `${partner.name} (${partner.vehicle})`,
            opacity: 0.8,
            optimized: true  // Optimize for performance
        });
        
        const infoWindow = new google.maps.InfoWindow({
            content: `<div style="padding:10px">
                <h3 style="margin:0 0 5px 0">${partner.name}</h3>
                <p style="margin:0">Vehicle: ${partner.vehicle}</p>
                <p style="margin:0">Status: ${partner.status}</p>
            </div>`
        });
        
        marker.addListener('click', () => {
            infoWindow.open(map, marker);
        });
        
        markers.partners.push(marker);
    });
}

function renderChargingStations() {
    // Clear existing charging station markers
    markers.chargingStations.forEach(m => m.setMap(null));
    markers.chargingStations = [];
    
    chargingStations.forEach(station => {
        const marker = new google.maps.Marker({
            position: { lat: station.lat, lng: station.lng },
            map: map,
            icon: {
                path: google.maps.SymbolPath.CIRCLE,
                scale: 6,
                fillColor: '#2196F3',
                fillOpacity: 0.8,
                strokeColor: 'white',
                strokeWeight: 2
            },
            title: `⚡ ${station.name} (Rating: ${station.rating})`,
            opacity: 0.8,
            optimized: true  // Optimize for performance
        });
        
        const infoWindow = new google.maps.InfoWindow({
            content: `<div style="padding:10px">
                <h3 style="margin:0 0 5px 0">⚡ ${station.name}</h3>
                <p style="margin:0">Rating: ${station.rating}/5</p>
                <p style="margin:0">Status: ${station.status}</p>
            </div>`
        });
        
        marker.addListener('click', () => {
            infoWindow.open(map, marker);
        });
        
        markers.chargingStations.push(marker);
    });
}

async function placeOrder(lat, lng) {
    // Show loading
    document.getElementById('loading').classList.add('active');
    
    // Clear previous order markers and routes
    clearPreviousOrder();
    
    try {
        const response = await fetch('/api/place-order', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ lat, lng })
        });
        
        if (!response.ok) {
            throw new Error('Failed to place order');
        }
        
        const result = await response.json();
        
        // Display results
        displayOrderResults(result);
        
        // Visualize on map
        visualizeRoute(result);
        
    } catch (error) {
        console.error('Error placing order:', error);
        alert('Error placing order. Please try again.');
    } finally {
        document.getElementById('loading').classList.remove('active');
    }
}

function clearPreviousOrder() {
    // Remove customer marker
    if (markers.customer) {
        markers.customer.setMap(null);
        markers.customer = null;
    }
    
    // Remove polylines
    polylines.forEach(p => p.setMap(null));
    polylines = [];
    
    // Reset all store and partner opacities
    markers.stores.forEach(m => m.setOpacity(0.7));
    markers.partners.forEach(m => m.setOpacity(0.8));
}

function displayOrderResults(result) {
    const { assigned_store, assigned_partner, metrics } = result;
    
    // Show stats panels
    document.getElementById('stats-panel').style.display = 'block';
    document.getElementById('metrics-panel').style.display = 'block';
    
    // Update assignment details
    document.getElementById('assigned-store').textContent = assigned_store.name;
    document.getElementById('store-chain').textContent = assigned_store.chain;
    document.getElementById('assigned-partner').textContent = assigned_partner.name;
    document.getElementById('partner-vehicle').textContent = assigned_partner.vehicle;
    
    // Update metrics
    document.getElementById('partner-to-store-km').textContent = `${metrics.partner_to_store_km} km`;
    document.getElementById('partner-to-store-time').textContent = `${metrics.partner_to_store_min} min`;
    document.getElementById('store-to-customer-km').textContent = `${metrics.store_to_customer_km} km`;
    document.getElementById('store-to-customer-time').textContent = `${metrics.store_to_customer_min} min`;
    document.getElementById('total-distance').textContent = `${metrics.total_distance_km} km`;
    document.getElementById('estimated-time').textContent = `${metrics.estimated_time_min} min`;
}

function visualizeRoute(result) {
    const { order, assigned_store, assigned_partner, routes } = result;
    
    // Keep all stores visible (don't dim)
    // Only dim non-assigned partners
    markers.partners.forEach(m => m.setOpacity(0.3));
    
    // Place customer marker
    markers.customer = new google.maps.Marker({
        position: { lat: order.customer_location.lat, lng: order.customer_location.lng },
        map: map,
        icon: MARKER_ICONS.customer,
        title: 'Your Order',
        animation: google.maps.Animation.DROP
    });
    
    // Highlight assigned store
    const storeMarker = new google.maps.Marker({
        position: { lat: assigned_store.lat, lng: assigned_store.lng },
        map: map,
        icon: MARKER_ICONS[assigned_store.chain.toLowerCase()],
        title: assigned_store.name,
        opacity: 1.0,
        animation: google.maps.Animation.BOUNCE
    });
    setTimeout(() => storeMarker.setAnimation(null), 2000);
    
    // Highlight assigned partner
    const partnerMarker = new google.maps.Marker({
        position: { lat: assigned_partner.lat, lng: assigned_partner.lng },
        map: map,
        icon: {
            path: google.maps.SymbolPath.CIRCLE,
            scale: 10,
            fillColor: '#34a853',
            fillOpacity: 1.0,
            strokeColor: 'white',
            strokeWeight: 3
        },
        title: assigned_partner.name,
        animation: google.maps.Animation.BOUNCE
    });
    setTimeout(() => partnerMarker.setAnimation(null), 2000);
    
    // Draw routes using real road polylines from Google Directions API
    if (routes.partner_to_store_polyline) {
        // Partner to Store - decode actual road route
        const path1 = google.maps.geometry.encoding.decodePath(routes.partner_to_store_polyline);
        const line1 = new google.maps.Polyline({
            path: path1,
            geodesic: false,  // Use actual route, not geodesic
            strokeColor: '#34a853',
            strokeOpacity: 0.9,
            strokeWeight: 5,
            map: map
        });
        polylines.push(line1);
    } else {
        // Fallback to straight line if API fails
        const line1 = new google.maps.Polyline({
            path: [
                { lat: assigned_partner.lat, lng: assigned_partner.lng },
                { lat: assigned_store.lat, lng: assigned_store.lng }
            ],
            geodesic: true,
            strokeColor: '#34a853',
            strokeOpacity: 0.6,
            strokeWeight: 3,
            strokePattern: [10, 5],  // Dashed to show it's estimated
            map: map
        });
        polylines.push(line1);
    }
    
    if (routes.store_to_customer_polyline) {
        // Store to Customer - decode actual road route
        const path2 = google.maps.geometry.encoding.decodePath(routes.store_to_customer_polyline);
        const line2 = new google.maps.Polyline({
            path: path2,
            geodesic: false,  // Use actual route, not geodesic
            strokeColor: '#667eea',
            strokeOpacity: 0.9,
            strokeWeight: 5,
            map: map
        });
        polylines.push(line2);
    } else {
        // Fallback to straight line if API fails
        const line2 = new google.maps.Polyline({
            path: [
                { lat: assigned_store.lat, lng: assigned_store.lng },
                { lat: order.customer_location.lat, lng: order.customer_location.lng }
            ],
            geodesic: true,
            strokeColor: '#667eea',
            strokeOpacity: 0.6,
            strokeWeight: 3,
            strokePattern: [10, 5],  // Dashed to show it's estimated
            map: map
        });
        polylines.push(line2);
    }
    
    // Fit bounds to show entire route
    const bounds = new google.maps.LatLngBounds();
    bounds.extend({ lat: assigned_partner.lat, lng: assigned_partner.lng });
    bounds.extend({ lat: assigned_store.lat, lng: assigned_store.lng });
    bounds.extend({ lat: order.customer_location.lat, lng: order.customer_location.lng });
    map.fitBounds(bounds);
}

// Initialize on load
if (typeof google !== 'undefined') {
    google.maps.event.addDomListener(window, 'load', initMap);
}

