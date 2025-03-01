import React, { useEffect, useState } from "react";
import { MapContainer, TileLayer, Marker, Polyline } from "react-leaflet";

const RouteOptimization = () => {
  const [routes, setRoutes] = useState<{ lat: number; lng: number }[]>([]);

  useEffect(() => {
    fetch("/api/routes")
      .then((res) => res.json())
      .then((data) => setRoutes(data.optimizedRoute));
  }, []);

  return (
    <div className="p-4 bg-white rounded-lg shadow-md">
      <h2 className="text-xl font-semibold mb-4">
        Garbage Truck Route Optimization
      </h2>
      <MapContainer
        center={[1.3521, 103.8198]}
        zoom={12}
        className="h-96 w-full"
      >
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        {routes.length > 0 && <Marker position={routes[0]} />}
        <Polyline positions={routes} color="blue" />
      </MapContainer>
    </div>
  );
};

export default RouteOptimization;
