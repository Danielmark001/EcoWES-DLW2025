import React from "react";
import VoltageMonitoring from "../components/VoltageMonitoring";
import EnergyDashboard from "../components/EnergyDashboard";
import FuelMonitoring from "../components/FuelMonitoring";
import RouteOptimization from "../components/RouteOptimization";

const Dashboard = () => {
  return (
    <div className="p-6 grid grid-cols-2 gap-4">
      <VoltageMonitoring />
      <EnergyDashboard />
      <FuelMonitoring />
      <RouteOptimization />
    </div>
  );
};

export default Dashboard;
