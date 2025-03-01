import React, { useEffect, useState } from "react";
import { Pie } from "react-chartjs-2";

const FuelMonitoring = () => {
  const [fuelLevels, setFuelLevels] = useState<
    { meter: string; level: number }[]
  >([]);

  useEffect(() => {
    fetch("/api/fuel")
      .then((res) => res.json())
      .then((data) => setFuelLevels(data.fuelLevels));
  }, []);

  const chartData = {
    labels: fuelLevels.map((f) => f.meter),
    datasets: [
      {
        label: "Fuel Levels",
        data: fuelLevels.map((f) => f.level),
        backgroundColor: ["#FF6384", "#36A2EB", "#FFCE56"],
      },
    ],
  };

  return (
    <div className="p-4 bg-white rounded-lg shadow-md">
      <h2 className="text-xl font-semibold mb-4">Fuel Monitoring</h2>
      <Pie data={chartData} />
    </div>
  );
};

export default FuelMonitoring;
