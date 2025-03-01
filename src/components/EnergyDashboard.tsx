import React, { useEffect, useState } from "react";
import { Bar } from "react-chartjs-2";

const EnergyDashboard = () => {
  const [energyData, setEnergyData] = useState<number[]>([]);
  const [labels, setLabels] = useState<string[]>([]);

  useEffect(() => {
    fetch("/api/energy")
      .then((res) => res.json())
      .then((data) => {
        setEnergyData(data.energyUsage);
        setLabels(data.timestamps);
      });
  }, []);

  const chartData = {
    labels: labels,
    datasets: [
      {
        label: "Energy Consumption (kWh)",
        data: energyData,
        backgroundColor: "rgba(255, 99, 132, 0.5)",
      },
    ],
  };

  return (
    <div className="p-4 bg-white rounded-lg shadow-md">
      <h2 className="text-xl font-semibold mb-4">
        Energy Consumption & Forecasting
      </h2>
      <Bar data={chartData} />
    </div>
  );
};

export default EnergyDashboard;
