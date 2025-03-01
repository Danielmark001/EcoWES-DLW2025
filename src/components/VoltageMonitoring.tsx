import React, { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";

const VoltageMonitoring = () => {
  const [voltageData, setVoltageData] = useState<number[]>([]);
  const [labels, setLabels] = useState<string[]>([]);

  useEffect(() => {
    fetch("/api/voltage")
      .then((res) => res.json())
      .then((data) => {
        setVoltageData(data.voltages);
        setLabels(data.timestamps);
      });
  }, []);

  const chartData = {
    labels: labels,
    datasets: [
      {
        label: "Voltage Levels (V)",
        data: voltageData,
        borderColor: "rgb(75, 192, 192)",
        backgroundColor: "rgba(75, 192, 192, 0.2)",
      },
    ],
  };

  return (
    <div className="p-4 bg-white rounded-lg shadow-md">
      <h2 className="text-xl font-semibold mb-4">Voltage Monitoring</h2>
      <Line data={chartData} />
    </div>
  );
};

export default VoltageMonitoring;
