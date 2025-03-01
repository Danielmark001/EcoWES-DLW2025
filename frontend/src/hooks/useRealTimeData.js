import { useState, useEffect } from "react";

const useRealTimeData = (url) => {
  const [data, setData] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      const response = await fetch(url);
      const result = await response.json();
      setData(result);
    };

    fetchData();
  }, [url]);

  return data;
};

export default useRealTimeData;
