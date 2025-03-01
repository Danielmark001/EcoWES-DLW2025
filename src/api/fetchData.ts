export const fetchData = async (endpoint: string) => {
  try {
    const response = await fetch(`/api/${endpoint}`);
    const data = await response.json();
    return data;
  } catch (error) {
    console.error(`Error fetching ${endpoint}:`, error);
    return null;
  }
};
