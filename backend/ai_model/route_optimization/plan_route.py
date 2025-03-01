import pandas as pd
import numpy as np
import gym
from gym import spaces
from sklearn.cluster import KMeans
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

# --------------------------
# Step 1: Load and parse dataset
# --------------------------
df = pd.read_csv('src/data/waste_collection.csv', delimiter=';')

def parse_geo_point(geo_str):
    try:
        lat_str, lon_str = geo_str.split(',')
        return float(lat_str.strip()), float(lon_str.strip())
    except:
        return np.nan, np.nan

df[['latitude', 'longitude']] = df['geo_point_2d'].apply(lambda x: pd.Series(parse_geo_point(x)))
df.dropna(subset=['latitude', 'longitude'], inplace=True)

# --------------------------
# Step 2: KMeans clustering
# --------------------------
X = df[['latitude', 'longitude']].values
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_
print("Cluster centroids (lat, lon):")
for i, c in enumerate(centroids):
    print(f"Cluster {i}: {c}")

# --------------------------
# Step 3: Custom Gym Environment
# --------------------------
class RouteOptimizationEnv(gym.Env):
    """
    The agent visits each cluster centroid exactly once.
    Reward structure:
      - Negative distance penalty (scaled)
      - Positive bonus for each new cluster visited
      - Large bonus upon visiting all clusters
      - Moderate penalty if cluster is revisited
    """
    def init(self, centroids):
        super(RouteOptimizationEnv, self).init()
        self.centroids = centroids
        self.n_clusters = len(centroids)

        # Action: choose next cluster (0..n_clusters-1)
        self.action_space = spaces.Discrete(self.n_clusters)

        # Observation: current lat/lon + visited mask
        self.observation_space = spaces.Box(
            low=-180, high=180,
            shape=(2 + self.n_clusters,),
            dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.visited = np.zeros(self.n_clusters, dtype=bool)
        # Start at a random cluster
        self.current_index = np.random.randint(self.n_clusters)
        self.visited[self.current_index] = True
        self.done_flag = False
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        lat_lon = self.centroids[self.current_index]
        visited_mask = self.visited.astype(float)
        return np.concatenate([lat_lon, visited_mask])

    def step(self, action):
        self.step_count += 1

        # If the chosen cluster is already visited
        if self.visited[action]:
            # moderate penalty, end the episode
            reward = -50
            self.done_flag = True
            return self._get_obs(), reward, self.done_flag, {}

        # Calculate distance between current and next cluster
        current_pos = self.centroids[self.current_index]
        next_pos = self.centroids[action]
        distance = np.linalg.norm(next_pos - current_pos)

        # Negative distance penalty (scaled)
        reward = -1.5 * distance
        # Bonus for new cluster
        reward += 10

        # Update state
        self.current_index = action
        self.visited[action] = True

        # If all clusters are visited
        if self.visited.all():
            reward += 100  # big completion bonus
            self.done_flag = True

        return self._get_obs(), reward, self.done_flag, {}

    def render(self, mode='human'):
        print(f"Step {self.step_count} | Current cluster: {self.current_index}, Visited: {self.visited}")

# --------------------------
# Step 4: Train RL with PPO
# --------------------------
env = RouteOptimizationEnv(centroids)
model = PPO("MlpPolicy", env, verbose=1)

# Train for 100,000 timesteps to give the agent more chance to learn
model.learn(total_timesteps=100000)

# Save the model
model.save("ecoWES_rl_model")
print("Model saved as 'ecoWES_rl_model.zip'")
# --------------------------
# Step 5: Evaluate the Model
# --------------------------
obs = env.reset()
done = False
route = [env.current_index]
total_reward = 0

while not done:
    action, _states = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    route.append(env.current_index)

print("Optimized route (cluster indices):", route)
print("Total reward:", total_reward)

# --------------------------
# Step 6: Plot the Route
# --------------------------
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue', 'orange', 'purple']

for i, c in enumerate(centroids):
    plt.scatter(c[1], c[0], color=colors[i % len(colors)], label=f'Cluster {i}')

# Plot the route with dashed line
route_coords = centroids[route]
plt.plot(route_coords[:, 1], route_coords[:, 0], 'k--', label="Route")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Improved RL Route among Waste Collection Clusters")
plt.legend()
plt.show()
