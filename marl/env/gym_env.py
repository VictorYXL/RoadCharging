import numpy as np
import gym
import random
import time
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
import pandas as pd
import json
import os

from gym import Env, spaces

class RoadCharging(Env):
	def __init__(self, config_fname: str):
		super(RoadCharging, self).__init__()

		# Read configuration data from the JSON file
		with open(config_fname, "r") as file:
			config = json.load(file)
			
		self.n = config["fleet_size"]  # Number of EVs (agents) in the fleet
		self.m = config["n_chargers"]  # Number of available chargers
		self.k = config["max_time_step"]  # Maximum number of time steps
		self.delta_t = config["time_step_size"]  # Duration of one time step (in minutes)
		self.h = config["connection_fee($)"]  # First-time connection fee
		self.max_cap = config["max_cap"]  # Maximum battery capacity (kWh)
		self.low_SoC = config["low_SoC"]  # Threshold for low battery SoC (e.g., 10%)
		self.initial_SoCs = config["initial_SoCs"]  # Initial SoC values for each EV
		self.d_rates = config["d_rates(%)"]  # Discharging rates (as percentages)
		self.c_rates = config["c_rates(%)"]  # Charging rates (as percentages)
		self.c_r = config["charging_powers(kWh)"]  # Charging power in kWh
		self.w = config["w"]  # Payment per unit of ride time
		self.rho = config["rho"]  # Probability of order assignment
		self.p = config["p"]  # Charging prices
		self.ride_time_instance = np.array(config["ride_data_instance"])  # Ride time data sampled from distribution
		
		self.ride_data_type = config["ride_data_type"]
		self.charging_data_type = config["charging_data_type"]
		self.charging_data_type = config["charging_data_type"]
		self.payment_rates_24hrs = config["payment_rates_data($)"][self.ride_data_type]
		self.assign_probs_24hrs = config["order_assign_probs_data"][self.ride_data_type+f"_{self.delta_t}"]
		self.charging_prices_24hrs = config["charging_prices($/kWh)"]
		self.ride_time_probs_data = pd.DataFrame(config["ride_time_probs_data"])
		self.ride_scenario_probs = self.ride_time_probs_data[self.ride_data_type].values
		self.config = config
  
		self.stoch_simulate_future = False
		self.stoch_step = False
		# self.stoch_step = True


		# Observation space: n agents, each with 4 state variables
		self.observation_shape = (self.n, 4)

		# Define the observation space for each agent
		self.observation_space = spaces.Dict({
			"TimeStep": spaces.MultiDiscrete([self.k + 1] * self.n),  # Time step for each agent (0 to k)
			"RideTime": spaces.MultiDiscrete([self.k + 1] * self.n),  # Ride time for each agent (0 to k)
			"ChargingStatus": spaces.MultiDiscrete([2] * self.n),  # Charging status: 0 (not charging) or 1 (charging)
			"SoC": spaces.Box(low=0.0, high=1.0, shape=(self.n,), dtype=float),  # State of Charge (0 to 1)
		})

		# Action space: n agents, each can take a binary action (0 or 1)
		self.action_space = spaces.MultiBinary(self.n)

		self.observation_space_dim = sum(np.prod(space.shape) for key, space in self.observation_space.spaces.items() if key != "TimeStep")
		# self.observation_space_dim = sum(np.prod(space.shape) for key, space in self.observation_space.spaces.items())


	def seed(self, seed_value=None):
		"""Set the random seed for reproducibility."""
		self.np_random = np.random.RandomState(seed_value)  # Create a new random state
		random.seed(seed_value)  # Seed Python's built-in random
		return [seed_value]

	def summarize_env(self):
		
		summary = {
				"Environment Info": {
					"Number of EVs in the Fleet": self.n,
					"Total Number of Chargers": self.m,
					"Total Time Steps": self.k,
					"Time Range": f"{self.config['t_0']} to {self.config['t_T']}",
					"Fee for Connecting to Charger (USD)": self.h,
					"Battery Capacity of Each EV (kWh)": self.max_cap,
					"SoC Consumed Per Step (%)": self.d_rates,
					"SoC Charged Per Step (%)": self.c_rates,
					"Low Battery Threshold (SoC)": self.low_SoC,
					f"Probability of Receiving Ride Orders within {self.delta_t} Minutes": self.assign_probs_24hrs,
					"Hours Sorted by Probability of Receiving Ride Orders": np.argsort(self.assign_probs_24hrs)[::-1].tolist(),
				},
				"Ride Info": {
					"Discretized Ride Time Probability Distribution": dict(zip(self.ride_time_probs_data["Ride Time Range (Minutes)"].values, self.ride_scenario_probs)),
					"Unit Step Ride Order Payment Rate (USD)": self.payment_rates_24hrs,
					"Hour of Maximum Payment Rate": np.argmax(self.payment_rates_24hrs),
					"Hour of Minimum Payment Rate": np.argmin(self.payment_rates_24hrs),
					"Hours Sorted by Payment per Step (Max to Min)": np.argsort(self.payment_rates_24hrs)[::-1].tolist(),
				},
				"Charging Price Info": {
					"Charging Price (USD/kWh)": self.charging_prices_24hrs,
					"Hour of Maximum Charging Price (USD)": np.argmax(self.charging_prices_24hrs),
					"Hour of Minimum Charging Price (USD)": np.argmin(self.charging_prices_24hrs),
					"Hours Sorted by Charging Price (Max to Min)": np.argsort(self.charging_prices_24hrs)[::-1].tolist(),
				}
			}


	def get_action_meanings(self):
		return {0: "Available for taking ride orders", 1: "Go to charge"}
	
	
	def get_operational_status(self):
		# Determine operational status for all agents
		operational_status = []  # List to store the status of each vehicle

		for i in range(self.n): 
			if self.obs["RideTime"][i] == 0 and self.obs["ChargingStatus"][i] == 0:
				status = "Idle"
			elif self.obs["RideTime"][i] > 0 and self.obs["ChargingStatus"][i] == 0:
				status = "Ride"
			elif self.obs["RideTime"][i] == 0 and self.obs["ChargingStatus"][i] == 1:
				status = "Charge"
			else:
				raise ValueError(f"Unexpected state for agent {i}: "
						 f"RideTime={self.obs['RideTime'][i]}, "
						 f"ChargingStatus={self.obs['ChargingStatus'][i]}")  # Raise an error
	
			operational_status.append((i, status))  # Append (agent_id, status) to the list

		return operational_status


	def reset(self):

		self.done = False
		# Reset the reward
		self.ep_return  = 0

		# Reset the observation
		state = {
			"TimeStep": np.zeros(self.n, dtype=int),
			"RideTime": np.zeros(self.n, dtype=int),
			"ChargingStatus": np.zeros(self.n, dtype=int),
			"SoC": np.zeros(self.n, dtype=float),
		}

		# Initialize battery SoC randomly
		# state["SoC"] = np.random.uniform(0, 1, size=self.n).round(3)
		state["SoC"] = self.initial_SoCs

		self.obs = state  # Store it as the environment's state

		# Empty trajectory
		self.trajectory = {'RideTime': np.zeros((self.n, self.k+1)),
							 'ChargingStatus': np.zeros((self.n, self.k+1)),
							 'SoC': np.zeros((self.n, self.k+1)),
							 'actions': np.zeros((self.n, self.k), dtype=int),
							 'rewards': []} # ride time, charging status, state of charge, action
		
		
		# return the observation
		return self.obs


	def is_zero(self, x):

		return 1 if x == 0 else 0


	def simulate_future_rides(self, start_timepoint):
		X = []  # Store simulated ride times

		for t in range(start_timepoint, self.k):
			x_t = np.zeros(self.n, dtype=int)  # Pre-allocate ride times for all vehicles at time t

			order_probs = np.array([self.rho[self.obs["TimeStep"][i]] for i in range(self.n)])  # Get order probabilities
			order_mask = np.random.random(self.n) < order_probs  # Boolean mask for accepted orders

			if np.any(order_mask):  # Process only if there are orders
				selected_indices = np.random.choice(
					self.ride_time_probs_data.index, 
					size=np.sum(order_mask), 
					p=self.ride_scenario_probs
				)

				bin_ranges = self.ride_time_probs_data.loc[selected_indices, 'Ride Time Range (Minutes)'].values
				lower_bounds, upper_bounds = np.array([
					list(map(int, r.split(' - '))) for r in bin_ranges
				]).T  # Extract lower and upper bounds

				ride_durations = np.random.uniform(lower_bounds, upper_bounds)
				x_t[order_mask] = np.ceil(ride_durations / self.delta_t).astype(int)

			X.append(x_t.tolist())  # Store results for this time step

		return X


	def simulate_ride_requests(self):
     
		x_t = np.zeros(self.n, dtype=int)  # Pre-allocate ride times for all vehicles at time t

		order_probs = np.array([self.rho[self.obs["TimeStep"][i]] for i in range(self.n)])  # Get order probabilities
		order_mask = np.random.random(self.n) < order_probs  # Boolean mask for accepted orders

		if np.any(order_mask):  # Process only if there are orders
			selected_indices = np.random.choice(
				self.ride_time_probs_data.index, 
				size=np.sum(order_mask), 
				p=self.ride_scenario_probs
			)

			bin_ranges = self.ride_time_probs_data.loc[selected_indices, 'Ride Time Range (Minutes)'].values
			lower_bounds, upper_bounds = np.array([
				list(map(int, r.split(' - '))) for r in bin_ranges
			]).T  # Extract lower and upper bounds

			ride_durations = np.random.uniform(lower_bounds, upper_bounds)
			x_t[order_mask] = np.ceil(ride_durations / self.delta_t).astype(int)

		return x_t
	
	def get_agent_state(self, agent_index):

		return (self.obs["TimeStep"][agent_index],
				self.obs["RideTime"][agent_index],
				self.obs["ChargingStatus"][agent_index],
				self.obs["SoC"][agent_index])


	def feasible_action(self, actions: list[int]) -> bool:
		# If actions in feasible return True, else return str to explain why action is infeasible.
		# if xxxxx:
			# return f"actions[{i}] if feasible because fleet is on ride"
		# if xxx:
			# return f"The number of charging fleets exceed xxx"
		for i in range(self.n):
			state_t = self.get_agent_state(i)
			action_t = actions[i]
			
			if state_t[1] >= 2:
				assert action_t == 0, (
				f"Agent {i}: Action must be 0 if ride leading time >= 2."
			)
			
			if state_t[3] >= 1.:
				assert action_t == 0, (
					f"Agent {i}: Continuing to charge would exceed battery capacity."
				)
	
		assert sum(actions) <=self.m,  (
			f"Total charging exceeds available chargers at time step."
		)
				
		return True

	def step(self, actions):

		# Assert that it is a valid action
		assert self.action_space.contains(actions), "Invalid Action"
		feasible = self.feasible_action(actions)
		if isinstance(feasible, str):
			raise BaseException(feasible)

		current_step = self.obs["TimeStep"][0]
		
		if self.stoch_step:
			ride_time_instance = self.simulate_ride_requests()
			# print("ride_time_instance:", ride_time_instance)
   
		else:
			ride_time_instance = self.ride_time_instance[:, self.obs["TimeStep"][0]]
		

		sum_rewards = 0
		for i in range(self.n):

			t, rt, ct, SoC = self.get_agent_state(i)
			action = actions[i]
			assigned_ride_time = ride_time_instance[i]

			next_SoC = np.maximum(np.minimum(SoC + ct * self.c_rates[i] + (1-ct) * (-self.d_rates[i]), 1.0), 0.)

			if action == 0:
				if SoC <= self.low_SoC:
					order_time = 0
				else:
					order_time = np.minimum(assigned_ride_time, int(round(SoC/self.d_rates[i])))
			
			if rt >= 2 and ct == 0:
				# Active ride scenario
				# (ride_time, charg_time) transitions to (ride_time-1,0)
				# Payment handled at trip start, so reward is 0
				next_state = (rt - 1, 0, next_SoC)
				reward = 0

			elif rt == 1 and ct == 0:
				if action == 0:
					# print("about to finish ride, start taking orders.")
					next_state = (order_time, 0, next_SoC)
					reward = self.w[t] * order_time

				elif action == 1:
					# print("about to finish ride, start charging.")
					next_state = (0, 1, next_SoC)
					reward = -self.h - self.p[t] * np.minimum(self.c_r[i], (1-next_SoC)*self.max_cap)
			   

			elif rt == 0 and ct > 0:
				# Charging scenario
				# (ride_time, charg_time) transitions from (0, >0) to (0, a) dependent on a

				if action == 0:
					# print("start taking orders")
					next_state = (order_time, 0, next_SoC)
					reward = self.w[t] * order_time

				elif action == 1:
					# print("continue charging.")
					next_state = (0, 1, next_SoC)
					reward = - self.p[t] * np.minimum(self.c_r[i], (1-next_SoC)*self.max_cap)

			elif rt == 0 and ct== 0: # Idle state
				
				if action == 0:
					# print("start taking orders.")
					next_state = (order_time, 0, next_SoC)
					reward = self.w[t] * order_time

				elif action == 1:
					# print("start charging.")
					next_state = (0, 1, next_SoC)
					reward = -self.h - self.p[t] * np.minimum(self.c_r[i], (1-next_SoC)*self.max_cap)

			else:
				raise ValueError("This condition should never occur.")
				
			self.obs["TimeStep"][i] = t + 1
			self.obs["RideTime"][i] = next_state[0]
			self.obs["ChargingStatus"][i] = next_state[1]
			self.obs["SoC"][i] = next_state[2]
			sum_rewards += reward

			# print(f'state, action {(rt, ct, SoC), action}')
			# print(f'next state {next_state}')
			# print(f"agent {i} has reward {reward}.")
			# print("\n")

		# Increment the episodic return: no discount factor for now
		self.ep_return += sum_rewards

		# save trajectory
		# next_step = current_step+1
		self.trajectory['actions'][:,current_step] = actions
		self.trajectory['RideTime'][:,current_step+1] = self.obs["RideTime"]
		self.trajectory['ChargingStatus'][:,current_step+1] = self.obs["ChargingStatus"]
		self.trajectory['SoC'][:,current_step+1] =self.obs["SoC"]
		self.trajectory['rewards'].append(sum_rewards)

		# If all values in the first column are equal to k, terminate the episode
		done = np.all(self.obs["TimeStep"] == self.k)

		return self.obs, sum_rewards, done, []
		# return self.obs, reward, done, []


	def render(self):

		# i = np.random.randint(0, self.n-1)
		for i in range(self.n):
			print('Show trajectory of agent %d ......' % i)
			# agents = random.sample(range(self.n), 3)

			ride_times = self.trajectory['RideTime'][i,1:]
			fractions_of_cap = self.trajectory['SoC'][i,1:] # to range [0,1]
			actions = self.trajectory['actions'][i,:]


			_, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(6, 5))
			# First plot
			ax1.step(range(self.k), ride_times, color='blue', linestyle='-', label='ride times')
			ax1.set_ylabel('Remaining Ride Time Steps', color='black')
			ax1.tick_params(axis='y', labelcolor='black')
			ax1.yaxis.set_major_locator(MaxNLocator(integer=True)) # Ensure integer ticks
			ax1.legend(loc="upper right")
			ax1.set_xlabel('Time step')

			# Second plot
			ax2.step(range(self.k), fractions_of_cap, color='black', linestyle='-.', label='state of charge')
			ax2.set_ylabel('State of Charge', color='black')
			ax2.tick_params(axis='y', labelcolor='black')
			ax2.legend(loc="upper left")

			# Create a secondary y-axis for the second plot
			ax2_secondary = ax2.twinx()
			ax2_secondary.step(range(self.k), actions, color='red', linestyle='-', label='actions')
			ax2_secondary.set_ylabel('Actions', color='red')
			ax2_secondary.tick_params(axis='y', labelcolor='red')
			ax2_secondary.yaxis.set_major_locator(MaxNLocator(integer=True)) # Ensure integer ticks
			ax2_secondary.legend(loc="upper right")
			ax2.set_xlabel('Time step')


			plt.tight_layout()
			plt.show()


class ConstrainAction(gym.ActionWrapper):
	def __init__(self, config_fname: str):
		self.env = RoadCharging(config_fname)
		super().__init__(self.env)
	# def __init__(self, env):
	# 	super().__init__(env)

	def action(self, action):
		for i in range(self.n):
			if self.obs["RideTime"][i] >= 1: # if on a ride, not charge
				action[i] = 0
			elif self.obs["SoC"][i] > 1-self.c_rates[i]: # if full capacity, not charge
				action[i] = 0
			elif self.obs["SoC"][i] <= self.low_SoC: # if low capacity has to charge
				action[i] = 1

		total_charging_requests = sum(1 for a, s in zip(action, self.obs["ChargingStatus"]) if s == 0 and a == 1)
		total_continue_charging = sum(1 for a, s in zip(action, self.obs["ChargingStatus"]) if s == 1 and a == 1)

		if total_charging_requests + total_continue_charging > self.m: # limit charging requests to available charging capacity
			print('Exceed charger capacity!')
			continue_agents = [i for i, (a, s) in enumerate(zip(action, self.obs["ChargingStatus"])) if s == 1 and a == 1]
			requesting_agents = [i for i, (a, s) in enumerate(zip(action, self.obs["ChargingStatus"])) if s == 0 and a == 1]

			available_capacity = self.m - total_continue_charging

			if available_capacity <= 0:
				print('No charger available now.')

				to_flip = requesting_agents
				for i in to_flip:
					action[i] = 0

			elif available_capacity > 0:

				if np.any(action == 1):
					to_flip = random.sample(requesting_agents, total_charging_requests-available_capacity)
					
					action[to_flip] = 0

		return action


def main():
	SoC_data_type = "high"
	n_EVs = 5
	instance_num = 1

	# data_file = "D://ORLLM//repo//road_charging//output//road_charging//data//test_data//configuration//"
	test_case = f"all_days_negativePrices_{SoC_data_type}InitSoC_1for{n_EVs}"
	test_cases_dir = os.path.join("test_cases_updated", test_case)  
	data_file = os.path.join(test_cases_dir, f"config{instance_num}_{n_EVs}EVs_1chargers.json")
 
	env = ConstrainAction(RoadCharging(data_file))

	env.summarize_env()
	env.seed(42)

	# Number of steps you run the agent for

	num_steps = env.k

	# Reset the environment to generate the first observation
	obs = env.reset()

	for step in range(num_steps):
		# # this is where you would insert your policy:
		# take random action
		action = env.action_space.sample()

		# step (transition) through the environment with the action
		# receiving the next observation, reward and if the episode has terminated or truncated
		obs, reward, done, info = env.step(action)
		# print('next ride time', obs["RideTime"])
		# print('next charging status', obs["ChargingStatus"])
  
		# if step == 80:
		# 	future_rides = env.simulate_future_rides(step)
		# 	print("future_rides:", future_rides)
   

		# If the episode has ended then we can reset to start a new episode
		if done:
			print('Final SoC:', env.obs['SoC'])
			# Render the env
			env.render()
			obs = env.reset()


	# save results
	# with open(env.save_path+'saved_trajectory.pkl', 'wb') as f:
	#     pickle.dump(env.trajectory, f)

	# Close the env
	env.close()


if __name__ == "__main__":
	main()