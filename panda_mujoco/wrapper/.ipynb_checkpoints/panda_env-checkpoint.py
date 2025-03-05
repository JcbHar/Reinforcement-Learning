import numpy as np

import gym
import gymnasium as gym
from gymnasium import spaces

import mujoco
from mujoco import viewer


class PandaEnv(gym.Env):
    def __init__(self, model_path, max_steps=500):
        super(PandaEnv, self).__init__()
        self.model = mujoco.MjModel.from_xml_path(model_path) 
        self.data = mujoco.MjData(self.model)

        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")

        self.obs_size = self.model.nq + self.model.nv + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)

        self.target_pos = self.data.xpos[self.target_id]

        self.viewer = None     

        self.max_steps = max_steps
        self.current_step = 0

    def step(self, action):        
        action = np.array(action, dtype=np.float32).reshape(-1)

        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_done()
        truncated = False

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        print(f"action:   {action.round(5)}\nreward:   {reward:.5f}\ntimestep: {self.current_step}")
        print("="*81)
        print("\n"*2)

        return obs, reward, done, truncated, {}

    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
    
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0 
        
        return self._get_obs(), {}

    
    def launch_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data) 

    
    def render(self, mode="human"):
        if self.viewer: 
            self.viewer.sync()

    
    def _get_obs(self):
        hand_pos = self.data.xpos[self.hand_id]

        return np.concatenate([
            self.data.qpos,    # joint positions
            self.data.qvel,    # joint velocities
            hand_pos           # hand position (x, y, z)
        ]).ravel()

    
    def _compute_reward(self):
        hand_pos = self.data.xpos[self.hand_id]
        target_pos = self.data.xpos[self.target_id]
    
        distance = np.linalg.norm(hand_pos - target_pos)
        
        print("\n"*3)
        print("="*81)
        print(f"dist:     {distance:.5f}")
        
        reward = 1 / (1 + distance)

        # tier rewards
        if distance < 0.4: reward += 5
        if distance < 0.3: reward += 10
        if distance < 0.2: reward += 15
        if distance < 0.1: reward += 20

        # height reward
        target_height = 0.3
        height_diff = abs(hand_pos[2] - target_height)
        reward += max(0, 5 - height_diff * 50)

        # velocity reward
        velocity = np.linalg.norm(self.data.qvel)
        if distance < 0.2:  
            reward -= velocity * 0.1
        else:
            reward += velocity * 0.3

        # torque reward
        joint_torques = np.linalg.norm(self.data.qfrc_actuator)
        if distance > 0.2:  
            reward += joint_torques * 0.02
        else:
            reward -= joint_torques * 0.05
        
        reward -= distance * 2
        
        return reward


    def _check_done(self):
        hand_pos = self.data.xpos[self.hand_id]
        distance = np.linalg.norm(hand_pos - self.target_pos)
    
        done = distance < 0.05
    
        return done


    def get_model(self):
        return self.model

    
    def get_data(self):
        return self.data

    
    def get_observation_space(self):
        return self.observation_space

    
    def get_action_space(self):
        return self.action_space

    
    def print_it(self):
        np.set_printoptions(suppress=False, precision=10, linewidth=200, threshold=np.inf)
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        
        print("fixed target position:", self.target_pos)
        print("observation shape:", obs.shape)
        print("obs_size:", self.obs_size)

