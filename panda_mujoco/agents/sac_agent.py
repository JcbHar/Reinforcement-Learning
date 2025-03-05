import os
import time

from sbx import SAC

from stable_baselines3.common.evaluation import evaluate_policy


class SACAgent:
    def __init__(self, env, policy="MlpPolicy", model_path="sac_mujoco_pickup"):
        self.env = env
        self.model_path = model_path

        if os.path.exists(f"{model_path}.zip"):
            self.load(model_path)
        else:
            self.model = SAC(policy,
                env=env,
                learning_rate=3e-4,
                buffer_size=1_000_000,  
                learning_starts=1_000,  
                batch_size=256,  
                tau=0.005,  
                gamma=0.995,  
                train_freq=(5, "step"),
                gradient_steps=4,  
                verbose=1,
            )

    
    def train(self, timesteps=500):
        start_time = time.time()

        self.model.learn(total_timesteps=timesteps, reset_num_timesteps=False)

        elapsed_time = time.time() - start_time
        minutes = elapsed_time / 60
        hours = minutes / 60

        print(f"{hours}H, {minutes}m")
        print(f"total seconds elapsed: {elapsed_time}")
        
        mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=5)
        print(f"mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
        self.model.save(self.model_path)
        

    
    def load(self, model_path):
        self.model = SAC.load(model_path, env=self.env)

    
    def predict(self, obs):
        return self.model.predict(obs, deterministic=True)

