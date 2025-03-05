import os
import sys
import mujoco
import pandas as pd
import matplotlib.pyplot as plt


def plot_monitor_rewards(log_dir="logs", save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    monitor_file = os.path.join(log_dir, "monitor.csv")
    if not os.path.exists(monitor_file):
        print(f"ERROR: no monitor log file found at {monitor_file}")
        return

    df = pd.read_csv(monitor_file, skiprows=1)

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["r"], label="Episode Reward", color="blue", alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards Over Time")
    plt.legend()
    plt.grid()

    save_path = os.path.join(save_dir, "training_rewards.png")
    plt.savefig(save_path)
    plt.show()


def force_print_to_terminal():
    sys.stdout = sys.__stdout__


def print_mujoco_info(env):
    model = env.get_model()
    data = env.get_data()
    
    print("=" * 40)
    print("MuJoCo Model Properties")
    print("=" * 40)
    print(f"Number of Joints         : {model.njnt}")
    print(f"Number of Bodies         : {model.nbody}")
    print(f"Number of Actuators      : {model.nu}")
    print(f"Number of Degrees of Freedom (DOFs) : {model.nv}")
    print("=" * 40, "\n")
    
    print("=" * 49)
    print("Observation and Action Space")
    print("=" * 49)
    print(f"Observation Space: {env.get_observation_space()}")
    print(f"Action Space    : {env.get_action_space()}")
    print("=" * 49, "\n")
    
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
    ]
    print("=" * 129)
    print("Joint Information")
    print("=" * 129)
    print(f"Total Joints: {model.njnt}")
    print(f"Joint Names: {joint_names}")
    print("=" * 129, "\n")
    
    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
    ]
    print("=" * 120)
    print("Actuator Information")
    print("=" * 120)
    print(f"Total Actuators: {model.nu}")
    print(f"Actuator Names: {actuator_names}")
    print("=" * 120, "\n")
    
    for _ in range(10):
        mujoco.mj_step(model, data)

    print("=" * 97)
    print("Simulation State After 10 Steps")
    print("=" * 97)
    print(f"Updated Joint Positions (qpos): {data.qpos}")
    print(f"Updated Joint Velocities (qvel): {data.qvel}")
    print("=" * 97, "\n")
    
    print("=" * 89)
    print("Actuator Torques")
    print("=" * 89)
    print(f"Actuator Forces/Torques: {data.qfrc_actuator}")
    print("=" * 89, "\n")

