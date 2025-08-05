import argparse
import os
import numpy as np
import pandas as pd
import torch
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union

from fo_generate.unified_mdp_env import FlexOfferEnv, DeviceType
from fo_generate.battery_model import BatteryParameters
from fo_generate.heat_model import HeatPumpParameters
from fo_generate.ev_model import EVParameters, EVUserBehavior
from fo_generate.pv_model import PVParameters

def parse_args():
    parser = argparse.ArgumentParser(description="generate flexible offer with trained model")
    
    # model parameters
    parser.add_argument("--model_path", type=str, default=None, help="path to trained model (optional)")
    parser.add_argument("--algorithm", type=str, default="fomappo", choices=["fomappo"], help="algorithm to use")
    parser.add_argument("--time_horizon", type=int, default=24, help="time range (hours)")
    parser.add_argument("--time_step", type=float, default=1.0, help="time step (hours)")
    
    # device configuration
    parser.add_argument("--device_config", type=str, default=None, help="path to device configuration file (JSON format)")
    parser.add_argument("--price_data", type=str, default=None, help="path to price data file (CSV format)")
    parser.add_argument("--weather_data", type=str, default=None, help="path to weather data file (CSV format)")
    parser.add_argument("--pv_forecast", type=str, default=None, help="path to PV forecast data file (CSV format)")
    
    # user preferences
    parser.add_argument("--economic", type=float, default=0.25, help="economic preference weight")
    parser.add_argument("--comfort", type=float, default=0.25, help="comfort preference weight")
    parser.add_argument("--self_sufficient", type=float, default=0.25, help="self-sufficient preference weight")
    parser.add_argument("--environmental", type=float, default=0.25, help="environmental preference weight")
    
    # output path
    parser.add_argument("--output_dir", type=str, default="./fo_output", help="output directory")
    parser.add_argument("--visualize", action="store_true", help="generate visualization results")
    
    return parser.parse_args()

def load_generic_agent(model_path, env, algorithm="fomappo"):
    """load trained generic model"""
    
    if algorithm == "fomappo":
        try:
            from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
            multi_env = MultiAgentFlexOfferEnv(
                data_dir="data",
                time_horizon=env.time_horizon,
                time_step=env.time_step
            )
            return multi_env
        except ImportError:
            print("FOMAPPO multi-agent environment is not available, using default strategy")
            return None
    else:
        # other algorithm loading logic
        print(f"algorithm {algorithm} is not supported")
        return None

def load_price_data(file_path):
    """load price data"""
    if file_path is None or not os.path.exists(file_path):
        # create simulated price data
        hours = np.arange(24)
        prices = 0.5 + 0.3 * np.sin((hours - 8) * np.pi / 12)
        price_data = pd.DataFrame({'hour': hours, 'price': prices})
        price_data.set_index('hour', inplace=True)
        return price_data
    
    df = pd.read_csv(file_path)
    if 'hour' in df.columns:
        df.set_index('hour', inplace=True)
    return df

def load_weather_data(file_path):
    """load weather data"""
    if file_path is None or not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    return df

def load_pv_forecast(file_path):
    """load PV forecast data"""
    if file_path is None or not os.path.exists(file_path):
        return None
    
    return pd.read_csv(file_path)

def load_device_config(config_file: str) -> Dict[str, Dict]:
    """load device configuration"""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"device configuration file {config_file} does not exist")
        
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config

def create_default_devices() -> Dict[str, Dict]:
    """create default device configuration"""
    devices = {}
    
    # default battery
    battery_params = BatteryParameters(
        battery_id="battery_1",
        soc_min=0.2,
        soc_max=0.9,
        p_min=-5.0,
        p_max=5.0,
        efficiency=0.95,
        initial_soc=0.5,
        battery_type="lithium-ion",
        capacity_kwh=10.0
    )
    
    devices["battery_1"] = {
        "type": DeviceType.BATTERY,
        "params": battery_params
    }
    
    # default heat pump
    heat_pump_params = HeatPumpParameters(
        room_id="room_1",
        room_area=30.0,
        room_volume=75.0,
        temp_min=18.0,
        temp_max=26.0,
        initial_temp=22.0,
        cop=3.5,
        heat_loss_coef=0.1,
        primary_use_period="8:00-22:00",
        secondary_use_period="22:00-8:00",
        primary_target_temp=22.0,
        secondary_target_temp=19.0,
        max_power=2.0
    )
    
    devices["heat_pump_1"] = {
        "type": DeviceType.HEAT_PUMP,
        "params": heat_pump_params
    }
    
    # default EV
    ev_params = EVParameters(
        ev_id="ev_1",
        battery_capacity=60.0,
        soc_min=0.1,
        soc_max=0.95,
        max_charging_power=11.0,
        efficiency=0.9,
        initial_soc=0.3,
        fast_charge_capable=True
    )
    
    # create user behavior
    now = datetime.now()
    arrival_time = datetime(now.year, now.month, now.day, 18, 0)
    departure_time = datetime(now.year, now.month, now.day + 1, 7, 30)
    
    ev_behavior = EVUserBehavior(
        ev_id="ev_1",
        connection_time=arrival_time,
        disconnection_time=departure_time,
        next_departure_time=departure_time,
        target_soc=0.85,
        min_required_soc=0.6,
        fast_charge_preferred=False,
        location="home",
        priority=3
    )
    
    devices["ev_1"] = {
        "type": DeviceType.EV,
        "params": ev_params,
        "behavior": ev_behavior
    }
    
    # default PV
    pv_params = PVParameters(
        pv_id="pv_1",
        max_power=5.0,
        efficiency=0.18,
        area=28.0,
        location="roof",
        tilt_angle=35.0,
        azimuth_angle=180.0,
        weather_dependent=True,
        forecast_accuracy=0.85
    )
    
    devices["pv_1"] = {
        "type": DeviceType.PV,
        "params": pv_params
    }
    
    return devices

def generate_fo_with_agent(env, agent, visualize=False, output_dir=None):
    """generate flexible offer with trained agent (generic interface)"""
    
    if agent is None:
        print("use environment's default strategy to generate FlexOffer")
        return env.generate_dfo()
    
    if hasattr(agent, 'generate_all_dfos'):
        # multi-agent environment (e.g. FOMAPPO)
        print("use multi-agent algorithm to generate FlexOffer")
        
        # execute one round to generate FlexOffer
        obs, infos = agent.reset()
        done = False
        step_count = 0
        time_horizon = getattr(agent, 'time_horizon', 24)
        
        while not done and step_count < time_horizon:
            actions = {}
            for manager_id in obs.keys():
                action_space_size = agent.action_spaces[manager_id].shape[0]
                actions[manager_id] = np.random.uniform(-1, 1, action_space_size)
            
            next_obs, rewards, dones, truncated, infos = agent.step(actions)
            obs = next_obs
            done = all(dones.values()) if isinstance(dones, dict) else dones
            step_count += 1
        
        # generate FlexOffer for all Managers
        fo_systems = agent.generate_all_dfos()
        return fo_systems
    else:
        print("use single-agent algorithm to generate FlexOffer")
        state = env.reset()
        done = False
        step = 0
        
        states = [state]
        actions = []
        rewards = []
        power_actions_history = []
        
        while not done:
            # select action (use agent's selection method)
            if hasattr(agent, 'select_action'):
                action = agent.select_action(state, add_noise=False)
            else:
                # default random action
                action = env.action_space.sample()
            
            next_state, reward, done, info = env.step(action)
            
            actions.append(action)
            rewards.append(reward)
            states.append(next_state)
            
            # record power action history
            if 'power_actions' in info:
                power_actions_history.append(info['power_actions'])
            
            state = next_state
            step += 1
        
        # get DFO from environment
        dfo_dict = env.generate_dfo()
        
        # visualize results (if needed)
        if visualize and output_dir:
            visualize_results(env, states, actions, rewards, power_actions_history, dfo_dict, output_dir)
        
        return dfo_dict

def visualize_results(env, states, actions, rewards, power_actions_history, dfo_dict, output_dir):
    """visualize results"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. plot rewards
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('rewards per step')
    plt.xlabel('step')
    plt.ylabel('reward')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'rewards.png'))
    plt.close()
    
    # 2. plot actions
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    for i in range(len(actions[0])):
        device_id = env.device_ids[i]
        plt.plot([a[i] for a in actions], label=f'{device_id}')
    plt.title('agent actions')
    plt.xlabel('step')
    plt.ylabel('normalized action')
    plt.legend()
    plt.grid(True)
    
    # 3. plot power of each device
    plt.subplot(212)
    for device_id in env.device_ids:
        powers = [pa[device_id] for pa in power_actions_history]
        plt.plot(powers, label=f'{device_id} power')
    plt.title('device power')
    plt.xlabel('step')
    plt.ylabel('power (kW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actions_powers.png'))
    plt.close()
    
    # 4. 绘制每个设备的FO
    for device_id, dfo in dfo_dict.items():
        plt.figure(figsize=(10, 6))
        time_steps = range(dfo.time_horizon)
        e_mins = [dfo.slices[t].energy_min for t in time_steps]
        e_maxs = [dfo.slices[t].energy_max for t in time_steps]
        
        plt.fill_between(time_steps, e_mins, e_maxs, alpha=0.3, label='energy range')
        plt.plot(time_steps, e_mins, 'b-', label='minimum energy')
        plt.plot(time_steps, e_maxs, 'r-', label='maximum energy')
        
        # if PV device, plot both forecast and actual power
        if env.device_types[device_id] == DeviceType.PV:
            device_mdp = env.device_mdps[device_id]
            device = device_mdp.model
            if hasattr(device, 'power_history') and len(device.power_history) > 0:
                plt.plot(range(len(device.power_history)), device.power_history, 'g--', label='actual power')
            if hasattr(device, 'forecast_data') and device.forecast_data is not None:
                plt.plot(range(len(device.forecast_data)), device.forecast_data, 'y--', label='forecast power')
        
        plt.title(f'{device_id} flexible offer')
        plt.xlabel('time step')
        plt.ylabel('energy (kWh)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{device_id}_fo.png'))
        plt.close()
    
    # 5. plot device SOC change (battery, PV storage and EV)
    plt.figure(figsize=(10, 6))
    
    for device_id, device_mdp in env.device_mdps.items():
        device = device_mdp.model
        device_type = env.device_types[device_id]
        
        if device_type in [DeviceType.BATTERY, DeviceType.PV, DeviceType.EV] and hasattr(device, 'current_soc'):
            # extract SOC for each time step
            soc_history = []
            for s in states:
                # parse state to get SOC
                device_idx = env.device_ids.index(device_id)
                offset = 30  # basic state dimension (time, price, preference)
                
                # calculate index of current device state in state vector
                for i in range(device_idx):
                    prev_device_type = env.device_types[env.device_ids[i]]
                    if prev_device_type == DeviceType.BATTERY:
                        offset += 4
                    elif prev_device_type == DeviceType.HEAT_PUMP:
                        offset += 3
                    elif prev_device_type == DeviceType.EV:
                        offset += 4
                    elif prev_device_type == DeviceType.PV:
                        offset += 5
                
                # SOC is usually the first value of device state
                soc_history.append(s[offset])
            
            plt.plot(soc_history, label=f'{device_id} SOC')
    
    plt.title('SOC change of energy storage devices')
    plt.xlabel('time step')
    plt.ylabel('SOC')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'soc_history.png'))
    plt.close()

def save_fo(dfo_dict, output_dir):
    """save generated FO to file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for device_id, dfo in dfo_dict.items():
        with open(os.path.join(output_dir, f"{device_id}_fo.json"), "w") as f:
            json.dump(dfo.to_dict(), f, indent=2)
            
    print(f"flexible offer saved to {output_dir}")

def main():
    args = parse_args()
    
    # create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # load data
    price_data = load_price_data(args.price_data)
    weather_data = load_weather_data(args.weather_data)
    pv_forecast = load_pv_forecast(args.pv_forecast)
    devices = load_device_config(args.device_config)
    
    # user preferences
    user_preferences = {
        "economic": args.economic,
        "comfort": args.comfort,
        "self_sufficient": args.self_sufficient,
        "environmental": args.environmental
    }
    
    # create environment
    env = FlexOfferEnv(
        devices=devices,
        time_horizon=args.time_horizon,
        time_step=args.time_step,
        start_time=datetime.now(),
        price_data=price_data,
        user_preferences=user_preferences,
        weather_data=weather_data if weather_data is not None else pd.DataFrame()
    )
    
    # set PV device forecast data in environment
    if pv_forecast is not None:
        for device_id, device_mdp in env.device_mdps.items():
            if env.device_types[device_id] == DeviceType.PV and device_id in pv_forecast.columns:
                device_mdp.model.set_forecast_data(pv_forecast[device_id].tolist())
    
    # load trained model
    agent = load_generic_agent(args.model_path, env, args.algorithm)
    
    # use agent to generate FO
    print("generate flexible offer...")
    dfo_dict = generate_fo_with_agent(env, agent, visualize=args.visualize, output_dir=args.output_dir)
    
    # save FO
    save_fo(dfo_dict, args.output_dir)
    
    # save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        config = {
            "time_horizon": args.time_horizon,
            "time_step": args.time_step,
            "user_preferences": user_preferences,
            "model_path": args.model_path,
            "device_types": {k: env.device_types[k] for k in env.device_ids}
        }
        json.dump(config, f, indent=2)
    
    print(f"all outputs saved to {args.output_dir}")
    print(f"total reward: {sum(dfo_dict.values()):.2f}")

if __name__ == "__main__":
    main() 