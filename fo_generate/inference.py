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
    parser = argparse.ArgumentParser(description="使用训练好的模型生成灵活性报价")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, default=None, help="训练好的模型路径（可选）")
    parser.add_argument("--algorithm", type=str, default="fomappo", choices=["fomappo"], help="使用的算法")
    parser.add_argument("--time_horizon", type=int, default=24, help="时间范围（小时）")
    parser.add_argument("--time_step", type=float, default=1.0, help="时间步长（小时）")
    
    # 设备配置
    parser.add_argument("--device_config", type=str, default=None, help="设备配置文件路径 (JSON格式)")
    parser.add_argument("--price_data", type=str, default=None, help="电价数据文件路径 (CSV格式)")
    parser.add_argument("--weather_data", type=str, default=None, help="天气数据文件路径 (CSV格式)")
    parser.add_argument("--pv_forecast", type=str, default=None, help="光伏预测数据文件路径 (CSV格式)")
    
    # 用户偏好
    parser.add_argument("--economic", type=float, default=0.25, help="经济性偏好权重")
    parser.add_argument("--comfort", type=float, default=0.25, help="舒适性偏好权重")
    parser.add_argument("--self_sufficient", type=float, default=0.25, help="自给自足偏好权重")
    parser.add_argument("--environmental", type=float, default=0.25, help="环保性偏好权重")
    
    # 输出路径
    parser.add_argument("--output_dir", type=str, default="./fo_output", help="输出目录")
    parser.add_argument("--visualize", action="store_true", help="是否生成可视化结果")
    
    return parser.parse_args()

def load_generic_agent(model_path, env, algorithm="fomappo"):
    """加载训练好的通用模型"""
    
    if algorithm == "fomappo":
        # FOMAPPO是多智能体算法，直接使用多智能体环境
        try:
            from fo_generate.multi_agent_env import MultiAgentFlexOfferEnv
            multi_env = MultiAgentFlexOfferEnv(
                data_dir="data",
                time_horizon=env.time_horizon,
                time_step=env.time_step
            )
            return multi_env
        except ImportError:
            print("FOMAPPO多智能体环境不可用，使用默认策略")
            return None
    else:
        # 其他算法的加载逻辑
        print(f"算法 {algorithm} 暂不支持")
        return None

def load_price_data(file_path):
    """加载电价数据"""
    if file_path is None or not os.path.exists(file_path):
        # 创建模拟电价数据
        hours = np.arange(24)
        # 简单的日内价格曲线：早晚高峰，午夜低谷
        prices = 0.5 + 0.3 * np.sin((hours - 8) * np.pi / 12)
        price_data = pd.DataFrame({'hour': hours, 'price': prices})
        price_data.set_index('hour', inplace=True)
        return price_data
    
    df = pd.read_csv(file_path)
    if 'hour' in df.columns:
        df.set_index('hour', inplace=True)
    return df

def load_weather_data(file_path):
    """加载天气数据"""
    if file_path is None or not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    return df

def load_pv_forecast(file_path):
    """加载光伏预测数据"""
    if file_path is None or not os.path.exists(file_path):
        return None
    
    return pd.read_csv(file_path)

def load_device_config(config_file: str) -> Dict[str, Dict]:
    """加载设备配置"""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"设备配置文件 {config_file} 不存在")
        
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config

def create_default_devices() -> Dict[str, Dict]:
    """创建默认设备配置"""
    devices = {}
    
    # 默认电池
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
    
    # 默认热泵
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
    
    # 默认电动汽车
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
    
    # 创建用户行为
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
    
    # 默认光伏
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
    """使用训练好的代理生成灵活性报价（通用接口）"""
    
    if agent is None:
        # 如果没有代理，使用环境的默认策略
        print("使用环境默认策略生成FlexOffer")
        return env.generate_dfo()
    
    # 检查是否是多智能体环境
    if hasattr(agent, 'generate_all_dfos'):
        # 多智能体环境（如FOMAPPO）
        print("使用多智能体算法生成FlexOffer")
        
        # 执行一个回合来生成FlexOffer
        obs, infos = agent.reset()
        done = False
        step_count = 0
        time_horizon = getattr(agent, 'time_horizon', 24)
        
        while not done and step_count < time_horizon:
            actions = {}
            for manager_id in obs.keys():
                action_space_size = agent.action_spaces[manager_id].shape[0]
                # 使用随机策略（可以替换为训练好的策略）
                actions[manager_id] = np.random.uniform(-1, 1, action_space_size)
            
            next_obs, rewards, dones, truncated, infos = agent.step(actions)
            obs = next_obs
            done = all(dones.values()) if isinstance(dones, dict) else dones
            step_count += 1
        
        # 生成所有Manager的FlexOffer
        fo_systems = agent.generate_all_dfos()
        return fo_systems
    else:
        # 单智能体算法
        print("使用单智能体算法生成FlexOffer")
        state = env.reset()
        done = False
        step = 0
        
        states = [state]
        actions = []
        rewards = []
        power_actions_history = []
        
        while not done:
            # 选择动作（使用代理的选择方法）
            if hasattr(agent, 'select_action'):
                action = agent.select_action(state, add_noise=False)
            else:
                # 默认随机动作
                action = env.action_space.sample()
            
            next_state, reward, done, info = env.step(action)
            
            actions.append(action)
            rewards.append(reward)
            states.append(next_state)
            
            # 记录功率动作历史
            if 'power_actions' in info:
                power_actions_history.append(info['power_actions'])
            
            state = next_state
            step += 1
        
        # 从环境中获取DFO
        dfo_dict = env.generate_dfo()
        
        # 可视化结果（如果需要）
        if visualize and output_dir:
            visualize_results(env, states, actions, rewards, power_actions_history, dfo_dict, output_dir)
        
        return dfo_dict

def visualize_results(env, states, actions, rewards, power_actions_history, dfo_dict, output_dir):
    """可视化结果"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 绘制奖励
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('每步奖励')
    plt.xlabel('步骤')
    plt.ylabel('奖励')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'rewards.png'))
    plt.close()
    
    # 2. 绘制动作
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    for i in range(len(actions[0])):
        device_id = env.device_ids[i]
        plt.plot([a[i] for a in actions], label=f'{device_id}')
    plt.title('代理动作')
    plt.xlabel('步骤')
    plt.ylabel('归一化动作')
    plt.legend()
    plt.grid(True)
    
    # 3. 绘制每个设备的功率
    plt.subplot(212)
    for device_id in env.device_ids:
        powers = [pa[device_id] for pa in power_actions_history]
        plt.plot(powers, label=f'{device_id} 功率')
    plt.title('设备功率')
    plt.xlabel('步骤')
    plt.ylabel('功率 (kW)')
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
        
        plt.fill_between(time_steps, e_mins, e_maxs, alpha=0.3, label='能量范围')
        plt.plot(time_steps, e_mins, 'b-', label='最小能量')
        plt.plot(time_steps, e_maxs, 'r-', label='最大能量')
        
        # 如果是PV设备，同时绘制预测和实际发电量
        if env.device_types[device_id] == DeviceType.PV:
            device_mdp = env.device_mdps[device_id]
            device = device_mdp.model
            if hasattr(device, 'power_history') and len(device.power_history) > 0:
                plt.plot(range(len(device.power_history)), device.power_history, 'g--', label='实际发电量')
            if hasattr(device, 'forecast_data') and device.forecast_data is not None:
                plt.plot(range(len(device.forecast_data)), device.forecast_data, 'y--', label='预测发电量')
        
        plt.title(f'{device_id} 灵活性报价')
        plt.xlabel('时间步')
        plt.ylabel('能量 (kWh)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{device_id}_fo.png'))
        plt.close()
    
    # 5. 绘制设备SOC变化 (电池、PV储能和EV)
    plt.figure(figsize=(10, 6))
    
    for device_id, device_mdp in env.device_mdps.items():
        device = device_mdp.model
        device_type = env.device_types[device_id]
        
        if device_type in [DeviceType.BATTERY, DeviceType.PV, DeviceType.EV] and hasattr(device, 'current_soc'):
            # 提取每个时间步的SOC
            soc_history = []
            for s in states:
                # 解析状态获取SOC
                # 这需要根据状态向量的实际布局进行调整
                device_idx = env.device_ids.index(device_id)
                offset = 30  # 基本状态维度 (时间、价格、偏好)
                
                # 计算当前设备状态在状态向量中的索引
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
                
                # SOC通常是设备状态的第一个值
                soc_history.append(s[offset])
            
            plt.plot(soc_history, label=f'{device_id} SOC')
    
    plt.title('储能设备SOC变化')
    plt.xlabel('时间步')
    plt.ylabel('SOC')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'soc_history.png'))
    plt.close()

def save_fo(dfo_dict, output_dir):
    """保存生成的FO到文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for device_id, dfo in dfo_dict.items():
        with open(os.path.join(output_dir, f"{device_id}_fo.json"), "w") as f:
            json.dump(dfo.to_dict(), f, indent=2)
            
    print(f"灵活性报价已保存到 {output_dir}")

def main():
    args = parse_args()
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 加载数据
    price_data = load_price_data(args.price_data)
    weather_data = load_weather_data(args.weather_data)
    pv_forecast = load_pv_forecast(args.pv_forecast)
    devices = load_device_config(args.device_config)
    
    # 用户偏好
    user_preferences = {
        "economic": args.economic,
        "comfort": args.comfort,
        "self_sufficient": args.self_sufficient,
        "environmental": args.environmental
    }
    
    # 创建环境
    env = FlexOfferEnv(
        devices=devices,
        time_horizon=args.time_horizon,
        time_step=args.time_step,
        start_time=datetime.now(),
        price_data=price_data,
        user_preferences=user_preferences,
        weather_data=weather_data if weather_data is not None else pd.DataFrame()
    )
    
    # 设置环境中PV设备的预测数据
    if pv_forecast is not None:
        for device_id, device_mdp in env.device_mdps.items():
            if env.device_types[device_id] == DeviceType.PV and device_id in pv_forecast.columns:
                device_mdp.model.set_forecast_data(pv_forecast[device_id].tolist())
    
    # 加载训练好的模型
    agent = load_generic_agent(args.model_path, env, args.algorithm)
    
    # 使用代理生成FO
    print("生成灵活性报价...")
    dfo_dict = generate_fo_with_agent(env, agent, visualize=args.visualize, output_dir=args.output_dir)
    
    # 保存FO
    save_fo(dfo_dict, args.output_dir)
    
    # 保存配置
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        config = {
            "time_horizon": args.time_horizon,
            "time_step": args.time_step,
            "user_preferences": user_preferences,
            "model_path": args.model_path,
            "device_types": {k: env.device_types[k] for k in env.device_ids}
        }
        json.dump(config, f, indent=2)
    
    print(f"所有输出已保存到 {args.output_dir}")
    print(f"总奖励: {sum(dfo_dict.values()):.2f}")

if __name__ == "__main__":
    main() 