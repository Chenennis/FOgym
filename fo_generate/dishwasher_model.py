from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from fo_generate.dfo import DFOSystem, DFOSlice

@dataclass
class DishwasherParameters:
    """洗碗机参数"""
    dishwasher_id: str           # 洗碗机ID
    total_energy: float          # 总需要能量 (kWh) 固定值
    power_rating: float          # 额定功率 (kW)
    operation_hours: float       # 运行时长 (小时) 通常3-4小时
    min_start_delay: float       # 最小启动延迟 (小时) 避免立即启动
    max_start_delay: float       # 最大启动延迟 (小时) 避免等待太久
    efficiency: float            # 能效比
    can_interrupt: bool          # 是否可中断（通常为False）
    behavior: Optional['DishwasherUserBehavior'] = None  # 用户行为
    
@dataclass
class DishwasherUserBehavior:
    """洗碗机用户行为模型"""
    dishwasher_id: str           # 洗碗机ID
    deployment_time: datetime    # 部署完毕时刻（用户按start时间）
    preferred_start_time: Optional[datetime] = None  # 优选启动时间
    latest_completion_time: Optional[datetime] = None  # 最晚完成时间
    priority: int = 3            # 优先级 (1-5，5最高)
    user_tolerance: float = 2.0  # 用户容忍延迟时间(小时)

class DishwasherModel:
    """洗碗机模型类"""
    def __init__(self, params: DishwasherParameters, user_behavior: Optional[DishwasherUserBehavior] = None):
        self.params = params
        self.user_behavior = user_behavior
        
        # 洗碗机状态
        self.is_deployed = False      # 是否已部署（用户按了start）
        self.is_running = False       # 是否正在运行
        self.is_completed = False     # 是否已完成
        self.current_cycle_step = 0   # 当前运行步骤
        self.total_cycle_steps = int(params.operation_hours)  # 总运行步骤数
        self.deployment_time = None   # 实际部署时间
        self.start_time = None        # 实际启动时间
        self.completion_time = None   # 实际完成时间
        self.energy_consumed = 0.0    # 已消耗能量
        
    def deploy(self, current_time: datetime):
        """部署洗碗机（用户按下start按钮）"""
        if not self.is_deployed:
            self.is_deployed = True
            self.deployment_time = current_time
            if self.user_behavior:
                self.user_behavior.deployment_time = current_time
            
    def can_start(self, current_time: datetime) -> bool:
        """检查是否可以启动"""
        if not self.is_deployed or self.is_running or self.is_completed:
            return False
            
        # 检查最小延迟
        if self.deployment_time:
            time_since_deployment = (current_time - self.deployment_time).total_seconds() / 3600
            if time_since_deployment < self.params.min_start_delay:
                return False
        
        # 检查最大延迟
        if self.user_behavior and self.deployment_time:
            time_since_deployment = (current_time - self.deployment_time).total_seconds() / 3600
            if time_since_deployment > self.params.max_start_delay:
                return True  # 必须启动了，不能再等
                
        return True
    
    def must_start(self, current_time: datetime) -> bool:
        """检查是否必须启动（不能再延迟）"""
        if not self.is_deployed or self.is_running or self.is_completed:
            return False
            
        if self.user_behavior and self.deployment_time:
            time_since_deployment = (current_time - self.deployment_time).total_seconds() / 3600
            
            # 如果超过最大延迟时间，必须启动
            if time_since_deployment >= self.params.max_start_delay:
                return True
                
            # 如果有最晚完成时间约束
            if self.user_behavior.latest_completion_time:
                time_to_deadline = (self.user_behavior.latest_completion_time - current_time).total_seconds() / 3600
                if time_to_deadline <= self.params.operation_hours:
                    return True
                    
        return False
    
    def start_operation(self, current_time: datetime) -> bool:
        """启动洗碗机运行"""
        if self.can_start(current_time) and not self.is_running:
            self.is_running = True
            self.start_time = current_time
            self.current_cycle_step = 0
            return True
        return False
    
    def step_operation(self, current_time: datetime, available_power: float) -> Tuple[float, bool]:
        """运行一个时间步
        
        Returns:
            required_power: 需要的功率
            is_completed: 是否完成
        """
        if not self.is_running or self.is_completed:
            return 0.0, self.is_completed
            
        # 洗碗机需要固定功率运行
        required_power = self.params.power_rating
        
        # 检查是否有足够功率
        if available_power >= required_power:
            # 消耗能量
            energy_step = required_power * 1.0  # 假设1小时时间步
            self.energy_consumed += energy_step
            self.current_cycle_step += 1
            
            # 检查是否完成
            if self.current_cycle_step >= self.total_cycle_steps:
                self.is_completed = True
                self.is_running = False
                self.completion_time = current_time
                return required_power, True
            
            return required_power, False
        else:
            # 功率不足，洗碗机无法运行（这种情况应该避免）
            # 在实际FlexOffer生成时，应该确保启动时有足够的连续功率
            return required_power, False
    
    def get_required_power_profile(self, start_time: datetime) -> List[float]:
        """获取从给定启动时间开始的功率需求曲线"""
        power_profile = []
        for i in range(self.total_cycle_steps):
            power_profile.append(self.params.power_rating)
        return power_profile
    
    def get_flexibility_window(self, current_time: datetime) -> Tuple[datetime, datetime]:
        """获取灵活性时间窗口"""
        if not self.is_deployed:
            return current_time, current_time
            
        earliest_start = self.deployment_time + timedelta(hours=self.params.min_start_delay)
        latest_start = self.deployment_time + timedelta(hours=self.params.max_start_delay)
        
        # 考虑最晚完成时间约束
        if self.user_behavior and self.user_behavior.latest_completion_time:
            latest_by_completion = self.user_behavior.latest_completion_time - timedelta(hours=self.params.operation_hours)
            latest_start = min(latest_start, latest_by_completion)
        
        return max(earliest_start, current_time), latest_start
    
    def calculate_urgency(self, current_time: datetime) -> float:
        """计算紧急度（0-1，1最紧急）"""
        if not self.is_deployed or self.is_completed:
            return 0.0
            
        if self.is_running:
            return 1.0  # 正在运行，最紧急
            
        earliest_start, latest_start = self.get_flexibility_window(current_time)
        
        if current_time >= latest_start:
            return 1.0  # 必须立即启动
            
        total_window = (latest_start - earliest_start).total_seconds() / 3600
        elapsed_time = (current_time - earliest_start).total_seconds() / 3600
        
        if total_window <= 0:
            return 1.0
            
        urgency = max(0.0, elapsed_time / total_window)
        return min(1.0, urgency)
    
    def generate_dfo(self, 
                     start_time=None, 
                     time_horizon: int = None, 
                     time_step: float = 1.0) -> DFOSystem:
        """生成DFO系统
        
        Args:
            start_time: 可选，起始时间，如果为None，使用当前时间。
                        如果是整数且time_horizon为None，则作为time_horizon使用
            time_horizon: 时间范围，如果为None且第一个参数为整数，则使用第一个参数
            time_step: 时间步长，默认为1小时
            
        Returns:
            DFO系统对象
        """
        # 兼容旧的调用方式
        if isinstance(start_time, int) and time_horizon is None:
            time_horizon = start_time
            start_time = None
            
        # 如果start_time为None，使用当前时间
        current_time = start_time if start_time is not None and not isinstance(start_time, int) else datetime.now()
        
        # 确保time_horizon有值
        if time_horizon is None:
            time_horizon = 24  # 默认值为24小时
        
        dfo = DFOSystem(time_horizon)
        
        # 如果还未部署，所有时间步的功率需求都是0
        if not self.is_deployed:
            for t in range(time_horizon):
                slice = DFOSlice(
                    time_step=t,
                    energy_min=0.0,
                    energy_max=0.0,
                    constraints=[]
                )
                dfo.add_slice(slice)
            return dfo
        
        # 如果已完成，所有时间步的功率需求都是0
        if self.is_completed:
            for t in range(time_horizon):
                slice = DFOSlice(
                    time_step=t,
                    energy_min=0.0,
                    energy_max=0.0,
                    constraints=[]
                )
                dfo.add_slice(slice)
            return dfo
        
        # 如果正在运行，必须持续提供功率
        if self.is_running:
            remaining_steps = self.total_cycle_steps - self.current_cycle_step
            for t in range(time_horizon):
                if t < remaining_steps:
                    # 必须运行
                    energy_required = self.params.power_rating
                    slice = DFOSlice(
                        time_step=t,
                        energy_min=energy_required,
                        energy_max=energy_required,
                        constraints=[]
                    )
                else:
                    # 运行完成
                    slice = DFOSlice(
                        time_step=t,
                        energy_min=0.0,
                        energy_max=0.0,
                        constraints=[]
                    )
                dfo.add_slice(slice)
            return dfo
        
        # 已部署但未运行：生成灵活性报价
        earliest_start, latest_start = self.get_flexibility_window(current_time)
        
        # 计算时间步对应的启动窗口
        earliest_step = max(0, int((earliest_start - current_time).total_seconds() / 3600 / time_step))
        latest_step = min(time_horizon - self.total_cycle_steps, 
                         int((latest_start - current_time).total_seconds() / 3600 / time_step))
        
        for t in range(time_horizon):
            # 检查这个时间步是否可能是启动时间
            can_start_at_t = earliest_step <= t <= latest_step
            
            if can_start_at_t:
                # 这个时间步可能启动，需要检查是否有足够的连续时间完成运行
                remaining_time_steps = time_horizon - t
                if remaining_time_steps >= self.total_cycle_steps:
                    # 有足够时间完成运行
                    energy_min = 0.0  # 可以选择不在这个时间步启动
                    energy_max = self.params.power_rating  # 如果启动，需要此功率
                else:
                    # 没有足够时间完成运行，不能在这个时间步启动
                    energy_min = 0.0
                    energy_max = 0.0
            else:
                # 不能在这个时间步启动
                energy_min = 0.0
                energy_max = 0.0
            
            # 如果当前时间已经到了必须启动的时刻
            current_step_time = current_time + timedelta(hours=t * time_step)
            if self.must_start(current_step_time) and t == 0:
                # 必须立即启动
                energy_min = self.params.power_rating
                energy_max = self.params.power_rating
            
            # 创建约束
            constraints = []
            
            # 添加运行连续性约束（如果启动，必须连续运行）
            # 这个约束比较复杂，在DFO聚合阶段处理
            
            slice = DFOSlice(
                time_step=t,
                energy_min=energy_min,
                energy_max=energy_max,
                constraints=constraints
            )
            dfo.add_slice(slice)
            
        return dfo

    @classmethod
    def from_csv(cls, params_file: str, behavior_file: str = None, dishwasher_id: str = None) -> 'DishwasherModel':
        """从CSV文件创建洗碗机模型"""
        # 读取参数文件
        params_df = pd.read_csv(params_file, comment='#')
        
        # 如果指定了dishwasher_id，查找对应数据；否则使用第一行
        if dishwasher_id:
            device_data = params_df[params_df['dishwasher_id'] == dishwasher_id]
            if device_data.empty:
                raise ValueError(f"Dishwasher ID {dishwasher_id} not found in {params_file}")
            device_data = device_data.iloc[0]
        else:
            device_data = params_df.iloc[0]
            dishwasher_id = device_data['dishwasher_id']
        
        # 创建参数对象
        params = DishwasherParameters(
            dishwasher_id=dishwasher_id,
            total_energy=float(device_data['total_energy']),
            power_rating=float(device_data['power_rating']),
            operation_hours=float(device_data['operation_hours']),
            min_start_delay=float(device_data['min_start_delay']),
            max_start_delay=float(device_data['max_start_delay']),
            efficiency=float(device_data['efficiency']),
            can_interrupt=device_data['can_interrupt'] == 'True'
        )
        
        # 如果提供了行为文件，读取用户行为
        user_behavior = None
        if behavior_file:
            behavior_df = pd.read_csv(behavior_file, comment='#')
            behavior_data = behavior_df[behavior_df['dishwasher_id'] == dishwasher_id]
            
            if not behavior_data.empty:
                behavior_data = behavior_data.iloc[0]
                user_behavior = DishwasherUserBehavior(
                    dishwasher_id=dishwasher_id,
                    deployment_time=pd.to_datetime(behavior_data['deployment_time']),
                    preferred_start_time=pd.to_datetime(behavior_data['preferred_start_time']) if pd.notna(behavior_data['preferred_start_time']) else None,
                    latest_completion_time=pd.to_datetime(behavior_data['latest_completion_time']) if pd.notna(behavior_data['latest_completion_time']) else None,
                    priority=int(behavior_data['priority']),
                    user_tolerance=float(behavior_data['user_tolerance'])
                )
        
        return cls(params, user_behavior)

    @classmethod
    def get_all_dishwasher_ids(cls, params_file: str) -> List[str]:
        """获取参数文件中的所有洗碗机ID"""
        try:
            params_df = pd.read_csv(params_file, comment='#')
            return params_df['dishwasher_id'].tolist()
        except Exception as e:
            print(f"读取洗碗机参数文件失败: {e}")
            return []

    def get_status_summary(self) -> Dict:
        """获取状态摘要"""
        return {
            'dishwasher_id': self.params.dishwasher_id,
            'is_deployed': self.is_deployed,
            'is_running': self.is_running,
            'is_completed': self.is_completed,
            'current_cycle_step': self.current_cycle_step,
            'total_cycle_steps': self.total_cycle_steps,
            'energy_consumed': self.energy_consumed,
            'total_energy_required': self.params.total_energy,
            'deployment_time': self.deployment_time.isoformat() if self.deployment_time else None,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'completion_time': self.completion_time.isoformat() if self.completion_time else None
        } 