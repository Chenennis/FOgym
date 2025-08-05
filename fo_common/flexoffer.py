from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import numpy as np

@dataclass
class FOSlice:
    """FlexOffer时间片 - 代表特定时间段内的能量需求/提供"""
    slice_id: int                    # 时间片ID (在小时内的序号)
    start_time: datetime            # 开始时间
    end_time: datetime              # 结束时间  
    energy_min: float               # 最小能量需求/提供 (kWh)
    energy_max: float               # 最大能量需求/提供 (kWh)
    duration_minutes: float         # 时间片长度(分钟)
    device_type: str = "unknown"    # 设备类型
    device_id: str = ""             # 设备ID
    priority: int = 3               # 优先级 (1-5, 1最高)
    flexibility_factor: float = 0.5 # 灵活性因子 [0, 1]
    
    def get_duration_hours(self) -> float:
        """获取时间片长度（小时）"""
        return self.duration_minutes / 60.0
    
    def get_energy_range(self) -> float:
        """获取能量范围"""
        return self.energy_max - self.energy_min
    
    def get_average_energy(self) -> float:
        """获取平均能量"""
        return (self.energy_min + self.energy_max) / 2.0

@dataclass  
class FlexOffer:
    """标准FlexOffer (FO) - 代表一小时内的能量需求/提供配置文件"""
    fo_id: str                      # FlexOffer ID
    hour: int                       # 小时 (0-23)
    start_time: datetime            # 开始时间
    end_time: datetime              # 结束时间
    device_id: str                  # 设备ID
    device_type: str                # 设备类型
    slices: List[FOSlice]           # 时间片列表
    total_energy_min: float = 0.0   # 总最小能量
    total_energy_max: float = 0.0   # 总最大能量
    profile_length: int = 0         # 轮廓长度(非零slice数量)
    time_flexibility: float = 0.0   # 时间灵活性
    
    def __post_init__(self):
        """初始化后处理"""
        self._calculate_properties()
    
    def _calculate_properties(self):
        """计算FO的基本属性"""
        if self.slices:
            self.total_energy_min = sum(s.energy_min for s in self.slices)
            self.total_energy_max = sum(s.energy_max for s in self.slices)
            
            # 计算轮廓长度（非零能量的slice数量）
            self.profile_length = sum(1 for s in self.slices 
                                    if s.energy_min != 0 or s.energy_max != 0)
            
            # 计算时间灵活性（平均能量范围）
            if self.profile_length > 0:
                self.time_flexibility = sum(s.get_energy_range() for s in self.slices) / self.profile_length
            else:
                self.time_flexibility = 0.0
    
    def add_slice(self, slice: FOSlice):
        """添加时间片"""
        self.slices.append(slice)
        self._calculate_properties()
    
    def get_slice(self, slice_id: int) -> Optional[FOSlice]:
        """获取指定ID的时间片"""
        for slice in self.slices:
            if slice.slice_id == slice_id:
                return slice
        return None
    
    def get_energy_bounds(self, slice_id: int) -> Tuple[float, float]:
        """获取指定时间片的能量边界"""
        slice = self.get_slice(slice_id)
        if slice:
            return slice.energy_min, slice.energy_max
        return 0.0, 0.0
    
    def get_energy_profile(self) -> Tuple[List[float], List[float]]:
        """获取能量轮廓"""
        e_min = [s.energy_min for s in self.slices]
        e_max = [s.energy_max for s in self.slices]
        return e_min, e_max
    
    def get_power_profile(self) -> Tuple[List[float], List[float]]:
        """获取功率轮廓 (kW)"""
        p_min = []
        p_max = []
        for s in self.slices:
            duration_hours = s.get_duration_hours()
            if duration_hours > 0:
                p_min.append(s.energy_min / duration_hours)
                p_max.append(s.energy_max / duration_hours)
            else:
                # 处理持续时间为0的情况
                p_min.append(0.0)
                p_max.append(0.0)
        return p_min, p_max
    
    def profile_size(self) -> int:
        """获取轮廓尺寸"""
        return self.profile_length
    
    def tf(self) -> float:
        """获取时间灵活性"""
        return self.time_flexibility
    
    def is_compatible_with(self, other: 'FlexOffer', tf_threshold: float = 1.0) -> bool:
        """检查与另一个FO的兼容性"""
        if not isinstance(other, FlexOffer):
            return False
        
        # 检查时间范围是否一致
        if len(self.slices) != len(other.slices):
            return False
        
        # 检查时间灵活性是否在阈值内
        tf_diff = abs(self.time_flexibility - other.time_flexibility)
        return tf_diff <= tf_threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'fo_id': self.fo_id,
            'hour': self.hour,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'device_id': self.device_id,
            'device_type': self.device_type,
            'total_energy_min': self.total_energy_min,
            'total_energy_max': self.total_energy_max,
            'profile_length': self.profile_length,
            'time_flexibility': self.time_flexibility,
            'slices': [
                {
                    'slice_id': s.slice_id,
                    'start_time': s.start_time.isoformat(),
                    'end_time': s.end_time.isoformat(),
                    'energy_min': s.energy_min,
                    'energy_max': s.energy_max,
                    'duration_minutes': s.duration_minutes,
                    'device_type': s.device_type,
                    'device_id': s.device_id,
                    'priority': s.priority,
                    'flexibility_factor': s.flexibility_factor
                }
                for s in self.slices
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FlexOffer':
        """从字典创建FlexOffer"""
        # 恢复时间片
        slices = []
        for slice_data in data['slices']:
            slice = FOSlice(
                slice_id=slice_data['slice_id'],
                start_time=datetime.fromisoformat(slice_data['start_time']),
                end_time=datetime.fromisoformat(slice_data['end_time']),
                energy_min=slice_data['energy_min'],
                energy_max=slice_data['energy_max'],
                duration_minutes=slice_data['duration_minutes'],
                device_type=slice_data.get('device_type', 'unknown'),
                device_id=slice_data.get('device_id', ''),
                priority=slice_data.get('priority', 3),
                flexibility_factor=slice_data.get('flexibility_factor', 0.5)
            )
            slices.append(slice)
        
        return cls(
            fo_id=data['fo_id'],
            hour=data['hour'],
            start_time=datetime.fromisoformat(data['start_time']),
            end_time=datetime.fromisoformat(data['end_time']),
            device_id=data['device_id'],
            device_type=data['device_type'],
            slices=slices
        )

class FOFactory:
    """FlexOffer工厂类 - 用于创建标准化的FlexOffer"""
    
    @staticmethod
    def create_hourly_fo(device_id: str, device_type: str, hour: int, 
                        base_time: datetime, slices_per_hour: int = 30,
                        energy_profile: Optional[List[Tuple[float, float]]] = None) -> FlexOffer:
        """
        创建小时级FlexOffer
        
        Args:
            device_id: 设备ID
            device_type: 设备类型
            hour: 小时 (0-23)
            base_time: 基准时间
            slices_per_hour: 每小时的时间片数量（默认30，每片2分钟）
            energy_profile: 能量轮廓 [(e_min, e_max), ...]
        """
        fo_id = f"{device_id}_fo_h{hour}"
        start_time = base_time.replace(hour=hour, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(hours=1)
        
        # 计算每个时间片的长度
        slice_duration_minutes = 60.0 / slices_per_hour
        
        slices = []
        for i in range(slices_per_hour):
            slice_start = start_time + timedelta(minutes=i * slice_duration_minutes)
            slice_end = slice_start + timedelta(minutes=slice_duration_minutes)
            
            # 获取能量值
            if energy_profile and i < len(energy_profile):
                e_min, e_max = energy_profile[i]
            else:
                # 默认值
                e_min, e_max = 0.0, 0.0
            
            slice = FOSlice(
                slice_id=i,
                start_time=slice_start,
                end_time=slice_end,
                energy_min=e_min,
                energy_max=e_max,
                duration_minutes=slice_duration_minutes,
                device_type=device_type,
                device_id=device_id
            )
            slices.append(slice)
        
        return FlexOffer(
            fo_id=fo_id,
            hour=hour,
            start_time=start_time,
            end_time=end_time,
            device_id=device_id,
            device_type=device_type,
            slices=slices
        )
    
    @staticmethod
    def convert_from_sfo(sfo_data: Dict[str, Any], device_id: str, 
                        device_type: str, hour: int, base_time: datetime) -> FlexOffer:
        """从SFO数据转换为标准FO"""
        # 假设SFO数据包含时间序列的能量边界
        e_min_list = sfo_data.get('e_min', [])
        e_max_list = sfo_data.get('e_max', [])
        
        # 创建能量轮廓
        energy_profile = [(e_min_list[i] if i < len(e_min_list) else 0.0,
                          e_max_list[i] if i < len(e_max_list) else 0.0)
                         for i in range(max(len(e_min_list), len(e_max_list), 30))]
        
        return FOFactory.create_hourly_fo(
            device_id=device_id,
            device_type=device_type,
            hour=hour,
            base_time=base_time,
            slices_per_hour=len(energy_profile),
            energy_profile=energy_profile
        ) 