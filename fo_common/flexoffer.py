from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import numpy as np

@dataclass
class FOSlice:
    """FlexOffer slice - represents energy demand/supply within a specific time period"""
    slice_id: int                    # slice ID (number in hour)
    start_time: datetime            # start time
    end_time: datetime              # end time  
    energy_min: float               # minimum energy demand/supply (kWh)
    energy_max: float               # maximum energy demand/supply (kWh)
    duration_minutes: float         # slice duration (minutes)
    device_type: str = "unknown"    # device type
    device_id: str = ""             # device ID
    priority: int = 3               # priority (1-5, 1 highest)
    flexibility_factor: float = 0.5 # flexibility factor [0, 1]
    
    def get_duration_hours(self) -> float:
        """get slice duration (hours)"""
        return self.duration_minutes / 60.0
    
    def get_energy_range(self) -> float:
        """get energy range"""
        return self.energy_max - self.energy_min
    
    def get_average_energy(self) -> float:
        """get average energy"""
        return (self.energy_min + self.energy_max) / 2.0

@dataclass  
class FlexOffer:
    """standard FlexOffer (FO) - represents energy demand/supply profile within an hour"""
    fo_id: str                      # FlexOffer ID
    hour: int                       # hour (0-23)
    start_time: datetime            # start time
    end_time: datetime              # end time
    device_id: str                  # device ID
    device_type: str                # device type
    slices: List[FOSlice]           # slice list
    total_energy_min: float = 0.0   # total minimum energy
    total_energy_max: float = 0.0   # total maximum energy
    profile_length: int = 0         # profile length (non-zero slice count)
    time_flexibility: float = 0.0   # time flexibility
    
    def __post_init__(self):
        """post-initialization processing"""
        self._calculate_properties()
    
    def _calculate_properties(self):
        """calculate FO basic properties"""
        if self.slices:
            self.total_energy_min = sum(s.energy_min for s in self.slices)
            self.total_energy_max = sum(s.energy_max for s in self.slices)
            
            # calculate profile length (non-zero energy slice count)
            self.profile_length = sum(1 for s in self.slices 
                                    if s.energy_min != 0 or s.energy_max != 0)
            
            # calculate time flexibility (average energy range)
            if self.profile_length > 0:
                self.time_flexibility = sum(s.get_energy_range() for s in self.slices) / self.profile_length
            else:
                self.time_flexibility = 0.0
    
    def add_slice(self, slice: FOSlice):
        """add slice"""
        self.slices.append(slice)
        self._calculate_properties()
    
    def get_slice(self, slice_id: int) -> Optional[FOSlice]:
        """get slice by ID"""
        for slice in self.slices:
            if slice.slice_id == slice_id:
                return slice
        return None
    
    def get_energy_bounds(self, slice_id: int) -> Tuple[float, float]:
        """get energy bounds of the slice"""
        slice = self.get_slice(slice_id)
        if slice:
            return slice.energy_min, slice.energy_max
        return 0.0, 0.0
    
    def get_energy_profile(self) -> Tuple[List[float], List[float]]:
        """get energy profile"""
        e_min = [s.energy_min for s in self.slices]
        e_max = [s.energy_max for s in self.slices]
        return e_min, e_max
    
    def get_power_profile(self) -> Tuple[List[float], List[float]]:
        """get power profile (kW)"""
        p_min = []
        p_max = []
        for s in self.slices:
            duration_hours = s.get_duration_hours()
            if duration_hours > 0:
                p_min.append(s.energy_min / duration_hours)
                p_max.append(s.energy_max / duration_hours)
            else:
                # handle zero duration
                p_min.append(0.0)
                p_max.append(0.0)
        return p_min, p_max
    
    def profile_size(self) -> int:
        """get profile size"""
        return self.profile_length
    
    def tf(self) -> float:
        """get time flexibility"""
        return self.time_flexibility
    
    def is_compatible_with(self, other: 'FlexOffer', tf_threshold: float = 1.0) -> bool:
        """check compatibility with another FO"""
        if not isinstance(other, FlexOffer):
            return False
        
        # check time range consistency
        if len(self.slices) != len(other.slices):
            return False
        
        # check time flexibility within threshold
        tf_diff = abs(self.time_flexibility - other.time_flexibility)
        return tf_diff <= tf_threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """convert to dictionary format"""
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
        """create FlexOffer from dictionary"""
        # restore slices
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
    """FlexOffer factory class - for creating standardized FlexOffer"""
    
    @staticmethod
    def create_hourly_fo(device_id: str, device_type: str, hour: int, 
                        base_time: datetime, slices_per_hour: int = 30,
                        energy_profile: Optional[List[Tuple[float, float]]] = None) -> FlexOffer:
        """
        create hourly FlexOffer
        
        Args:
            device_id: device ID
            device_type: device type
            hour: hour (0-23)
            base_time: base time
            slices_per_hour: number of slices per hour (default 30, 2 minutes per slice)
            energy_profile: energy profile [(e_min, e_max), ...]
        """
        fo_id = f"{device_id}_fo_h{hour}"
        start_time = base_time.replace(hour=hour, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(hours=1)
        
        # calculate slice duration
        slice_duration_minutes = 60.0 / slices_per_hour
        
        slices = []
        for i in range(slices_per_hour):
            slice_start = start_time + timedelta(minutes=i * slice_duration_minutes)
            slice_end = slice_start + timedelta(minutes=slice_duration_minutes)
            
            # get energy value
            if energy_profile and i < len(energy_profile):
                e_min, e_max = energy_profile[i]
            else:
                # default value
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
        """convert SFO data to standard FO"""
        # assume SFO data contains energy boundaries of time series
        e_min_list = sfo_data.get('e_min', [])
        e_max_list = sfo_data.get('e_max', [])
        
        # create energy profile
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