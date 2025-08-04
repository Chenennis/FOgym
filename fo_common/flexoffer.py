from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import numpy as np

@dataclass
class FOSlice:
    """FlexOffer time slice - represents energy demand/supply during a specific time period"""
    slice_id: int                    # Time slice ID (sequence number within an hour)
    start_time: datetime            # Start time
    end_time: datetime              # End time  
    energy_min: float               # Minimum energy demand/supply (kWh)
    energy_max: float               # Maximum energy demand/supply (kWh)
    duration_minutes: float         # Time slice duration (minutes)
    device_type: str = "unknown"    # Device type
    device_id: str = ""             # Device ID
    priority: int = 3               # Priority (1-5, 1 is highest)
    flexibility_factor: float = 0.5 # Flexibility factor [0, 1]
    
    def get_duration_hours(self) -> float:
        """Get time slice duration (hours)"""
        return self.duration_minutes / 60.0
    
    def get_energy_range(self) -> float:
        """Get energy range"""
        return self.energy_max - self.energy_min
    
    def get_average_energy(self) -> float:
        """Get average energy"""
        return (self.energy_min + self.energy_max) / 2.0

@dataclass  
class FlexOffer:
    """Standard FlexOffer (FO) - represents energy demand/supply profile for one hour"""
    fo_id: str                      # FlexOffer ID
    hour: int                       # Hour (0-23)
    start_time: datetime            # Start time
    end_time: datetime              # End time
    device_id: str                  # Device ID
    device_type: str                # Device type
    slices: List[FOSlice]           # List of time slices
    total_energy_min: float = 0.0   # Total minimum energy
    total_energy_max: float = 0.0   # Total maximum energy
    profile_length: int = 0         # Profile length (number of non-zero slices)
    time_flexibility: float = 0.0   # Time flexibility
    
    def __post_init__(self):
        """Post-initialization processing"""
        self._calculate_properties()
    
    def _calculate_properties(self):
        """Calculate basic properties of the FO"""
        if self.slices:
            self.total_energy_min = sum(s.energy_min for s in self.slices)
            self.total_energy_max = sum(s.energy_max for s in self.slices)
            
            # Calculate profile length (number of slices with non-zero energy)
            self.profile_length = sum(1 for s in self.slices 
                                    if s.energy_min != 0 or s.energy_max != 0)
            
            # Calculate time flexibility (average energy range)
            if self.profile_length > 0:
                self.time_flexibility = sum(s.get_energy_range() for s in self.slices) / self.profile_length
            else:
                self.time_flexibility = 0.0
    
    def add_slice(self, slice: FOSlice):
        """Add a time slice"""
        self.slices.append(slice)
        self._calculate_properties()
    
    def get_slice(self, slice_id: int) -> Optional[FOSlice]:
        """Get time slice by ID"""
        for slice in self.slices:
            if slice.slice_id == slice_id:
                return slice
        return None
    
    def get_energy_bounds(self, slice_id: int) -> Tuple[float, float]:
        """Get energy bounds for a specific time slice"""
        slice = self.get_slice(slice_id)
        if slice:
            return slice.energy_min, slice.energy_max
        return 0.0, 0.0
    
    def get_energy_profile(self) -> Tuple[List[float], List[float]]:
        """Get energy profile"""
        e_min = [s.energy_min for s in self.slices]
        e_max = [s.energy_max for s in self.slices]
        return e_min, e_max
    
    def get_power_profile(self) -> Tuple[List[float], List[float]]:
        """Get power profile (kW)"""
        p_min = []
        p_max = []
        for s in self.slices:
            duration_hours = s.get_duration_hours()
            if duration_hours > 0:
                p_min.append(s.energy_min / duration_hours)
                p_max.append(s.energy_max / duration_hours)
            else:
                # Handle case where duration is 0
                p_min.append(0.0)
                p_max.append(0.0)
        return p_min, p_max
    
    def profile_size(self) -> int:
        """Get profile size"""
        return self.profile_length
    
    def tf(self) -> float:
        """Get time flexibility"""
        return self.time_flexibility
    
    def is_compatible_with(self, other: 'FlexOffer', tf_threshold: float = 1.0) -> bool:
        """Check compatibility with another FO"""
        if not isinstance(other, FlexOffer):
            return False
        
        # Check if time ranges are consistent
        if len(self.slices) != len(other.slices):
            return False
        
        # Check if time flexibility is within threshold
        tf_diff = abs(self.time_flexibility - other.time_flexibility)
        return tf_diff <= tf_threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
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
        """Create FlexOffer from dictionary"""
        # Restore time slices
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
    """Factory class for creating FlexOffers"""
    
    @staticmethod
    def create_hourly_fo(device_id: str, device_type: str, hour: int, 
                        base_time: datetime, slices_per_hour: int = 30,
                        energy_profile: Optional[List[Tuple[float, float]]] = None) -> FlexOffer:
        """
        Create an hourly FlexOffer with the specified number of time slices.
        
        Args:
            device_id: Device ID
            device_type: Device type
            hour: Hour of the day (0-23)
            base_time: Base time for the FlexOffer (usually start of the day)
            slices_per_hour: Number of time slices per hour
            energy_profile: Optional energy profile [(min1, max1), (min2, max2), ...]
                            If provided, must match slices_per_hour in length
        
        Returns:
            FlexOffer object
        """
        # Calculate start and end times
        start_time = base_time + timedelta(hours=hour)
        end_time = start_time + timedelta(hours=1)
        
        # Create FlexOffer
        fo_id = f"{device_id}_{device_type}_{hour}_{base_time.strftime('%Y%m%d')}"
        fo = FlexOffer(
            fo_id=fo_id,
            hour=hour,
            start_time=start_time,
            end_time=end_time,
            device_id=device_id,
            device_type=device_type,
            slices=[]
        )
        
        # Create time slices
        slice_duration = 60 / slices_per_hour  # minutes
        
        for i in range(slices_per_hour):
            slice_start = start_time + timedelta(minutes=i * slice_duration)
            slice_end = slice_start + timedelta(minutes=slice_duration)
            
            # Set energy values
            if energy_profile and i < len(energy_profile):
                energy_min, energy_max = energy_profile[i]
            else:
                energy_min, energy_max = 0.0, 0.0
            
            # Create slice
            slice = FOSlice(
                slice_id=i,
                start_time=slice_start,
                end_time=slice_end,
                energy_min=energy_min,
                energy_max=energy_max,
                duration_minutes=slice_duration,
                device_type=device_type,
                device_id=device_id
            )
            
            fo.add_slice(slice)
        
        return fo
    
    @staticmethod
    def convert_from_sfo(sfo_data: Dict[str, Any], device_id: str, 
                        device_type: str, hour: int, base_time: datetime) -> FlexOffer:
        """
        Convert SFO data to FlexOffer format.
        
        Args:
            sfo_data: SFO data in dictionary format
            device_id: Device ID
            device_type: Device type
            hour: Hour of the day (0-23)
            base_time: Base time for the FlexOffer
            
        Returns:
            FlexOffer object
        """
        # Calculate start and end times
        start_time = base_time + timedelta(hours=hour)
        end_time = start_time + timedelta(hours=1)
        
        # Create FlexOffer
        fo_id = f"{device_id}_{device_type}_{hour}_{base_time.strftime('%Y%m%d')}"
        fo = FlexOffer(
            fo_id=fo_id,
            hour=hour,
            start_time=start_time,
            end_time=end_time,
            device_id=device_id,
            device_type=device_type,
            slices=[]
        )
        
        # Extract energy profile from SFO
        if 'slices' in sfo_data:
            slice_duration = 60 / len(sfo_data['slices'])  # minutes
            
            for i, sfo_slice in enumerate(sfo_data['slices']):
                slice_start = start_time + timedelta(minutes=i * slice_duration)
                slice_end = slice_start + timedelta(minutes=slice_duration)
                
                # Create slice
                slice = FOSlice(
                    slice_id=i,
                    start_time=slice_start,
                    end_time=slice_end,
                    energy_min=sfo_slice.get('energy_min', 0.0),
                    energy_max=sfo_slice.get('energy_max', 0.0),
                    duration_minutes=slice_duration,
                    device_type=device_type,
                    device_id=device_id,
                    priority=sfo_slice.get('priority', 3),
                    flexibility_factor=sfo_slice.get('flexibility_factor', 0.5)
                )
                
                fo.add_slice(slice)
        
        return fo 