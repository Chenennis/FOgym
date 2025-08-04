from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd
import math
import logging
import sys
import os
from dataclasses import dataclass
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# Add project root directory to system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import standard FlexOffer structure
from fo_common.flexoffer import FlexOffer, FOSlice

# Create logger
logger = logging.getLogger(__name__)

@dataclass
class AggregatedFlexOffer:
    """Aggregated FlexOffer (AFO)"""
    afo_id: str                           # Aggregated FlexOffer ID
    source_fo_ids: List[str]              # List of FO IDs participating in aggregation
    aggregated_fo: FlexOffer              # Aggregated FO
    aggregation_method: str               # Aggregation method ("LP" or "DP")
    total_energy_min: float = 0.0         # Total minimum energy
    total_energy_max: float = 0.0         # Total maximum energy
    power_profile_rmse: float = 0.0       # Power profile RMSE
    power_profile_cv: float = 0.0         # Power profile coefficient of variation
    slice_count: int = 0                  # Number of time slices
    
    def __post_init__(self):
        """Calculate attributes after initialization"""
        if self.aggregated_fo:
            self.total_energy_min = self.aggregated_fo.total_energy_min
            self.total_energy_max = self.aggregated_fo.total_energy_max
            self.slice_count = len(self.aggregated_fo.slices)
            self._calculate_power_metrics()
    
    def _calculate_power_metrics(self):
        """Calculate power-related metrics"""
        if not self.aggregated_fo.slices:
            return
        
        # Get power profile
        p_min, p_max = self.aggregated_fo.get_power_profile()
        avg_power = [(p_min[i] + p_max[i]) / 2 for i in range(len(p_min))]
        
        # Filter out possible NaN values
        valid_power = [p for p in avg_power if not np.isnan(p)]
        
        # If no valid values, set default values
        if not valid_power:
            self.power_profile_rmse = 0.0
            self.power_profile_cv = 0.0
            return
            
        # Calculate RMSE (relative to target power threshold 100kW)
        spt = 100.0  # Target power threshold
        self.power_profile_rmse = float(np.sqrt(np.mean([(p - spt) ** 2 for p in valid_power])))
        
        # Calculate coefficient of variation CV
        std_dev = float(np.std(valid_power))
        mean_power = float(np.mean(valid_power))
        self.power_profile_cv = std_dev / mean_power if mean_power != 0 else 0.0

class FOAggregator(ABC):
    """FlexOffer Aggregator Abstract Base Class"""
    
    def __init__(self, spt: float = 100.0, ppt: int = 23, tf_threshold: float = 1.0, 
                 power_deviation: float = 5.0):
        """
        Initialize aggregator
        
        Args:
            spt: Slice Power Threshold (kW)
            ppt: Power Profile Threshold (hours)
            tf_threshold: Time flexibility threshold
            power_deviation: Allowed power deviation (kW)
        """
        self.spt = spt  # Slice Power Threshold
        self.ppt = ppt  # Power Profile Threshold  
        self.tf_threshold = tf_threshold
        self.power_deviation = power_deviation
        self.results: List[AggregatedFlexOffer] = []
    
    @abstractmethod
    def initialize(self, flex_offers: List[FlexOffer]) -> Tuple[List[FlexOffer], List[FlexOffer], Optional[FlexOffer], int]:
        """
        Initialize aggregation process
        
        Args:
            flex_offers: Input FlexOffer list
            
        Returns:
            Tuple[PF, UF, fini, round]: Processing set, Unprocessed set, Initial FO, Round
        """
        pass
    
    def binary_aggregation(self, fo1: FlexOffer, fo2: FlexOffer) -> Optional[FlexOffer]:
        """
        Binary aggregation operation
        
        Args:
            fo1: First FlexOffer
            fo2: Second FlexOffer
            
        Returns:
            Aggregated FlexOffer or None
        """
        # Check compatibility
        if not fo1.is_compatible_with(fo2, self.tf_threshold):
            return None
        
        # Ensure both FOs have the same number of time slices
        max_slices = max(len(fo1.slices), len(fo2.slices))
        
        # Create aggregated time slices
        aggregated_slices = []
        for i in range(max_slices):
            # Get energy from both FOs at this time slice
            e1_min, e1_max = fo1.get_energy_bounds(i) if i < len(fo1.slices) else (0.0, 0.0)
            e2_min, e2_max = fo2.get_energy_bounds(i) if i < len(fo2.slices) else (0.0, 0.0)
            
            # Aggregate energy (simple addition)
            agg_e_min = e1_min + e2_min
            agg_e_max = e1_max + e2_max
            
            # Use time information from the first FO as reference
            base_slice = fo1.slices[i] if i < len(fo1.slices) else fo2.slices[i]
            
            aggregated_slice = FOSlice(
                slice_id=i,
                start_time=base_slice.start_time,
                end_time=base_slice.end_time,
                energy_min=agg_e_min,
                energy_max=agg_e_max,
                duration_minutes=base_slice.duration_minutes,
                device_type="aggregated",
                device_id=f"agg_{fo1.device_id}_{fo2.device_id}"
            )
            aggregated_slices.append(aggregated_slice)
        
        # Create aggregated FlexOffer
        aggregated_fo = FlexOffer(
            fo_id=f"agg_{fo1.fo_id}_{fo2.fo_id}",
            hour=fo1.hour,
            start_time=fo1.start_time,
            end_time=fo1.end_time,
            device_id=f"agg_{fo1.device_id}_{fo2.device_id}",
            device_type="aggregated",
            slices=aggregated_slices
        )
        
        return aggregated_fo
    
    def calculate_rmse(self, fo: FlexOffer) -> float:
        """Calculate FlexOffer RMSE relative to target power"""
        p_min, p_max = fo.get_power_profile()
        avg_power = [(p_min[i] + p_max[i]) / 2 for i in range(len(p_min))]
        
        # Filter out possible NaN or infinite values
        valid_power = [p for p in avg_power if not np.isnan(p) and np.isfinite(p)]
        
        if not valid_power:
            return 0.0
            
        return np.sqrt(np.mean([(p - self.spt) ** 2 for p in valid_power]))
    
    def calculate_cv(self, fo: FlexOffer) -> float:
        """Calculate coefficient of variation for FlexOffer power profile"""
        p_min, p_max = fo.get_power_profile()
        avg_power = [(p_min[i] + p_max[i]) / 2 for i in range(len(p_min))]
        
        # Filter out possible NaN or infinite values
        valid_power = [p for p in avg_power if not np.isnan(p) and np.isfinite(p)]
        
        if not valid_power:
            return 0.0
            
        std_dev = float(np.std(valid_power))
        mean_power = float(np.mean(valid_power))
        
        if mean_power == 0:
            return 0.0
            
        return std_dev / mean_power
    
    def process(self, PF: List[FlexOffer], fini: FlexOffer) -> FlexOffer:
        """
        Processing stage - Execute binary aggregation operations
        
        Args:
            PF: Processing set
            fini: Initial FlexOffer
            
        Returns:
            Final aggregated FlexOffer
        """
        current_fo = fini
        used_fos = [fini.fo_id]
        
        # Sort PF by time flexibility in descending order
        PF_sorted = sorted(PF, key=lambda fo: fo.tf(), reverse=True)
        
        for candidate_fo in PF_sorted:
            if candidate_fo.fo_id in used_fos:
                continue
                
            # Try binary aggregation
            aggregated = self.binary_aggregation(current_fo, candidate_fo)
            
            if aggregated:
                # Calculate quality metrics after aggregation
                new_rmse = self.calculate_rmse(aggregated)
                current_rmse = self.calculate_rmse(current_fo)
                
                # Accept aggregation if RMSE improves
                if new_rmse < current_rmse:
                    new_cv = self.calculate_cv(aggregated)
                    current_cv = self.calculate_cv(current_fo)
                    
                    # Further check if CV improves
                    if new_cv <= current_cv:
                        current_fo = aggregated
                        used_fos.append(candidate_fo.fo_id)
                        logger.debug(f"Aggregation successful: {candidate_fo.fo_id}, RMSE: {new_rmse:.2f}, CV: {new_cv:.2f}")
        
        return current_fo
    
    def aggregate(self, flex_offers: List[FlexOffer]) -> List[AggregatedFlexOffer]:
        """
        Aggregate FlexOffer list
        
        Args:
            flex_offers: Input FlexOffer list
            
        Returns:
            List of aggregation results
        """
        self.results = []
        
        if not flex_offers:
            return self.results
        
        # Enhanced logging: record aggregation start information
        logger.info(f"Starting aggregation - Method: {self.__class__.__name__}, Input FO count: {len(flex_offers)}")
        logger.info(f"Input FO characteristics - Average profile size: {sum(fo.profile_size() for fo in flex_offers) / len(flex_offers):.2f}, "
                   f"Average time flexibility: {sum(fo.tf() for fo in flex_offers) / len(flex_offers):.2f}")
        
        # Initialization
        PF, UF, fini, round_num = self.initialize(flex_offers)
        
        logger.info(f"Aggregation initialization complete: PF={len(PF)}, UF={len(UF)}, Algorithm={self.__class__.__name__}")
        
        # Processing stage - only process when fini is not None and PF is not empty
        if fini is not None and PF:
            aggregated_fo = self.process(PF, fini)
            
            # Create aggregation result
            source_fo_ids = [fini.fo_id] + [fo.fo_id for fo in PF]
            afo = AggregatedFlexOffer(
                afo_id=f"AFO_{round_num}_{self.__class__.__name__}",
                source_fo_ids=source_fo_ids,
                aggregated_fo=aggregated_fo,
                aggregation_method=self.__class__.__name__
            )
            
            self.results.append(afo)
            # Enhanced logging: record detailed aggregation result information
            logger.info(f"Aggregation complete: Method={self.__class__.__name__}, AFO contains {len(source_fo_ids)} FOs, "
                       f"Total energy range [{afo.total_energy_min:.2f}, {afo.total_energy_max:.2f}], "
                       f"Profile size={afo.aggregated_fo.profile_size()}, "
                       f"Time flexibility={afo.aggregated_fo.tf():.2f}, "
                       f"Power RMSE={afo.power_profile_rmse:.2f}, "
                       f"Power CV={afo.power_profile_cv:.2f}")
        elif fini is not None:
            # If only fini exists, with no PF, use fini directly as result
            afo = AggregatedFlexOffer(
                afo_id=f"AFO_{round_num}_{self.__class__.__name__}_single",
                source_fo_ids=[fini.fo_id],
                aggregated_fo=fini,
                aggregation_method=self.__class__.__name__
            )
            self.results.append(afo)
            # Enhanced logging: record single FO aggregation result
            logger.info(f"Single FO aggregation: Method={self.__class__.__name__}, FO_ID={fini.fo_id}, "
                       f"Total energy range [{afo.total_energy_min:.2f}, {afo.total_energy_max:.2f}], "
                       f"Profile size={afo.aggregated_fo.profile_size()}")
        
        # Process unprocessed FOs in UF
        for unused_fo in UF:
            afo = AggregatedFlexOffer(
                afo_id=f"AFO_unused_{unused_fo.fo_id}",
                source_fo_ids=[unused_fo.fo_id],
                aggregated_fo=unused_fo,
                aggregation_method=f"{self.__class__.__name__}_unused"
            )
            self.results.append(afo)
        
        # Enhanced logging: record final aggregation results
        logger.info(f"Aggregation result statistics: Method={self.__class__.__name__}, Result count={len(self.results)}, "
                   f"Processed FO count={len(flex_offers) - len(UF)}, Unprocessed FO count={len(UF)}")
        
        return self.results

class LongestProfileAggregator(FOAggregator):
    """Longest Profile (LP) Aggregation Algorithm"""
    
    def initialize(self, flex_offers: List[FlexOffer]) -> Tuple[List[FlexOffer], List[FlexOffer], Optional[FlexOffer], int]:
        """
        LP initialization method
        
        Implementation based on Algorithm 2:
        1. Find all FOs with maximum profile size
        2. Select the FO with highest time flexibility as fini from the longest FOs
        3. Add all other FOs to processing set PF
        """
        if not flex_offers:
            return [], [], None, 1
        
        # Step 1: Find maximum profile size
        max_profile_size = max(fo.profile_size() for fo in flex_offers)
        longest_fos = [fo for fo in flex_offers if fo.profile_size() == max_profile_size]
        
        logger.info(f"LP initialization: Max profile size={max_profile_size}, Longest FO count={len(longest_fos)}")
        
        # Step 2: Select FO with highest time flexibility as fini from longest FOs
        fini = max(longest_fos, key=lambda fo: fo.tf())
        
        # Step 3: Add all other FOs to processing set PF
        PF = [fo for fo in flex_offers if fo.fo_id != fini.fo_id]
        UF = []  # Unprocessed set is empty in LP method
        
        logger.info(f"LP initialization complete: fini={fini.fo_id}(profile_size={fini.profile_size()}, tf={fini.tf():.2f})")
        
        return PF, UF, fini, 1

class DynamicProfileAggregator(FOAggregator):
    """Dynamic Profile (DP) Aggregation Algorithm"""
    
    def initialize(self, flex_offers: List[FlexOffer]) -> Tuple[List[FlexOffer], List[FlexOffer], Optional[FlexOffer], int]:
        """
        DP initialization method
        
        Implementation based on Algorithm 3:
        1. Calculate profile size upper fence
        2. Filter FO set, exclude outliers
        3. Select the longest and most flexible FO from filtered set
        """
        if not flex_offers:
            return [], [], None, 1
        
        # Step 1: Calculate upper fence
        profile_sizes = [fo.profile_size() for fo in flex_offers]
        uf = self._upper_fence_profile_size(profile_sizes)
        
        logger.info(f"DP initialization: Profile size range [{min(profile_sizes)}, {max(profile_sizes)}], Upper fence={uf:.2f}")
        
        # Step 2: Filter FO set
        PF_candidates = [fo for fo in flex_offers if fo.profile_size() <= uf]
        UF = [fo for fo in flex_offers if fo.profile_size() > uf]  # Outliers
        
        logger.info(f"DP filtering: Candidate FOs={len(PF_candidates)}, Outlier FOs={len(UF)}")
        
        if not PF_candidates:
            # If all FOs are filtered out, fall back to the smallest FO
            fini = min(flex_offers, key=lambda fo: fo.profile_size())
            PF = [fo for fo in flex_offers if fo.fo_id != fini.fo_id]
            UF = []
            logger.warning("DP initialization: All FOs filtered out, falling back to smallest profile FO")
            return PF, UF, fini, 1
        
        # Step 3: Select the longest and most flexible FO from filtered set
        max_size_in_pf = max(fo.profile_size() for fo in PF_candidates)
        longest_in_pf = [fo for fo in PF_candidates if fo.profile_size() == max_size_in_pf]
        fini = max(longest_in_pf, key=lambda fo: fo.tf())
        
        # Step 4: Remove fini from processing set
        PF = [fo for fo in PF_candidates if fo.fo_id != fini.fo_id]
        
        logger.info(f"DP initialization complete: fini={fini.fo_id}(profile_size={fini.profile_size()}, tf={fini.tf():.2f})")
        
        return PF, UF, fini, 1
    
    def _upper_fence_profile_size(self, sizes: List[int]) -> float:
        """
        Calculate upper fence using quartile method
        Upper Fence = Q3 + 1.5 * IQR
        """
        if not sizes:
            return 0.0
        
        sorted_sizes = sorted(sizes)
        n = len(sorted_sizes)
        
        # Calculate quartiles
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        
        q1 = sorted_sizes[q1_idx] if q1_idx < n else sorted_sizes[-1]
        q3 = sorted_sizes[q3_idx] if q3_idx < n else sorted_sizes[-1]
        
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        
        logger.debug(f"Quartile calculation: Q1={q1}, Q3={q3}, IQR={iqr}, Upper Fence={upper_fence}")
        
        return upper_fence

class FOAggregatorFactory:
    """FlexOffer Aggregator Factory"""
    
    @staticmethod
    def create_aggregator(method: str, **kwargs) -> FOAggregator:
        """
        Create aggregator
        
        Args:
            method: Aggregation method ("LP" or "DP")
            **kwargs: Aggregator parameters
            
        Returns:
            Aggregator instance
        """
        if method.upper() == "LP":
            return LongestProfileAggregator(**kwargs)
        elif method.upper() == "DP":
            return DynamicProfileAggregator(**kwargs)
        else:
            raise ValueError(f"Unsupported aggregation method: {method}. Supported methods: LP, DP")
    
    @staticmethod
    def get_available_methods() -> List[str]:
        """Get list of available aggregation methods"""
        return ["LP", "DP"]

# Convenience function
def aggregate_flex_offers(flex_offers: List[FlexOffer], method: str = "DP", **kwargs) -> List[AggregatedFlexOffer]:
    """
    Convenience function for aggregating FlexOffers
    
    Args:
        flex_offers: List of FlexOffers
        method: Aggregation method ("LP" or "DP")
        **kwargs: Aggregator parameters
        
    Returns:
        List of aggregation results
    """
    aggregator = FOAggregatorFactory.create_aggregator(method, **kwargs)
    return aggregator.aggregate(flex_offers) 