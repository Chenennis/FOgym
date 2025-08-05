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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fo_common.flexoffer import FlexOffer, FOSlice

logger = logging.getLogger(__name__)

@dataclass
class AggregatedFlexOffer:
    """aggregated FlexOffer (AFO)"""
    afo_id: str                           # aggregated FlexOffer ID
    source_fo_ids: List[str]              # source FO ID list
    aggregated_fo: FlexOffer              # aggregated FO
    aggregation_method: str               # aggregation method ("LP" or "DP")
    total_energy_min: float = 0.0         # total minimum energy
    total_energy_max: float = 0.0         # total maximum energy
    power_profile_rmse: float = 0.0       # power profile RMSE
    power_profile_cv: float = 0.0         # power profile coefficient of variation
    slice_count: int = 0                  # slice count
    
    def __post_init__(self):
        """calculate properties after initialization"""
        if self.aggregated_fo:
            self.total_energy_min = self.aggregated_fo.total_energy_min
            self.total_energy_max = self.aggregated_fo.total_energy_max
            self.slice_count = len(self.aggregated_fo.slices)
            self._calculate_power_metrics()
    
    def _calculate_power_metrics(self):
        """calculate power related metrics"""
        if not self.aggregated_fo.slices:
            return
        
        # get power profile
        p_min, p_max = self.aggregated_fo.get_power_profile()
        avg_power = [(p_min[i] + p_max[i]) / 2 for i in range(len(p_min))]
        
        # filter out possible NaN values
        valid_power = [p for p in avg_power if not np.isnan(p)]
        
        # if no valid values, set default values
        if not valid_power:
            self.power_profile_rmse = 0.0
            self.power_profile_cv = 0.0
            return
            
        # calculate RMSE (relative to target power threshold 100kW)
        spt = 100.0  # target power threshold
        self.power_profile_rmse = float(np.sqrt(np.mean([(p - spt) ** 2 for p in valid_power])))
        
        # calculate coefficient of variation (CV)
        std_dev = float(np.std(valid_power))
        mean_power = float(np.mean(valid_power))
        self.power_profile_cv = std_dev / mean_power if mean_power != 0 else 0.0

class FOAggregator(ABC):
    """FlexOffer aggregator abstract base class"""
    
    def __init__(self, spt: float = 100.0, ppt: int = 23, tf_threshold: float = 1.0, 
                 power_deviation: float = 5.0):
        """
        initialize aggregator
        
        Args:
            spt: slice power threshold (kW)
            ppt: power profile threshold (hours)
            tf_threshold: time flexibility threshold
            power_deviation: allowed power deviation (kW)
        """
        self.spt = spt  # Slice Power Threshold
        self.ppt = ppt  # Power Profile Threshold  
        self.tf_threshold = tf_threshold
        self.power_deviation = power_deviation
        self.results: List[AggregatedFlexOffer] = []
    
    @abstractmethod
    def initialize(self, flex_offers: List[FlexOffer]) -> Tuple[List[FlexOffer], List[FlexOffer], Optional[FlexOffer], int]:
        """
        initialize aggregation process
        
        Args:
            flex_offers: input FlexOffer list
            
        Returns:
            Tuple[PF, UF, fini, round]: processed set, unprocessed set, initial FO, round
        """
        pass
    
    def binary_aggregation(self, fo1: FlexOffer, fo2: FlexOffer) -> Optional[FlexOffer]:
        """
        binary aggregation operation
        
        Args:
            fo1: first FlexOffer
            fo2: second FlexOffer
            
        Returns:
            aggregated FlexOffer or None
        """
        # check compatibility
        if not fo1.is_compatible_with(fo2, self.tf_threshold):
            return None
        
        # ensure two FOs have the same number of slices
        max_slices = max(len(fo1.slices), len(fo2.slices))
        
        # create aggregated slices
        aggregated_slices = []
        for i in range(max_slices):
            # get energy of two FOs in the slice
            e1_min, e1_max = fo1.get_energy_bounds(i) if i < len(fo1.slices) else (0.0, 0.0)
            e2_min, e2_max = fo2.get_energy_bounds(i) if i < len(fo2.slices) else (0.0, 0.0)
            
            # aggregate energy 
            agg_e_min = e1_min + e2_min
            agg_e_max = e1_max + e2_max
            
            # use the time information of the first FO as the base
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
        
        # create aggregated FlexOffer
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
        """calculate RMSE of FlexOffer relative to target power"""
        p_min, p_max = fo.get_power_profile()
        avg_power = [(p_min[i] + p_max[i]) / 2 for i in range(len(p_min))]
        
        # filter out possible NaN or infinite values
        valid_power = [p for p in avg_power if not np.isnan(p) and np.isfinite(p)]
        
        if not valid_power:
            return 0.0
            
        return np.sqrt(np.mean([(p - self.spt) ** 2 for p in valid_power]))
    
    def calculate_cv(self, fo: FlexOffer) -> float:
        """calculate coefficient of variation of FlexOffer power profile"""
        p_min, p_max = fo.get_power_profile()
        avg_power = [(p_min[i] + p_max[i]) / 2 for i in range(len(p_min))]
        
        # filter out possible NaN or infinite values
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
        processing phase - execute binary aggregation operation
        
        Args:
            PF: processed set
            fini: initial FlexOffer
            
        Returns:
            final aggregated FlexOffer
        """
        current_fo = fini
        used_fos = [fini.fo_id]
        
        # sort PF by time flexibility in descending order
        PF_sorted = sorted(PF, key=lambda fo: fo.tf(), reverse=True)
        
        for candidate_fo in PF_sorted:
            if candidate_fo.fo_id in used_fos:
                continue
                
            # try binary aggregation
            aggregated = self.binary_aggregation(current_fo, candidate_fo)
            
            if aggregated:
                # calculate quality metrics after aggregation
                new_rmse = self.calculate_rmse(aggregated)
                current_rmse = self.calculate_rmse(current_fo)
                
                # if RMSE improves, accept aggregation
                if new_rmse < current_rmse:
                    new_cv = self.calculate_cv(aggregated)
                    current_cv = self.calculate_cv(current_fo)
                    
                    # further check if CV improves
                    if new_cv <= current_cv:
                        current_fo = aggregated
                        used_fos.append(candidate_fo.fo_id)
                        logger.debug(f"aggregation successful: {candidate_fo.fo_id}, RMSE: {new_rmse:.2f}, CV: {new_cv:.2f}")
        
        return current_fo
    
    def aggregate(self, flex_offers: List[FlexOffer]) -> List[AggregatedFlexOffer]:
        """
        aggregate FlexOffer list
        
        Args:
            flex_offers: input FlexOffer list
            
        Returns:
            aggregated result list
        """
        self.results = []
        
        if not flex_offers:
            return self.results
        
        # enhance logging: record aggregation start information
        logger.info(f"start aggregation - method: {self.__class__.__name__}, input FO count: {len(flex_offers)}")
        logger.info(f"input FO features - average profile size: {sum(fo.profile_size() for fo in flex_offers) / len(flex_offers):.2f}, "
                   f"average time flexibility: {sum(fo.tf() for fo in flex_offers) / len(flex_offers):.2f}")
        
        # initialize
        PF, UF, fini, round_num = self.initialize(flex_offers)
        
        logger.info(f"aggregation initialization completed: PF={len(PF)}, UF={len(UF)}, algorithm={self.__class__.__name__}")
        
        # processing phase - only process when fini is not None and PF is not empty
        if fini is not None and PF:
            aggregated_fo = self.process(PF, fini)
            
            # create aggregated result
            source_fo_ids = [fini.fo_id] + [fo.fo_id for fo in PF]
            afo = AggregatedFlexOffer(
                afo_id=f"AFO_{round_num}_{self.__class__.__name__}",
                source_fo_ids=source_fo_ids,
                aggregated_fo=aggregated_fo,
                aggregation_method=self.__class__.__name__
            )
            
            self.results.append(afo)
            # enhance logging: record detailed aggregation result information
            logger.info(f"aggregation completed: method={self.__class__.__name__}, AFO contains {len(source_fo_ids)} FOs, "
                       f"total energy range [{afo.total_energy_min:.2f}, {afo.total_energy_max:.2f}], "
                       f"profile size={afo.aggregated_fo.profile_size()}, "
                       f"time flexibility={afo.aggregated_fo.tf():.2f}, "
                       f"power RMSE={afo.power_profile_rmse:.2f}, "
                       f"power CV={afo.power_profile_cv:.2f}")
        elif fini is not None:
            # if only fini, no PF, use fini as result
            afo = AggregatedFlexOffer(
                afo_id=f"AFO_{round_num}_{self.__class__.__name__}_single",
                source_fo_ids=[fini.fo_id],
                aggregated_fo=fini,
                aggregation_method=self.__class__.__name__
            )
            self.results.append(afo)
            # enhance logging: record single FO aggregation result
            logger.info(f"single FO aggregation: method={self.__class__.__name__}, FO_ID={fini.fo_id}, "
                       f"total energy range [{afo.total_energy_min:.2f}, {afo.total_energy_max:.2f}], "
                       f"profile size={afo.aggregated_fo.profile_size()}")
        
        # process FOs in unprocessed set
        for unused_fo in UF:
            afo = AggregatedFlexOffer(
                afo_id=f"AFO_unused_{unused_fo.fo_id}",
                source_fo_ids=[unused_fo.fo_id],
                aggregated_fo=unused_fo,
                aggregation_method=f"{self.__class__.__name__}_unused"
            )
            self.results.append(afo)
        
        # enhance logging: record final aggregation result
        logger.info(f"aggregation result statistics: method={self.__class__.__name__}, result count={len(self.results)}, "
                   f"processed FO count={len(flex_offers) - len(UF)}, unprocessed FO count={len(UF)}")
        
        return self.results

class LongestProfileAggregator(FOAggregator):
    """Longest Profile (LP) aggregation algorithm"""
    
    def initialize(self, flex_offers: List[FlexOffer]) -> Tuple[List[FlexOffer], List[FlexOffer], Optional[FlexOffer], int]:
        """
        LP initialization method
        
        based on algorithm 2:
        1. find all FOs with the largest profile size
        2. select the FO with the highest time flexibility as fini
        3. add all other FOs to the processed set PF
        """
        if not flex_offers:
            return [], [], None, 1
        
        # step 1: find the largest profile size
        max_profile_size = max(fo.profile_size() for fo in flex_offers)
        longest_fos = [fo for fo in flex_offers if fo.profile_size() == max_profile_size]
        
        logger.info(f"LP initialization: largest profile size={max_profile_size}, longest FO count={len(longest_fos)}")
        
        # step 2: select the FO with the highest time flexibility as fini
        fini = max(longest_fos, key=lambda fo: fo.tf())
        
        # step 3: add all other FOs to the processed set PF
        PF = [fo for fo in flex_offers if fo.fo_id != fini.fo_id]
        UF = []  # in LP method, the unprocessed set is empty
        
        logger.info(f"LP initialization completed: fini={fini.fo_id}(profile_size={fini.profile_size()}, tf={fini.tf():.2f})")
        
        return PF, UF, fini, 1

class DynamicProfileAggregator(FOAggregator):
    """Dynamic Profile (DP) aggregation algorithm"""
    
    def initialize(self, flex_offers: List[FlexOffer]) -> Tuple[List[FlexOffer], List[FlexOffer], Optional[FlexOffer], int]:
        """
        DP initialization method
        
        based on algorithm 3:
        1. calculate the upper fence of the profile size
        2. filter the FO set, exclude abnormal values
        3. select the longest and most flexible FO in the filtered set
        """
        if not flex_offers:
            return [], [], None, 1
        
        # step 1: calculate the upper fence
        profile_sizes = [fo.profile_size() for fo in flex_offers]
        uf = self._upper_fence_profile_size(profile_sizes)
        
        logger.info(f"DP initialization: profile size range [{min(profile_sizes)}, {max(profile_sizes)}], upper fence={uf:.2f}")
        
        # step 2: filter the FO set
        PF_candidates = [fo for fo in flex_offers if fo.profile_size() <= uf]
        UF = [fo for fo in flex_offers if fo.profile_size() > uf]  # abnormal values
        
        logger.info(f"DP filtering: candidate FO={len(PF_candidates)}, abnormal FO={len(UF)}")
        
        if not PF_candidates:
            # if all FOs are filtered, revert to the smallest FO
            fini = min(flex_offers, key=lambda fo: fo.profile_size())
            PF = [fo for fo in flex_offers if fo.fo_id != fini.fo_id]
            UF = []
            logger.warning("DP initialization: all FOs are filtered, revert to the smallest FO")
            return PF, UF, fini, 1
        
        # step 3: select the longest and most flexible FO in the filtered set
        max_size_in_pf = max(fo.profile_size() for fo in PF_candidates)
        longest_in_pf = [fo for fo in PF_candidates if fo.profile_size() == max_size_in_pf]
        fini = max(longest_in_pf, key=lambda fo: fo.tf())
        
        # step 4: remove fini from the processed set
        PF = [fo for fo in PF_candidates if fo.fo_id != fini.fo_id]
        
        logger.info(f"DP initialization completed: fini={fini.fo_id}(profile_size={fini.profile_size()}, tf={fini.tf():.2f})")
        
        return PF, UF, fini, 1
    
    def _upper_fence_profile_size(self, sizes: List[int]) -> float:
        """
        calculate the upper fence using the quartile method
        Upper Fence = Q3 + 1.5 * IQR
        """
        if not sizes:
            return 0.0
        
        sorted_sizes = sorted(sizes)
        n = len(sorted_sizes)
        
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        
        q1 = sorted_sizes[q1_idx] if q1_idx < n else sorted_sizes[-1]
        q3 = sorted_sizes[q3_idx] if q3_idx < n else sorted_sizes[-1]
        
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        
        logger.debug(f"quartile calculation: Q1={q1}, Q3={q3}, IQR={iqr}, Upper Fence={upper_fence}")
        
        return upper_fence

class FOAggregatorFactory:
    """FlexOffer aggregator factory"""
    
    @staticmethod
    def create_aggregator(method: str, **kwargs) -> FOAggregator:
        """
        create aggregator
        
        Args:
            method: aggregation method ("LP" or "DP")
            **kwargs: aggregator parameters
            
        Returns:
            aggregator instance
        """
        if method.upper() == "LP":
            return LongestProfileAggregator(**kwargs)
        elif method.upper() == "DP":
            return DynamicProfileAggregator(**kwargs)
        else:
            raise ValueError(f"unsupported aggregation method: {method}. supported methods: LP, DP")
    
    @staticmethod
    def get_available_methods() -> List[str]:
        """get available aggregation method list"""
        return ["LP", "DP"]

# convenience function
def aggregate_flex_offers(flex_offers: List[FlexOffer], method: str = "DP", **kwargs) -> List[AggregatedFlexOffer]:
    """
    convenience function to aggregate FlexOffer
    
    Args:
        flex_offers: FlexOffer list
        method: aggregation method ("LP" or "DP")
        **kwargs: aggregator parameters
        
    Returns:
        aggregated result list
    """
    aggregator = FOAggregatorFactory.create_aggregator(method, **kwargs)
    return aggregator.aggregate(flex_offers) 