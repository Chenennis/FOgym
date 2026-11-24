import logging
import random
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from fo_generate.dfo import DFOSystem, DFOSlice
from fo_generate.sfo import SFOSystem, SFOSlice
from fo_aggregate import Manager, AggregatedFlexOffer
from fo_aggregate.manager import Manager


logger = logging.getLogger(__name__)

class WeatherModel:
    """weather model, process weather data and prediction"""
    
    WEATHER_TYPES = ["sunny", "cloudy", "rainy", "snowy"]
    
    def __init__(self, weather_data_file: Optional[str] = None, time_horizon: int = 24):
        """
        initialize weather model
        
        Args:
            weather_data_file: weather data file path
            time_horizon: time range
        """
        self.time_horizon = time_horizon
        self.current_step = 0
        
        # weather data
        self.weather_data = {
            'weather': ["sunny"] * time_horizon,
            'temperature': [20.0] * time_horizon,
            'solar_irradiance': [800.0] * time_horizon
        }
        
        # load weather data from file or generate weather data
        if weather_data_file and os.path.exists(weather_data_file):
            self.load_weather_data(weather_data_file)
        else:
            self.generate_weather_data()
    
    def load_weather_data(self, weather_data_file: str):
        """
        load weather data from file
        
        Args:
            weather_data_file: weather data file path
        """
        try:
            df = pd.read_csv(weather_data_file)
            
            # check if columns exist
            if 'weather' in df.columns:
                self.weather_data['weather'] = df['weather'].tolist()[:self.time_horizon]
                
            if 'temperature' in df.columns:
                self.weather_data['temperature'] = df['temperature'].tolist()[:self.time_horizon]
                
            if 'solar_irradiance' in df.columns:
                self.weather_data['solar_irradiance'] = df['solar_irradiance'].tolist()[:self.time_horizon]
                
            logger.info(f"success load weather data from {weather_data_file}")
        except Exception as e:
            logger.error(f"failed to load weather data: {e}")
            self.generate_weather_data()
    
    def generate_weather_data(self):
        """generate random weather data"""
        weather_probs = [0.5, 0.3, 0.15, 0.05]  # probability of each weather type
        
        for t in range(self.time_horizon):
            # weather type
            weather_type = np.random.choice(self.WEATHER_TYPES, p=weather_probs)
            self.weather_data['weather'][t] = weather_type
            
            # generate other parameters based on weather type
            if weather_type == "sunny":
                self.weather_data['temperature'][t] = random.uniform(20, 30)
                self.weather_data['solar_irradiance'][t] = random.uniform(800, 1000)
            elif weather_type == "cloudy":
                self.weather_data['temperature'][t] = random.uniform(15, 25)
                self.weather_data['solar_irradiance'][t] = random.uniform(300, 600)
            elif weather_type == "rainy":
                self.weather_data['temperature'][t] = random.uniform(10, 20)
                self.weather_data['solar_irradiance'][t] = random.uniform(100, 300)
            else:  # snowy
                self.weather_data['temperature'][t] = random.uniform(-5, 5)
                self.weather_data['solar_irradiance'][t] = random.uniform(50, 200)
                
        logger.info("success generate weather data")
    
    def get_current_weather(self) -> Dict:
        """
        get current time step weather data
        
        Returns:
            Dict: current weather data
        """
        return {
            'weather': self.weather_data['weather'][self.current_step],
            'temperature': self.weather_data['temperature'][self.current_step],
            'solar_irradiance': self.weather_data['solar_irradiance'][self.current_step]
        }
    
    def get_weather_impact(self, energy_type: str) -> float:
        """
        get weather impact coefficient
        
        Args:
            energy_type: energy type (solar_pv, wind_turbine, etc.)
            
        Returns:
            float: impact coefficient
        """
        current_weather = self.weather_data['weather'][self.current_step]
        
        if energy_type == "solar_pv":
            # solar power generation efficiency
            if current_weather == "sunny":
                return 1.0
            elif current_weather == "cloudy":
                return 0.6
            elif current_weather == "rainy":
                return 0.2
            else:  # snowy
                return 0.1
        else:
            return 1.0  # default not affected by weather
    
    def step(self):
        """update current time step"""
        self.current_step = (self.current_step + 1) % self.time_horizon
        
    def save_weather_data(self, filename: str):
        """
        save weather data to file
        
        Args:
            filename: file name
        """
        df = pd.DataFrame(self.weather_data)
        df.to_csv(filename, index=False)
        logger.info(f"weather data saved to {filename}")

class DemandModel:
    """energy demand model"""
    
    def __init__(self, demand_data_file: Optional[str] = None, time_horizon: int = 24):
        """
        initialize demand model
        
        Args:
            demand_data_file: demand data file path
            time_horizon: time range
        """
        self.time_horizon = time_horizon
        self.current_step = 0
        
        # demand data
        self.demand_data = {
            'total_demand': np.zeros(time_horizon),
            'predicted_demand': np.zeros(time_horizon)
        }
        
        # load demand data from file or generate demand data
        if demand_data_file and os.path.exists(demand_data_file):
            self.load_demand_data(demand_data_file)
        else:
            self.generate_demand_data()
    
    def load_demand_data(self, demand_data_file: str):
        """
        load demand data from file
        
        Args:
            demand_data_file: demand data file path
        """
        try:
            df = pd.read_csv(demand_data_file)
            
            # check if columns exist
            if 'demand' in df.columns:
                demand_values = df['demand'].values[:self.time_horizon]
                self.demand_data['total_demand'] = np.array(demand_values, dtype=np.float64)
                # add some random noise as prediction error
                noise = np.random.normal(0, 0.05 * np.mean(self.demand_data['total_demand']), self.time_horizon)
                self.demand_data['predicted_demand'] = self.demand_data['total_demand'] + noise
                
            logger.info(f"success load demand data from {demand_data_file}")
        except Exception as e:
            logger.error(f"failed to load demand data: {e}")
            self.generate_demand_data()
    
    def generate_demand_data(self):

        base_demand = np.array([
            200, 150, 120, 100, 100, 150,  # 0:00 - 5:00
            250, 350, 400, 380, 360, 380,  # 6:00 - 11:00
            400, 380, 350, 330, 350, 400,  # 12:00 - 17:00
            450, 500, 450, 400, 300, 250   # 18:00 - 23:00
        ])[:self.time_horizon]
        
        # add random noise
        noise = np.random.normal(0, 20, self.time_horizon)
        self.demand_data['total_demand'] = base_demand + noise
        
        # add larger noise to prediction value
        prediction_noise = np.random.normal(0, 40, self.time_horizon)
        self.demand_data['predicted_demand'] = base_demand + prediction_noise
        
        logger.info("success generate demand data")
    
    def get_current_demand(self) -> float:
        """
        get current time step demand
        
        Returns:
            float: current demand
        """
        return self.demand_data['total_demand'][self.current_step]
    
    def get_predicted_demand(self, steps_ahead: int = 1) -> float:
        """
        get future time step predicted demand
        
        Args:
            steps_ahead: number of time steps ahead
            
        Returns:
            float: predicted demand
        """
        future_step = (self.current_step + steps_ahead) % self.time_horizon
        return self.demand_data['predicted_demand'][future_step]
    
    def step(self):
        """update current time step"""
        self.current_step = (self.current_step + 1) % self.time_horizon
        
    def save_demand_data(self, filename: str):
        """
        save demand data to file
        
        Args:
            filename: file name
        """
        df = pd.DataFrame({
            'hour': range(self.time_horizon),
            'demand': self.demand_data['total_demand'],
            'predicted': self.demand_data['predicted_demand']
        })
        df.to_csv(filename, index=False)
        logger.info(f"demand data saved to {filename}")

# data structure definition
@dataclass
class Bid:
    """bid/offer data structure"""
    bid_id: str
    participant_id: str
    bid_type: str = "fixed"  # fixed, block, curve
    price: float = 0.0       # dkk/kWh
    quantity: float = 0.0    # kWh
    time_step: int = 0
    side: str = "buy"        # buy, sell
    priority: int = 3        # priority 1-5
    is_flexible: bool = True
    min_quantity: float = 0.0
    max_quantity: float = 0.0
    created_time: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if self.max_quantity == 0.0:
            self.max_quantity = self.quantity
        if self.min_quantity == 0.0:
            self.min_quantity = min(self.quantity * 0.1, 1.0)
    
    def to_dict(self) -> Dict:
        """convert to dictionary"""
        return {
            'bid_id': self.bid_id,
            'participant_id': self.participant_id,
            'bid_type': self.bid_type,
            'price': self.price,
            'quantity': self.quantity,
            'time_step': self.time_step,
            'side': self.side,
            'priority': self.priority,
            'is_flexible': self.is_flexible,
            'min_quantity': self.min_quantity,
            'max_quantity': self.max_quantity,
            'created_time': self.created_time
        }

@dataclass
class ClearingResult:
    """clearing result data structure"""
    clearing_id: str
    clearing_price: float
    clearing_quantity: float
    matched_bids: List[Tuple[str, float]]  # (bid_id, matched_quantity)
    clearing_time: datetime = field(default_factory=datetime.now)
    clearing_method: str = "uniform_price"  # uniform_price, pay_as_bid, lmp
    market_efficiency: float = 0.0
    total_welfare: float = 0.0
    matched_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """convert to dictionary"""
        return {
            'clearing_id': self.clearing_id,
            'clearing_price': self.clearing_price,
            'clearing_quantity': self.clearing_quantity,
            'matched_bids': self.matched_bids,
            'matched_pairs': self.matched_pairs,
            'clearing_time': self.clearing_time,
            'clearing_method': self.clearing_method,
            'market_efficiency': self.market_efficiency,
            'total_welfare': self.total_welfare
        }

@dataclass
class Trade:
    """trade record"""
    trade_id: str
    buyer_id: str
    seller_id: str
    energy_type: str
    quantity: float
    price: float
    time_step: int
    trade_time: Optional[datetime] = None
    status: str = "pending"  # pending, completed, cancelled
    clearing_result_id: Optional[str] = None
    bid_id: Optional[str] = None
    
    def __post_init__(self):
        if self.trade_time is None:
            self.trade_time = datetime.now()
    
    def to_dict(self) -> Dict:
        """convert to dictionary"""
        return {
            'trade_id': self.trade_id,
            'buyer_id': self.buyer_id,
            'seller_id': self.seller_id,
            'energy_type': self.energy_type,
            'quantity': self.quantity,
            'price': self.price,
            'time_step': self.time_step,
            'trade_time': self.trade_time,
            'status': self.status,
            'clearing_result_id': self.clearing_result_id,
            'bid_id': self.bid_id
        }

# abstract trading algorithm base class
class TradingAlgorithm(ABC):
    """abstract trading algorithm base class"""
    
    def __init__(self, algorithm_name: str):

        self.algorithm_name = algorithm_name
        self.logger = logging.getLogger(f"TradingAlgorithm.{algorithm_name}")
    
    @abstractmethod
    def process_bids(self, bids: List[Bid]) -> List[ClearingResult]:
        """
        process bid list
        
        Args:
            bids: bid list
            
        Returns:
            List[ClearingResult]: clearing result list
        """
        pass
    
    @abstractmethod
    def generate_trades(self, clearing_results: List[ClearingResult], 
                       bids: List[Bid]) -> List[Trade]:
        """
        generate trades based on clearing results
        
        Args:
            clearing_results: clearing result list
            bids: original bid list
            
        Returns:
            List[Trade]: trade list
        """
        pass
    
    def validate_bids(self, bids: List[Bid]) -> List[Bid]:
        """
        validate bid validity
        
        Args:
            bids: bid list
            
        Returns:
            List[Bid]: valid bid list
        """
        valid_bids = []
        for bid in bids:
            if self._is_valid_bid(bid):
                valid_bids.append(bid)
            else:
                self.logger.warning(f"invalid bid: {bid.bid_id}")
        return valid_bids
    
    def _is_valid_bid(self, bid: Bid) -> bool:
        """
        check if a single bid is valid
        
        Args:
            bid: bid
            
        Returns:
            bool: whether valid
        """
        if bid.price < 0:
            return False
        if bid.quantity <= 0:
            return False
        if bid.min_quantity > bid.max_quantity:
            return False
        return True
    
    def calculate_market_metrics(self, clearing_results: List[ClearingResult]) -> Dict:
        """
        calculate market metrics
        
        Args:
            clearing_results: clearing result list
            
        Returns:
            Dict: market metrics
        """
        if not clearing_results:
            return {}
        
        total_quantity = sum(cr.clearing_quantity for cr in clearing_results)
        avg_price = sum(cr.clearing_price * cr.clearing_quantity for cr in clearing_results) / total_quantity if total_quantity > 0 else 0
        
        return {
            'total_quantity': total_quantity,
            'average_price': avg_price,
            'num_clearings': len(clearing_results),
            'total_welfare': sum(cr.total_welfare for cr in clearing_results)
        }

# Bidding算法实现
class BiddingAlgorithm(TradingAlgorithm):
    """
    bidding algorithm implementation
    """
    
    def __init__(self):
        super().__init__("bidding")
        self.collected_bids: Dict[str, List[Bid]] = {}  # bids organized by time step
        self.participants: Dict[str, Dict] = {}  # participant information
    
    def register_participant(self, participant_id: str, participant_info: Dict):
        """
        register market participant
        
        Args:
            participant_id: participant ID
            participant_info: participant information
        """
        self.participants[participant_id] = participant_info
        self.logger.info(f"participant {participant_id} registered")
    
    def submit_bid(self, bid: Bid) -> bool:
        """
        submit bid
        
        Args:
            bid: bid object
            
        Returns:
            bool: whether successful submission
        """
        if not self._is_valid_bid(bid):
            self.logger.warning(f"invalid bid: {bid.bid_id}")
            return False
        
        time_step_key = str(bid.time_step)
        if time_step_key not in self.collected_bids:
            self.collected_bids[time_step_key] = []
        
        self.collected_bids[time_step_key].append(bid)
        self.logger.info(f"received bid: {bid.bid_id}, participant: {bid.participant_id}, "
                        f"type: {bid.side}, price: {bid.price}, quantity: {bid.quantity}")
        return True
    
    def process_bids(self, bids: List[Bid]) -> List[ClearingResult]:

        # validate bids
        valid_bids = self.validate_bids(bids)
        
        # group bids by time step and type
        buy_bids = [bid for bid in valid_bids if bid.side == "buy"]
        sell_bids = [bid for bid in valid_bids if bid.side == "sell"]
        
        # sort bids by price
        buy_bids.sort(key=lambda x: x.price, reverse=True)  # buy bids from high to low
        sell_bids.sort(key=lambda x: x.price)  # sell bids from low to high
        
        self.logger.info(f"processed {len(buy_bids)} buy bids, {len(sell_bids)} sell bids")
        
        return []
    
    def generate_trades(self, clearing_results: List[ClearingResult], 
                       bids: List[Bid]) -> List[Trade]:

        return []
    
    def get_bids_by_timestep(self, time_step: int) -> List[Bid]:
        """
        get all bids by time step
        
        Args:
            time_step: time step
            
        Returns:
            List[Bid]: bid list
        """
        time_step_key = str(time_step)
        return self.collected_bids.get(time_step_key, [])
    
    def get_market_summary(self, time_step: int) -> Dict:
        """
        get market summary
        
        Args:
            time_step: time step
            
        Returns:
            Dict: market summary
        """
        bids = self.get_bids_by_timestep(time_step)
        buy_bids = [bid for bid in bids if bid.side == "buy"]
        sell_bids = [bid for bid in bids if bid.side == "sell"]
        
        buy_quantity = sum(bid.quantity for bid in buy_bids)
        sell_quantity = sum(bid.quantity for bid in sell_bids)
        
        return {
            'total_bids': len(bids),
            'buy_bids': len(buy_bids),
            'sell_bids': len(sell_bids),
            'buy_quantity': buy_quantity,
            'sell_quantity': sell_quantity,
            'demand_supply_ratio': buy_quantity / sell_quantity if sell_quantity > 0 else float('inf')
        }

# Market Clearing algorithm implementation
class MarketClearingAlgorithm(TradingAlgorithm):

    def __init__(self, clearing_method: str = "uniform_price"):
        super().__init__("market_clearing")
        self.clearing_method = clearing_method  # uniform_price, pay_as_bid, lmp
        self.clearing_history: List[ClearingResult] = []
    
    def process_bids(self, bids: List[Bid]) -> List[ClearingResult]:
        """
        process bid list, execute market clearing
        
        Args:
            bids: bid list
            
        Returns:
            List[ClearingResult]: clearing result list
        """
        # validate bids
        valid_bids = self.validate_bids(bids)
        
        if not valid_bids:
            self.logger.warning("no valid bids, cannot execute clearing")
            return []
        
        # group bids by time step and execute clearing
        bids_by_timestep = {}
        for bid in valid_bids:
            time_step = bid.time_step
            if time_step not in bids_by_timestep:
                bids_by_timestep[time_step] = []
            bids_by_timestep[time_step].append(bid)
        
        clearing_results = []
        for time_step, step_bids in bids_by_timestep.items():
            result = self._clear_market_for_timestep(step_bids, time_step)
            if result:
                clearing_results.append(result)
        
        self.clearing_history.extend(clearing_results)
        return clearing_results
    
    def _clear_market_for_timestep(self, bids: List[Bid], time_step: int) -> Optional[ClearingResult]:
        """
        execute market clearing for a single time step
        
        Args:
            bids: bid list for the time step
            time_step: time step
            
        Returns:
            Optional[ClearingResult]: clearing result
        """
        # separate buy and sell bids
        buy_bids = [bid for bid in bids if bid.side == "buy"]
        sell_bids = [bid for bid in bids if bid.side == "sell"]
        
        if not buy_bids or not sell_bids:
            self.logger.warning(f"time step {time_step}: missing buy or sell bids")
            return None
        
        # sort bids by price
        buy_bids.sort(key=lambda x: x.price, reverse=True)  # buy bids from high to low
        sell_bids.sort(key=lambda x: x.price)  # sell bids from low to high
        
        # find clearing point
        clearing_price, clearing_quantity, matched_bids = self._find_clearing_point(buy_bids, sell_bids)
        
        if clearing_quantity == 0:
            self.logger.warning(f"time step {time_step}: cannot find clearing point")
            return None
        
        # calculate market welfare
        total_welfare = sum(q * (buyer.price - seller.price)
                            for buyer, seller, q in self._last_matched_pairs)
        
        # create clearing result
        clearing_result = ClearingResult(
            clearing_id=f"clearing_{time_step}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            clearing_price=clearing_price,
            clearing_quantity=clearing_quantity,
            matched_bids=matched_bids,
            matched_pairs=[(b.bid_id, s.bid_id, q) for b, s, q in self._last_matched_pairs],
            clearing_method=self.clearing_method,
            total_welfare=total_welfare
        )
        
        self.logger.info(f"time step {time_step} clearing completed: price {clearing_price:.4f}, quantity {clearing_quantity:.2f}")
        return clearing_result
    
    def _find_clearing_point(self, buy_bids: List[Bid], sell_bids: List[Bid]) -> Tuple[float, float, List[Tuple[str, float]]]:
        """Greedy price-compatible matching with exact per-bid energy budgets.

        The returned quantities record one allocation per bid. No transaction is
        created when prices do not cross or only self-matching is possible.
        """
        self._last_matched_pairs = []
        remaining_buy = {bid.bid_id: float(bid.quantity) for bid in buy_bids}
        remaining_sell = {bid.bid_id: float(bid.quantity) for bid in sell_bids}
        allocated = {}
        matched_pairs = []
        highest_ask = float("-inf")
        lowest_bid = float("inf")
        for buyer in sorted(buy_bids, key=lambda b: b.price, reverse=True):
            for seller in sorted(sell_bids, key=lambda b: b.price):
                if buyer.price < seller.price:
                    break
                if buyer.participant_id == seller.participant_id:
                    continue
                if max(highest_ask, seller.price) > min(lowest_bid, buyer.price):
                    continue
                quantity = min(remaining_buy[buyer.bid_id],
                               remaining_sell[seller.bid_id])
                if quantity <= 1e-12:
                    continue
                remaining_buy[buyer.bid_id] -= quantity
                remaining_sell[seller.bid_id] -= quantity
                allocated[buyer.bid_id] = allocated.get(buyer.bid_id, 0.0) + quantity
                allocated[seller.bid_id] = allocated.get(seller.bid_id, 0.0) + quantity
                matched_pairs.append((buyer, seller, quantity))
                highest_ask = max(highest_ask, seller.price)
                lowest_bid = min(lowest_bid, buyer.price)
                if remaining_buy[buyer.bid_id] <= 1e-12:
                    break

        if not matched_pairs:
            return 0.0, 0.0, []

        self._last_matched_pairs = matched_pairs
        quantity = sum(q for _, _, q in matched_pairs)
        # Price is between the marginal bid and ask of the executed pairs.
        lower_price = max(seller.price for _, seller, _ in matched_pairs)
        upper_price = min(buyer.price for buyer, _, _ in matched_pairs)
        clearing_price = (lower_price + upper_price) / 2.0
        matches = [(bid.bid_id, allocated[bid.bid_id])
                   for bid in buy_bids + sell_bids
                   if allocated.get(bid.bid_id, 0.0) > 1e-12]
        return clearing_price, quantity, matches

    def _match_bids(self, buy_bids: List[Bid], sell_bids: List[Bid], clearing_quantity: float) -> List[Tuple[str, float]]:
        """
        match bids
        
        Args:
            buy_bids: buy bid list
            sell_bids: sell bid list
            clearing_quantity: clearing quantity
            
        Returns:
            List[Tuple[str, float]]: matched bid list (bid_id, matched_quantity)
        """
        matched_bids = []
        remaining_quantity = clearing_quantity
        
        # match buy bids first
        for bid in buy_bids:
            if remaining_quantity <= 0:
                break
            matched_quantity = min(bid.quantity, remaining_quantity)
            matched_bids.append((bid.bid_id, matched_quantity))
            remaining_quantity -= matched_quantity
        
        # match sell bids
        remaining_quantity = clearing_quantity
        for bid in sell_bids:
            if remaining_quantity <= 0:
                break
            matched_quantity = min(bid.quantity, remaining_quantity)
            matched_bids.append((bid.bid_id, matched_quantity))
            remaining_quantity -= matched_quantity
        
        return matched_bids
    
    def _calculate_welfare(self, buy_bids: List[Bid], sell_bids: List[Bid], 
                          clearing_price: float, clearing_quantity: float) -> float:
        """
        calculate market welfare
        
        Args:
            buy_bids: buy bid list
            sell_bids: sell bid list
            clearing_price: clearing price
            clearing_quantity: clearing quantity
            
        Returns:
            float: total welfare
        """
        # consumer surplus: buy price - actual payment price
        consumer_surplus = 0.0
        remaining_quantity = clearing_quantity
        
        for bid in buy_bids:
            if remaining_quantity <= 0:
                break
            matched_quantity = min(bid.quantity, remaining_quantity)
            consumer_surplus += matched_quantity * (bid.price - clearing_price)
            remaining_quantity -= matched_quantity
        
        # producer surplus: actual received price - sell price
        producer_surplus = 0.0
        remaining_quantity = clearing_quantity
        
        for bid in sell_bids:
            if remaining_quantity <= 0:
                break
            matched_quantity = min(bid.quantity, remaining_quantity)
            producer_surplus += matched_quantity * (clearing_price - bid.price)
            remaining_quantity -= matched_quantity
        
        return consumer_surplus + producer_surplus
    
    def generate_trades(self, clearing_results: List[ClearingResult], 
                       bids: List[Bid]) -> List[Trade]:
        """
        generate trades based on clearing results
        
        Args:
            clearing_results: clearing result list
            bids: original bid list
            
        Returns:
            List[Trade]: trade list
        """
        self.logger.info(f"start generating trades: {len(clearing_results)} clearing results, {len(bids)} original bids")
        
        trades = []
        bid_dict = {bid.bid_id: bid for bid in bids}
        
        for i, clearing_result in enumerate(clearing_results):
            self.logger.info(f"process clearing result {i}: price {clearing_result.clearing_price:.4f}, quantity {clearing_result.clearing_quantity:.2f}, matched bids {len(clearing_result.matched_bids)}")
            
            # generate trades from matched bids
            buy_matches = []
            sell_matches = []
            
            for bid_id, matched_quantity in clearing_result.matched_bids:
                self.logger.info(f"process matched bid: {bid_id}, quantity {matched_quantity:.2f}")
                if bid_id in bid_dict:
                    bid = bid_dict[bid_id]
                    if bid.side == "buy":
                        buy_matches.append((bid, matched_quantity))
                        self.logger.info(f"add buy match: {bid.participant_id}, quantity {matched_quantity:.2f}")
                    else:
                        sell_matches.append((bid, matched_quantity))
                        self.logger.info(f"add sell match: {bid.participant_id}, quantity {matched_quantity:.2f}")
                else:
                    self.logger.warning(f"bid ID not found: {bid_id}")
            
            self.logger.info(f"buy matches: {len(buy_matches)}, sell matches: {len(sell_matches)}")
            
            # Use the exact buyer-seller allocations produced by clearing.
            exact_pairs = getattr(clearing_result, "matched_pairs", None)
            if exact_pairs:
                for trade_id_counter, (buyer_id, seller_id, quantity) in enumerate(exact_pairs):
                    buyer = bid_dict.get(buyer_id)
                    seller = bid_dict.get(seller_id)
                    if buyer is None or seller is None:
                        raise ValueError("clearing contains unknown bid")
                    if (buyer.side != "buy" or seller.side != "sell" or
                            buyer.participant_id == seller.participant_id or
                            buyer.price < clearing_result.clearing_price - 1e-10 or
                            seller.price > clearing_result.clearing_price + 1e-10):
                        raise ValueError("infeasible clearing pair")
                    trades.append(Trade(
                        trade_id=f"{clearing_result.clearing_id}_trade_{trade_id_counter}",
                        buyer_id=buyer.participant_id,
                        seller_id=seller.participant_id,
                        energy_type="electricity",
                        quantity=quantity,
                        price=clearing_result.clearing_price,
                        time_step=buyer.time_step,
                        status="completed",
                        clearing_result_id=clearing_result.clearing_id,
                        bid_id=f"{buyer.bid_id}_{seller.bid_id}",
                    ))
                continue

            # Legacy ClearingResult without explicit pairs.
            buy_remaining = [[bid, float(q)] for bid, q in buy_matches]
            sell_remaining = [[bid, float(q)] for bid, q in sell_matches]
            trade_id_counter = 0
            for buyer_item in buy_remaining:
                for seller_item in sell_remaining:
                    buyer = buyer_item[0]
                    seller = seller_item[0]
                    if (buyer.participant_id == seller.participant_id or
                            buyer.price < clearing_result.clearing_price - 1e-10 or
                            seller.price > clearing_result.clearing_price + 1e-10):
                        continue
                    trade_quantity = min(buyer_item[1], seller_item[1])
                    if trade_quantity <= 1e-12:
                        continue
                    trades.append(Trade(
                        trade_id=f"{clearing_result.clearing_id}_trade_{trade_id_counter}",
                        buyer_id=buyer.participant_id,
                        seller_id=seller.participant_id,
                        energy_type="electricity",
                        quantity=trade_quantity,
                        price=clearing_result.clearing_price,
                        time_step=buyer.time_step,
                        status="completed",
                        clearing_result_id=clearing_result.clearing_id,
                        bid_id=f"{buyer.bid_id}_{seller.bid_id}",
                    ))
                    trade_id_counter += 1
                    buyer_item[1] -= trade_quantity
                    seller_item[1] -= trade_quantity
                    if buyer_item[1] <= 1e-12:
                        break

        self.logger.info(f"trade generation completed: {len(trades)} trades")
        return trades

# trading algorithm factory
class TradingAlgorithmFactory:
    """trading algorithm factory"""
    
    _algorithms = {
        "bidding": BiddingAlgorithm,
        "market_clearing": MarketClearingAlgorithm
    }
    
    @classmethod
    def create_algorithm(cls, algorithm_name: str, **kwargs) -> TradingAlgorithm:
        """
        create trading algorithm instance
        
        Args:
            algorithm_name: algorithm name
            **kwargs: algorithm parameters
            
        Returns:
            TradingAlgorithm: algorithm instance
        """
        if algorithm_name not in cls._algorithms:
            raise ValueError(f"unknown trading algorithm: {algorithm_name}")
        
        algorithm_class = cls._algorithms[algorithm_name]
        return algorithm_class(**kwargs)
    
    @classmethod
    def register_algorithm(cls, algorithm_name: str, algorithm_class: type):
        """
        register new trading algorithm
        
        Args:
            algorithm_name: algorithm name
            algorithm_class: algorithm class
        """
        if not issubclass(algorithm_class, TradingAlgorithm):
            raise ValueError(f"algorithm class must inherit from TradingAlgorithm")
        
        cls._algorithms[algorithm_name] = algorithm_class
        logger.info(f"trading algorithm {algorithm_name} registered")
    
    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """
        get available trading algorithms
        
        Returns:
            List[str]: algorithm name list
        """
        return list(cls._algorithms.keys())

class TradingPool:
    """
    trading pool - support multiple trading algorithms
    
    main features:
    1. manage FlexOffer and bids
    2. support bidding and market clearing algorithms
    3. execute trades and record
    4. provide market analysis
    """
    
    def __init__(self, weather_model: WeatherModel, demand_model: DemandModel, 
                 trading_algorithm: str = "market_clearing", **algorithm_kwargs):
        """
        initialize trading pool
        
        Args:
            weather_model: weather model
            demand_model: demand model
            trading_algorithm: trading algorithm name
            **algorithm_kwargs: algorithm parameters
        """
        self.weather_model = weather_model
        self.demand_model = demand_model
        self.time_horizon = weather_model.time_horizon
        self.current_step = 0
        
        # trading algorithm
        self.trading_algorithm_name = trading_algorithm
        self.trading_algorithm = TradingAlgorithmFactory.create_algorithm(trading_algorithm, **algorithm_kwargs)
        
        # support multiple algorithms
        self.algorithms = {
            "bidding": TradingAlgorithmFactory.create_algorithm("bidding"),
            "market_clearing": self.trading_algorithm
        }
        
        # data storage
        self.managers: Dict[str, Manager] = {}
        self.participants: Dict[str, Dict] = {}
        self.bids: List[Bid] = []
        self.clearing_results: List[ClearingResult] = []
        self.trade_history: List[Trade] = []
        self.closed_timesteps: set = set()
        
        # keep original compatibility
        self.available_offers: Dict[str, Dict] = {}
        
        # price model
        self.grid_prices = np.random.uniform(0.1, 0.3, self.time_horizon)
        self.energy_prices = np.random.uniform(0.08, 0.25, self.time_horizon)
        
        logger.info(f"trading pool initialized, main algorithm: {trading_algorithm}")
    
    def add_manager(self, manager_id: str, manager: Manager):
        """
        add manager
        
        Args:
            manager_id: manager ID
            manager: manager object
        """
        self.managers[manager_id] = manager
        
        # register as trading participant
        participant_info = {
            'type': 'manager',
            'manager_object': manager,
            'registered_time': datetime.now()
        }
        self.participants[manager_id] = participant_info
        
        # register to bidding algorithm
        bidding_algo = self.algorithms.get("bidding")
        if bidding_algo and hasattr(bidding_algo, 'register_participant'):
            # safe call method
            getattr(bidding_algo, 'register_participant')(manager_id, participant_info)
        
        logger.info(f"manager {manager_id} added to trading pool")
    
    def create_bid_from_aggregated_fo(self, manager_id: str, aggregated_fo: AggregatedFlexOffer, 
                                     time_step: int, side: str = "sell", price: Optional[float] = None) -> Bid:
        """
        create bid from aggregated FlexOffer
        
        Args:
            manager_id: manager ID
            aggregated_fo: aggregated FlexOffer
            time_step: time step
            side: bid direction (buy/sell)
            price: bid price, if None then calculate automatically
            
        Returns:
            Bid: bid object
        """
        owner = getattr(aggregated_fo, "manager_id", None)
        if owner is not None and owner != manager_id:
            raise ValueError(f"aggregated FO manager {owner} differs from bid manager {manager_id}")

        # Positive grid energy is consumption (buy); negative is generation
        # or discharge (sell). Never invent a nominal 100-kWh offer.
        lo = float(aggregated_fo.total_energy_min)
        hi = float(aggregated_fo.total_energy_max)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
            raise ValueError("invalid aggregated energy bounds")
        if side not in ("buy", "sell"):
            raise ValueError(f"invalid bid side: {side}")
        total_energy = max(0.0, hi) if side == "buy" else max(0.0, -lo)
        if total_energy <= 0:
            raise ValueError(f"no {side}-side energy available in aggregated offer")

        if price is None:
            # calculate bid price based on grid price and demand prediction
            base_price = self.get_energy_price(time_step)
            demand_factor = self.demand_model.get_predicted_demand(time_step) / 100.0
            weather_impact = self.weather_model.get_weather_impact("solar_pv")
            
            random_factor = random.uniform(-0.25, 0.25)  
            
            # reduce market adjustment to make buy and sell prices closer
            market_adjustment = 0.0001 * (demand_factor - 0.5) + 0.00005 * (weather_impact - 0.5)
            

            if side == "sell":
                # sell price: base price * 0.9 - offset (lower sell price) + random fluctuation
                price = base_price * (0.9 - market_adjustment + random_factor)
            else:  # buy
                # buy price: base price * 1.1 + offset (increase buy price) + random fluctuation
                price = base_price * (1.1 + market_adjustment + random_factor)
            
            # ensure price is in reasonable range
            price = max(0.01, min(price, 2.0))  # price limit between 0.01 and 2.0
            

            if hasattr(self, 'manager_prices') and manager_id in self.manager_prices:
                prev_price = self.manager_prices.get(manager_id, {}).get(side, None)
                other_side = "buy" if side == "sell" else "sell"
                other_price = self.manager_prices.get(manager_id, {}).get(other_side, None)
                
                if prev_price is None and other_price is not None:
                    # buy and sell prices overlap
                    if side == "sell" and other_price is not None:
                        # sell price is lower than buy price
                        price = min(price, other_price * 0.9)
                    elif side == "buy" and other_price is not None:
                        # buy price is higher than sell price
                        price = max(price, other_price * 1.1)
            
            # buy and sell prices overlap for all managers
            if hasattr(self, 'manager_prices'):
                all_sell_prices = []
                all_buy_prices = []
                
                for m_id, prices in self.manager_prices.items():
                    if 'sell' in prices:
                        all_sell_prices.append(prices['sell'])
                    if 'buy' in prices:
                        all_buy_prices.append(prices['buy'])
                
                # if other manager's prices exist, price overlap
                if all_sell_prices and all_buy_prices:
                    avg_sell = sum(all_sell_prices) / len(all_sell_prices)
                    avg_buy = sum(all_buy_prices) / len(all_buy_prices)
                    
                    if side == "sell":
                        # ensure new sell price is not too high
                        price = min(price, avg_buy * 0.95)
                    else:  # buy
                        # ensure new buy price is not too low
                        price = max(price, avg_sell * 1.05)
            
            # record prices for later reference
            if not hasattr(self, 'manager_prices'):
                self.manager_prices = {}
            if manager_id not in self.manager_prices:
                self.manager_prices[manager_id] = {}
            self.manager_prices[manager_id][side] = price
        
        # Multiple device/AFO bids from the same manager are often created in
        # one second. A per-pool sequence prevents bid_id collisions, which
        # otherwise overwrite distinct bids during clearing-to-trade lookup.
        self._next_bid_sequence = getattr(self, '_next_bid_sequence', 0) + 1
        bid = Bid(
            bid_id=f"bid_{manager_id}_{side}_{time_step}_{self._next_bid_sequence}",
            participant_id=manager_id,
            price=price,
            quantity=total_energy,
            time_step=time_step,
            side=side,
            is_flexible=True,
            min_quantity=total_energy * 0.1,
            max_quantity=total_energy
        )
        
        # add log to show bid details
        logger.info(f"create {side} bid for {manager_id}: price={price:.4f}, quantity={total_energy:.2f}")
        
        return bid
    
    def submit_bid(self, bid: Bid) -> bool:
        """
        submit bid
        
        Args:
            bid: bid object
            
        Returns:
            bool: whether bid is successfully submitted
        """
        if bid.time_step in getattr(self, "closed_timesteps", set()):
            logger.warning(f"Rejecting bid {bid.bid_id}: timestep {bid.time_step} is closed")
            return False

        # safe call bidding algorithm method
        bidding_algo = self.algorithms.get("bidding")
        if bidding_algo and hasattr(bidding_algo, 'submit_bid'):
            success = getattr(bidding_algo, 'submit_bid')(bid)
            if success:
                self.bids.append(bid)
                logger.info(f"bid submitted successfully: {bid.bid_id}")
            return success
        else:
            # if no bidding algorithm, add bid directly to list
            self.bids.append(bid)
            logger.info(f"bid added directly: {bid.bid_id}")
            return True
    
    def execute_trading_round(self, time_step: int) -> Dict:
        """
        execute one trading round
        
        Args:
            time_step: time step
            
        Returns:
            Dict: trading result
        """
        if time_step in getattr(self, "closed_timesteps", set()):
            logger.info(f"Timestep {time_step} already settled")
            return {'trades': [], 'clearing_results': []}

        # get current time step bids
        current_bids = [bid for bid in self.bids if bid.time_step == time_step]
        
        if not current_bids:
            self.closed_timesteps = getattr(self, "closed_timesteps", set())
            self.closed_timesteps.add(time_step)
            logger.warning(f"time step {time_step}: no bids; round closed")
            return {'trades': [], 'clearing_results': []}
        
        # execute market clearing
        clearing_results = self.trading_algorithm.process_bids(current_bids)
        
        # generate trades
        trades = self.trading_algorithm.generate_trades(clearing_results, current_bids)
        
        # Each bid belongs to one settlement round. Retire all submitted bids
        # for this timestep, including unmatched ones, to prevent double clearing.
        # Leave bids for other timesteps untouched.
        self.bids = [bid for bid in self.bids if bid.time_step != time_step]
        self.closed_timesteps = getattr(self, "closed_timesteps", set())
        self.closed_timesteps.add(time_step)

        # record results
        self.clearing_results.extend(clearing_results)
        self.trade_history.extend(trades)
        
        logger.info(f"time step {time_step}: {len(trades)} trades completed")
        
        # get market summary
        market_summary = {}
        bidding_algo = self.algorithms.get("bidding")
        if bidding_algo and hasattr(bidding_algo, 'get_market_summary'):
            market_summary = getattr(bidding_algo, 'get_market_summary')(time_step)
        
        return {
            'trades': trades,
            'clearing_results': clearing_results,
            'market_summary': market_summary
        }
    
    # 保留原有兼容性方法
    def add_offer(self, manager_id: str, offer_id: str, offer_type: str, 
                 aggregated_result: AggregatedFlexOffer):
        """
        add offer (compatibility method)
        
        Args:
            manager_id: manager ID
            offer_id: Offer ID
            offer_type: offer type
            aggregated_result: aggregated result
        """
        self.available_offers[offer_id] = {
            'manager_id': manager_id,
            'offer_type': offer_type,
            'aggregated_result': aggregated_result,
            'status': 'available',
            'created_time': datetime.now()
        }
        
        # create bid at the same time
        bid = self.create_bid_from_aggregated_fo(manager_id, aggregated_result, self.current_step)
        self.submit_bid(bid)
    
    def execute_trade(self, buyer_id: str, seller_id: str, offer_id: str, 
                     quantity: float, price: float) -> Optional[Trade]:
        """
        execute trade (compatibility method)
        
        Args:
            buyer_id: buyer ID
            seller_id: seller ID
            offer_id: Offer ID
            quantity: trade quantity
            price: trade price
            
        Returns:
            Optional[Trade]: trade record
        """
        if offer_id not in self.available_offers:
            logger.warning(f"Offer ID {offer_id} does not exist")
            return None
        
        offer = self.available_offers[offer_id]
        if offer['status'] != 'available':
            logger.warning(f"Offer ID {offer_id} is not available, current status: {offer['status']}")
            return None
        
        # create trade record
        trade_id = f"trade_{len(self.trade_history)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        trade = Trade(
            trade_id=trade_id,
            buyer_id=buyer_id,
            seller_id=seller_id,
            energy_type=offer['offer_type'],
            quantity=quantity,
            price=price,
            time_step=self.current_step,
            status="completed"
        )
        
        # update Offer status
        self.available_offers[offer_id]['status'] = 'traded'
        
        # add trade record
        self.trade_history.append(trade)
        
        logger.info(f"trade completed: {trade_id}, buyer: {buyer_id}, seller: {seller_id}, " +
                   f"quantity: {quantity}, price: {price}")
        
        return trade
    
    def get_available_offers(self) -> Dict:
        """
        get available offers
        
        Returns:
            Dict: available offers
        """
        return {k: v for k, v in self.available_offers.items() if v['status'] == 'available'}
    
    def get_grid_price(self, time_step: Optional[int] = None) -> float:
        """
        get grid price
        
        Args:
            time_step: time step, if None then return current time step price
            
        Returns:
            float: grid price
        """
        if time_step is None:
            time_step = self.current_step
        
        return self.grid_prices[time_step]
    
    def get_energy_price(self, time_step: Optional[int] = None) -> float:
        """
        get energy price
        
        Args:
            time_step: time step, if None then return current time step price
            
        Returns:
            float: energy price
        """
        if time_step is None:
            time_step = self.current_step
        
        return self.energy_prices[time_step]
    
    def get_trade_statistics(self) -> Dict:
        """
        get trade statistics
        
        Returns:
            Dict: trade statistics
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'total_energy': 0.0,
                'total_value': 0.0,
                'avg_price': 0.0,
                'market_efficiency': 0.0
            }
        
        total_trades = len(self.trade_history)
        total_energy = sum(trade.quantity for trade in self.trade_history)
        total_value = sum(trade.quantity * trade.price for trade in self.trade_history)
        avg_price = total_value / total_energy if total_energy > 0 else 0.0
        
        # calculate market efficiency
        market_efficiency = sum(cr.market_efficiency for cr in self.clearing_results) / len(self.clearing_results) if self.clearing_results else 0.0
        
        return {
            'total_trades': total_trades,
            'total_energy': total_energy,
            'total_value': total_value,
            'avg_price': avg_price,
            'market_efficiency': market_efficiency,
            'clearing_results': len(self.clearing_results)
        }
    
    def step(self):
        """update current time step"""
        self.current_step = (self.current_step + 1) % self.time_horizon
        self.weather_model.step()
        self.demand_model.step()
        
        logger.info(f"trading pool time step updated to: {self.current_step}")
    
    def reset(self):
        """reset trading pool"""
        self.current_step = 0
        self.weather_model.current_step = 0
        self.demand_model.current_step = 0
        self.bids = []
        self.clearing_results = []
        self.trade_history = []
        self.available_offers = {}
        
        logger.info("trading pool reset")
    
    def visualize_trading_results(self, save_path: Optional[str] = None):
        """
        visualize trading results
        
        Args:
            save_path: save path, if None then show graph
        """
        if not self.trade_history:
            logger.info("no trade history")
            return
        
        # group trades by time step
        trades_by_step = {}
        for trade in self.trade_history:
            step = trade.time_step
            if step not in trades_by_step:
                trades_by_step[step] = []
            trades_by_step[step].append(trade)
        
        # calculate total quantity and average price for each time step
        steps = sorted(trades_by_step.keys())
        quantities = []
        prices = []
        
        for step in steps:
            step_trades = trades_by_step[step]
            total_quantity = sum(trade.quantity for trade in step_trades)
            avg_price = sum(trade.quantity * trade.price for trade in step_trades) / total_quantity
            
            quantities.append(total_quantity)
            prices.append(avg_price)
        
        # plot charts
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # trade quantity
        ax1.bar(steps, quantities, color='blue', alpha=0.7)
        ax1.set_title('trade quantity (by time step)')
        ax1.set_xlabel('time step')
        ax1.set_ylabel('trade quantity (kWh)')
        ax1.grid(True)
        
        # average price
        ax2.plot(steps, prices, color='red', marker='o')
        ax2.set_title('average price (by time step)')
        ax2.set_xlabel('time step')
        ax2.set_ylabel('price ($/kWh)')
        ax2.grid(True)
        
        # clearing results
        if self.clearing_results:
            clearing_prices = [cr.clearing_price for cr in self.clearing_results]
            clearing_quantities = [cr.clearing_quantity for cr in self.clearing_results]
            
            ax3.scatter(clearing_quantities, clearing_prices, color='green', alpha=0.7)
            ax3.set_title('clearing results (price vs quantity)')
            ax3.set_xlabel('clearing quantity (kWh)')
            ax3.set_ylabel('clearing price ($/kWh)')
            ax3.grid(True)
        
        # market welfare
        if self.clearing_results:
            welfare_values = [cr.total_welfare for cr in self.clearing_results]
            ax4.bar(range(len(welfare_values)), welfare_values, color='orange', alpha=0.7)
            ax4.set_title('market welfare')
            ax4.set_xlabel('clearing round')
            ax4.set_ylabel('total welfare')
            ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show() 