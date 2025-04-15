import random
import numpy as np
class TradingSystem:
    def __init__(self, config: dict, env):
        self.config = config
        self.bids = {
            "wood": [],
            "stone": []
        }
        self.asks = {
            "wood": [],
            "stone": []
        }

        self.env = env

        self.max_order_lifetime = config["max_order_lifetime"]

        self.num_trades = {
            "wood": 0,
            "stone": 0
        }

        self.last_price = {
            "wood": 0,
            "stone": 0
        }
            

    def make_order(
            self,
            agent_id: int,
            resource: str,
            price: int,
            transaction: str,
    ):
        if transaction == "bid":
            bid = {
                "agent_id": agent_id,
                "price": price,
                "lifetime": 0
            }
            self.bids[resource].append(bid)
        elif transaction == "ask":
            ask = {
                "agent_id": agent_id,
                "price": price,
                "lifetime": 0
            }
            self.asks[resource].append(ask)

    def satisfy_ask(self, resource, ask):
        agent_id = ask["agent_id"]
        price = ask["price"]
        
        agent = next((a for a in self.env.mobile_agents if a.agent_id == agent_id), None)
        if agent:
            agent.inventory["coins"] += price
            agent.escrow[resource] -= 1
            agent.active_orders -= 1
        
        self.asks[resource].remove(ask)

    def satisfy_bid(self, resource, bid, ask_price):
        agent_id = bid["agent_id"]
        
        agent = next((a for a in self.env.mobile_agents if a.agent_id == agent_id), None)
        if agent:
            agent.inventory[resource] += 1
            agent.escrow["coins"] -= ask_price
            agent.active_orders -= 1

            difference = bid["price"] - ask_price
            agent.escrow["coins"] -= difference
            agent.inventory["coins"] += difference
        
        self.bids[resource].remove(bid)

    
    def step(self):
        resources = ["wood", "stone"]

        for resource in resources:
            sorted_bids = sorted(self.bids[resource], key=lambda x: x["price"], reverse=True)
            sorted_asks = sorted(self.asks[resource], key=lambda x: x["price"])

            matched_bids = []
            matched_asks = []

            for bid in sorted_bids:
                bid_agent_id = bid["agent_id"]

                if bid in matched_bids:
                    continue
                
                for ask in sorted_asks:
                    ask_agent_id = ask["agent_id"]

                    if ask in matched_asks or bid_agent_id == ask_agent_id:
                        continue
                    
                    if bid["price"] >= ask["price"]:
                        matched_bids.append(bid)
                        matched_asks.append(ask)

                        self.satisfy_ask(resource, ask)
                        self.satisfy_bid(resource, bid, ask["price"])

                        self.num_trades[resource] += 1

                        if self.env.bank:   
                            self.env.bank.trades[resource].append((self.env.time, ask["price"]))

                        break
            
            if len(matched_asks) > 0:
                matched_prices = [ask["price"] for ask in matched_asks]
                self.last_price[resource] = np.mean(matched_prices)
        
        for resource in resources:
            expired_asks = []

            for ask in self.asks[resource]:
                ask["lifetime"] += 1

                if ask["lifetime"] > self.max_order_lifetime:
                    expired_asks.append(ask)
                    
            for ask in expired_asks:
                agent = next((a for a in self.env.mobile_agents if a.agent_id == ask["agent_id"]), None)
                if agent:
                    agent.inventory[resource] += 1
                    agent.escrow[resource] -= 1
                    agent.active_orders -= 1

                self.asks[resource].remove(ask)
            
            expired_bids = []

            for bid in self.bids[resource]:
                bid["lifetime"] += 1

                if bid["lifetime"] > self.max_order_lifetime:
                    expired_bids.append(bid)
                    
            for bid in expired_bids:
                agent = next((a for a in self.env.mobile_agents if a.agent_id == bid["agent_id"]), None)
                if agent:
                    agent.inventory["coins"] += bid["price"]
                    agent.escrow["coins"] -= bid["price"]
                    agent.active_orders -= 1 

                self.bids[resource].remove(bid)

    def reset_episode(self):
        self.bids = {
            "wood": [],
            "stone": []
        }
        self.asks = {
            "wood": [],
            "stone": []
        }

        self.num_trades = {
            "wood": 0,
            "stone": 0
        }

        self.last_price = {
            "wood": 0,
            "stone": 0
        }
                
        


