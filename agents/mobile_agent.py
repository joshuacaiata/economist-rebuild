from agents.base_agent import BaseAgent
import random
import numpy as np
import math
import torch
import torch.nn.functional as F
from training.mobile_agent_policy import MobileAgentPolicy
# Mobile Agent file

class MobileAgent(BaseAgent):
    def __init__(
            self, 
            agent_class, 
            agent_id, 
            config, 
            build_payout,
            risk_aversion,
            env
        ):
        super().__init__(agent_class, agent_id, config, env)
        self.position = None

        self.inventory = {
            "wood": 0,
            "stone": 0,
            "coins": config["starting_coins"]
        }

        self.build_payout = build_payout
        self.original_build_payout = build_payout
        self.labour = 0
        self.ending_coins_previous_year = 0
        self.escrow = {
            "wood": 0,
            "stone": 0,
            "coins": 0.0
        }

        self.action_range = 5 + 4 * (self.config.get("max_order_price", 10) + 1)

        self.houses_built = 0
        self.moves_made = 0
        self.no_ops_made = 0
        self.unsuccessful_moves = 0
        self.risk_aversion = risk_aversion
        self.max_num_orders = self.config.get("max_num_orders", 5)
        self.active_orders = 0

        self.util_step_0 = self.get_utility()

        self.rollout_buffer = []


    def get_observations(self, env):
        """
        Agents observation space:
        1. Neighbourhood (everything in the view_size radius)
        2. Inventory
        3. Build payout
        4. Taxation
            i. Planner tax schedule
            ii. Agents current tax bracket
            iii. Anonymized and sorted income distribution of all agents
        5. Time to next tax period
        6. Monetary policy
            i. Interest rate
            ii. Inflation rate
            iii. Money supply
            iv. Target inflation rate
        7. Position
        """

        row, col = self.position
        view_size = self.config["view_size"]

        local_view = {}

        for layer in ["Wood", "Stone", "Water", "Houses"]:
            padded_map = np.pad(
                env.map[layer],
                ((view_size, view_size), (view_size, view_size)),
                mode="constant",
                constant_values=0
            )

            padded_row = row + view_size
            padded_col = col + view_size

            local_view[layer] = padded_map[
                padded_row-view_size:padded_row+view_size+1,
                padded_col-view_size:padded_col+view_size+1
            ]
        
        agent_view = np.zeros((view_size*2+1, view_size*2+1))
        for agent in self.env.mobile_agents:
            # Calculate relative position within view
            rel_row = agent.position[0] - row + view_size
            rel_col = agent.position[1] - col + view_size
            
            # Only add agents that are within view range
            if 0 <= rel_row < 2*view_size+1 and 0 <= rel_col < 2*view_size+1:
                agent_view[rel_row, rel_col] = agent.agent_id

        local_view["Agents"] = agent_view

        # get income from all agents
        # this is just inventory["coins"] - ending_coins_previous_year
        incomes = [agent.inventory["coins"] - agent.ending_coins_previous_year for agent in self.env.mobile_agents]
        incomes = sorted(incomes)

        observation = {
            "position": self.position,
            "neighbourhood": local_view,
            "inventory": self.inventory,
            "build_payout": self.build_payout,
            "time_to_tax": self.env.time % self.config["tax_period_length"],
            "incomes": incomes,
        }

        # Only include tax-related info if a planner exists
        if self.env.has_planner and self.env.planner is not None:
            planner = self.env.planner
            observation["tax_rates"] = planner.tax_rates
            observation["tax_bracket"] = planner.get_tax_bracket(self.inventory["coins"])

        if self.env.has_bank and self.env.bank is not None:
            bank = self.env.bank
            # get the interest rate, inflation rate, money supply, target inflation rate
            observation["interest_rate"] = bank.interest_rate
            observation["inflation_rate"] = bank.inflation_rate
            observation["money_supply"] = bank.money_supply
            observation["target_inflation"] = bank.target_inflation

        return observation
    

    def flatten_observation(self, observation):
        channels = []

        for key in ["Wood", "Stone", "Water", "Houses", "Agents"]:
            layer = torch.FloatTensor(observation["neighbourhood"][key])
            channels.append(layer)

        neighbourhood_tensor = torch.stack(channels, dim=0)

        pos = np.array(observation["position"], dtype=np.float32)
        inv = np.array([observation["inventory"]["wood"],
                        observation["inventory"]["stone"],
                        observation["inventory"]["coins"]], dtype=np.float32)
        
        bp = np.array([observation["build_payout"]], dtype=np.float32)
        ttt = np.array([observation["time_to_tax"]], dtype=np.float32)
        incomes = np.array(observation["incomes"], dtype=np.float32)
        
        # Prepare basic features
        features = [pos, inv, bp, ttt, incomes]
        
        # Add tax-related features if they exist
        if "tax_rates" in observation and "tax_bracket" in observation:
            tax_rates = np.array(observation["tax_rates"], dtype=np.float32)
            tax_bracket = np.array([observation["tax_bracket"]], dtype=np.float32)
            features.extend([tax_rates, tax_bracket])

        if "interest_rate" in observation and "inflation_rate" in observation and "money_supply" in observation:
            interest_rate = np.array([observation["interest_rate"]], dtype=np.float32)
            inflation_rate = np.array([observation["inflation_rate"]], dtype=np.float32)
            money_supply = np.array([observation["money_supply"]], dtype=np.float32)
            target_inflation = np.array([observation["target_inflation"]], dtype=np.float32)
            features.extend([interest_rate, inflation_rate, money_supply, target_inflation]) 
        
        numeric = np.concatenate(features)
        numeric_tensor = torch.FloatTensor(numeric)
        
        return neighbourhood_tensor, numeric_tensor

    def get_action(self, neighbourhood, numeric, random_sampling=False):
        # Get action mask first since we need it for both cases
        action_mask = self.get_action_mask()
        
        if random_sampling:
            # Get indices of valid actions
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 0:
                return int(np.random.choice(valid_actions))
            return 0 
            
        # Ensure inputs are on the same device as the policy network
        device = next(self.policy_net.parameters()).device
        neighbourhood = neighbourhood.to(device)
        numeric = numeric.to(device)
        
        self.policy_net.eval()
        with torch.no_grad():
            logits, value, lstm_state = self.policy_net(neighbourhood, numeric)
            
            # Apply mask to logits
            action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=device) 
            action_mask_tensor = action_mask_tensor.unsqueeze(0)
            masked_logits = logits.clone()
            masked_logits[~action_mask_tensor] = -1e10
            
            # Calculate probabilities and sample
            probs = F.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action_tensor = dist.sample()
            
        return int(action_tensor.item())

    def step(self, action):
        """
        Action space:
        1. No-op (0)
        2. Move up (1), down (2), left (3), right (4)
        3. Build House (5)
        4. Buy Wood for price (6+min_order_price...6+max_order_price)
        5. Sell Wood for price (6+max_order_price+1...6+2*max_order_price)
        6. Buy Stone for price (6+2*max_order_price+1...6+3*max_order_price)
        7. Sell Stone for price (6+3*max_order_price+1...6+4*max_order_price)

        The agent can take any action number between 0 and 6 + 4 * max_order_price
        """
        
        # No operation
        if action == 0:
            self.labour += self.config["no_op_labour"]
            self.no_ops_made += 1
            return True
        
        # Movement actions
        elif 1 <= action <= 4:
            success = self._handle_movement(action)
            if success:
                self.moves_made += 1
            else:
                self.unsuccessful_moves += 1

        # Build house action
        elif action == 5:
            success = self._handle_building()
        
        # Trading actions
        elif 6 <= action <= self.action_range:
            success = self._handle_trading(action)
        
        else:
            raise ValueError(f"Invalid action: {action}. Max valid action is {self.action_range}")
        
        return success
        
    def _handle_movement(self, action):
        """Handle agent movement and resource collection."""
        # Define direction vectors for each movement action
        # [row_change, col_change]
        movement_vectors = {
            1: (-1, 0),
            2: (1, 0),
            3: (0, -1),
            4: (0, 1)
        }
        
        # Calculate new position
        row_delta, col_delta = movement_vectors[action]
        new_row = self.position[0] + row_delta
        new_col = self.position[1] + col_delta
        new_position = (new_row, new_col)
        
        # Check if movement is valid
        if self._is_valid_move(new_position):
            self.position = new_position
            self.labour += self.config["move_labour"]
            self._try_gather_resources()

            return True
        else:
            return False

    def _is_valid_move(self, new_position):
        """Check if a move to the given position is valid."""
        row, col = new_position
        
        # Check if within map boundaries
        if not (0 <= row < self.env.map_size[0] and 0 <= col < self.env.map_size[1]):
            return False
        
        # Check if water tile
        if self.env.map["Water"][row, col] == 1:
            return False
        
        # Check if another agent is at the position
        other_agent_positions = [agent.position for agent in self.env.mobile_agents 
                                if agent.agent_id != self.agent_id]
        if new_position in other_agent_positions:
            return False
        
        return True

    def _try_gather_resources(self):
        """Try to gather resources at the current position."""
        row, col = self.position
        
        if self.env.map["Wood"][row, col] > 0:
            if random.random() < self.config["gather_prob"]:
                self.labour += self.config["gather_labour"]
                self.inventory["wood"] += 1
                self.env.map["Wood"][row, col] = 0
        
        elif self.env.map["Stone"][row, col] > 0:
            if random.random() < self.config["gather_prob"]:
                self.labour += self.config["gather_labour"]
                self.inventory["stone"] += 1
                self.env.map["Stone"][row, col] = 0
 
        
        return False

    def can_build(self):
        return self.inventory["wood"] >= self.config["build_wood_cost"] and self.inventory["stone"] >= self.config["build_stone_cost"]

    def _handle_building(self):
        """Handle house building action."""
        row, col = self.position
        
        # Check if location is buildable
        if self.env.map["Buildable"][row, col] == 1 and self.can_build():
            # Update map
            self.env.map["Buildable"][row, col] = 0
            
            # If we have a Houses layer, update it
            if "Houses" in self.env.map:
                self.env.map["Houses"][row, col] = self.agent_id
            
            # Apply labor cost
            self.labour += self.config["build_labour"]
            
            # Update income and inventory
            self.inventory["coins"] += self.build_payout
            self.inventory["wood"] -= self.config["build_wood_cost"]
            self.inventory["stone"] -= self.config["build_stone_cost"]
            self.houses_built += 1

            return True
        else:
            return False

    def _handle_trading(self, action):
        """Handle trading actions."""
        if self.active_orders >= self.max_num_orders:
            return False

        max_order_price = self.config.get("max_order_price", 10)
        
        # Determine action category and price in one step
        trade_index = action - 6
        segment_size = max_order_price + 1

        resource_index = math.floor(trade_index / (2 * segment_size))

        resource = "wood" if resource_index == 0 else "stone"
        local_index = trade_index % (2 * segment_size)
        order_type = "bid" if local_index < segment_size else "ask"
        price = local_index if order_type == "bid" else local_index - segment_size

        if 0 <= price <= max_order_price:
            # If its a buy, we need to check if we have enough coins
            if order_type == "bid" and self.inventory["coins"] >= price:
                self.inventory["coins"] -= price
                self.escrow["coins"] += price

                self.env.trading_system.make_order(
                    self.agent_id,
                    resource,
                    price,
                    order_type
                )

                self.labour += self.config["trade_labour"]
                self.active_orders += 1 
                return True

            elif order_type == "ask" and self.inventory[resource] > 0:
                self.inventory[resource] -= 1
                self.escrow[resource] += 1

                self.env.trading_system.make_order(
                    self.agent_id,
                    resource,
                    price,
                    order_type
                )

                self.labour += self.config["trade_labour"]
                self.active_orders += 1 
                return True
            else:
                return False
        else:
            return False

    def get_utility(self):
        if self.risk_aversion == 1:
            return np.log(self.inventory["coins"])
        else:
            utility = (self.inventory["coins"] ** (1 - self.risk_aversion) - 1) / (1 - self.risk_aversion)
            return utility - self.labour
        
    def reset_year(self):
        self.ending_coins_previous_year = self.inventory["coins"]

    def reset_episode(self):
        self.inventory = {
            "wood": 0,
            "stone": 0,
            "coins": 0
        }

        self.escrow = {
            "wood": 0,
            "stone": 0,
            "coins": 0.0
        }

        self.labour = 0
        self.houses_built = 0
        self.ending_coins_previous_year = 0
        self.util_step_0 = self.get_utility()
        self.active_orders = 0
        self.build_payout = self.original_build_payout
        
    def get_action_mask(self):
        """
        Returns a boolean mask indicating which actions are valid for the current state.
        True means the action is valid, False means it's invalid.
        """
        mask = np.ones(self.action_range + 1, dtype=bool)  # Default all actions to valid
        
        # Check movement actions (1-4)
        movement_vectors = {
            1: (-1, 0),
            2: (1, 0),
            3: (0, -1),
            4: (0, 1)
        }
        
        for action in range(1, 5):
            row_delta, col_delta = movement_vectors[action]
            new_row = self.position[0] + row_delta
            new_col = self.position[1] + col_delta
            new_position = (new_row, new_col)
            mask[action] = self._is_valid_move(new_position)
        
        # Check building action (5)
        row, col = self.position
        can_build = (self.env.map["Buildable"][row, col] == 1 and 
                    self.inventory["wood"] >= self.config["build_wood_cost"] and 
                    self.inventory["stone"] >= self.config["build_stone_cost"])
        mask[5] = can_build
        
        # Check trading actions (6+)
        max_order_price = self.config.get("max_order_price", 10)
        segment_size = max_order_price + 1

        if self.active_orders >= self.max_num_orders:
            mask[6:] = False
        
        # Buy actions
        for price in range(segment_size):
            # Buy wood (not enough coins)
            if self.inventory["coins"] < price:
                mask[6 + price] = False
                
            # Buy stone (not enough coins)
            if self.inventory["coins"] < price:
                mask[6 + 2*segment_size + price] = False
        
        # Sell actions
        # Sell wood (no wood to sell)
        if self.inventory["wood"] <= 0:
            for price in range(segment_size):
                mask[6 + segment_size + price] = False
                
        # Sell stone (no stone to sell)
        if self.inventory["stone"] <= 0:
            for price in range(segment_size):
                mask[6 + 3*segment_size + price] = False
                
        return mask

    
        