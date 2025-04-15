import torch
import torch.nn as nn
import torch.nn.functional as F

class PlannerPolicy(nn.Module):
    def __init__(self, config, input_size, output_size):
        super(PlannerPolicy, self).__init__()
        
        planner_config = config.get("planner_agent_training", {})
        
        fc_hidden_sizes = planner_config.get("fc_hidden_sizes", [128, 128])
        
        self.lstm_hidden_size = planner_config.get("lstm_hidden_size", 128)
        self.lstm_num_layers = planner_config.get("lstm_num_layers", 1)
        
        post_lstm_hidden_sizes = planner_config.get("post_lstm_hidden_sizes", [128, 64])
        
        fc_layers = []
        current_size = input_size
        
        for h_size in fc_hidden_sizes:
            fc_layers.append(nn.Linear(current_size, h_size))
            fc_layers.append(nn.ReLU())
            current_size = h_size
            
        self.fc_pre_lstm = nn.Sequential(*fc_layers)
        
        self.lstm = nn.LSTM(
            input_size=current_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True
        )
        
        post_fc_layers = []
        current_size = self.lstm_hidden_size
        
        for h_size in post_lstm_hidden_sizes:
            post_fc_layers.append(nn.Linear(current_size, h_size))
            post_fc_layers.append(nn.ReLU())
            current_size = h_size
            
        self.fc_post_lstm = nn.Sequential(*post_fc_layers)
        
        self.policy_head = nn.Linear(current_size, output_size)
        self.value_head = nn.Linear(current_size, 1)
        
    def forward(self, x, lstm_state=None):
        pre_lstm_features = self.fc_pre_lstm(x)
        
        if len(pre_lstm_features.shape) == 2:
            pre_lstm_features = pre_lstm_features.unsqueeze(1)
            
        batch_size = pre_lstm_features.shape[0]
        
        if lstm_state is None:
            h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, 
                            device=x.device)
            c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, 
                            device=x.device)
            lstm_state = (h0, c0)
        
        lstm_out, lstm_state = self.lstm(pre_lstm_features, lstm_state)
        
        if lstm_out.shape[1] == 1:
            lstm_out = lstm_out.squeeze(1)
        
        post_lstm_features = self.fc_post_lstm(lstm_out)
        
        tax_rates_logits = self.policy_head(post_lstm_features)
        value = self.value_head(post_lstm_features)
        
        return tax_rates_logits, value, lstm_state 