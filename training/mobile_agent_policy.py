import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileAgentPolicy(nn.Module):
    def __init__(self, config, num_numeric, action_range):
        super(MobileAgentPolicy, self).__init__()

        view_size = config["view_size"]
        training_config = config["mobile_agent_training"]

        out_channels_list = training_config["cnn_layers"]["out_channels"]
        kernel_size_list = training_config["cnn_layers"]["kernel_size"]
        stride_list = training_config["cnn_layers"]["stride"]
        padding_list = training_config["cnn_layers"]["padding"]

        cnn_layers = []
        in_channels = 5

        for i in range(len(out_channels_list)):
            layer_cfg = {
                "out_channels": out_channels_list[i],
                "kernel_size": kernel_size_list[i],
                "stride": stride_list[i],
                "padding": padding_list[i]
            }

            cnn_layers.append(nn.Conv2d(in_channels,
                                        layer_cfg["out_channels"],
                                        kernel_size=layer_cfg["kernel_size"],
                                        stride=layer_cfg["stride"],
                                        padding=layer_cfg["padding"]))
            cnn_layers.append(nn.ReLU())
            in_channels = layer_cfg["out_channels"]
        cnn_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*cnn_layers)

        image_size = 2 * view_size + 1
        cnn_out_size = image_size
        for i in range(len(out_channels_list)):
            kernel_size = kernel_size_list[i]
            stride = stride_list[i]
            padding = padding_list[i]
            cnn_out_size = (cnn_out_size + 2 * padding - kernel_size) // stride + 1
        cnn_output_size = in_channels * (cnn_out_size ** 2)

        fc_numeric_hidden_sizes = training_config.get("fc_numeric_hidden_sizes", [])
        numeric_layers = []
        numeric_in = num_numeric
        for h in fc_numeric_hidden_sizes:
            numeric_layers.append(nn.Linear(numeric_in, h))
            numeric_layers.append(nn.ReLU())
            numeric_in = h
        self.fc_numeric = nn.Sequential(*numeric_layers)
        
        combined_in = cnn_output_size + numeric_in
        
        self.lstm_hidden_size = training_config.get("lstm_hidden_size", 256)
        self.lstm_num_layers = training_config.get("lstm_num_layers", 1)
        
        self.lstm = nn.LSTM(
            input_size=combined_in,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True
        )
        
        fc_combined_hidden_sizes = training_config.get("fc_combined_hidden_size", [])
        combined_layers = []
        combined_in = self.lstm_hidden_size 
        for h in fc_combined_hidden_sizes:
            combined_layers.append(nn.Linear(combined_in, h))
            combined_layers.append(nn.ReLU())
            combined_in = h
        self.fc_combined = nn.Sequential(*combined_layers)
        
        self.policy_head = nn.Linear(combined_in, action_range + 1)
        self.value_head = nn.Linear(combined_in, 1)
    
    def forward(self, neighbourhood, numeric, lstm_state=None):
        cnn_out = self.cnn(neighbourhood)
        num_out = self.fc_numeric(numeric)
        combined = torch.cat([cnn_out, num_out], dim=1)
        
        batch_size = combined.shape[0]
        lstm_input = combined.unsqueeze(1)
        
        if lstm_state is None:
            h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, 
                             device=neighbourhood.device)
            c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, 
                             device=neighbourhood.device)
            lstm_state = (h0, c0)
        
        lstm_out, lstm_state = self.lstm(lstm_input, lstm_state)
        
        lstm_out = lstm_out.squeeze(1) 
        
        features = self.fc_combined(lstm_out)
        
        logits = self.policy_head(features)
        value = self.value_head(features)
        
        return logits, value, lstm_state
    
