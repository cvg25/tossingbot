import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()  
        self.shortcut_path = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        )

        self.residual_path = nn.Sequential( # Pre-Activation Residual Blocks: https://arxiv.org/pdf/1603.05027
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.SiLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
        ) 

    def forward(self, x):
        return self.shortcut_path(x) + self.residual_path(x)

class PerceptionModel(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=64, out_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=128, out_channels=256),
            ResidualBlock(in_channels=256, out_channels=out_channels)
        )

    def forward(self, x):
        return self.feature_extractor(x)

class ActionModel(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.action_model = nn.Sequential(
            ResidualBlock(in_channels=in_channels, out_channels=256),
            ResidualBlock(in_channels=256, out_channels=128),
            nn.UpsamplingBilinear2d(scale_factor=2),
            ResidualBlock(in_channels=128, out_channels=64),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, stride=1, padding=2, padding_mode='reflect')
        )

    def forward(self, x):
        return self.action_model(x)

class FiLM_Layer(nn.Module):

    def __init__(self, in_features, out_features):
        """
        Feature-wise Linear Modulation: https://arxiv.org/abs/1709.07871
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.film_layer = nn.Linear(in_features=in_features, out_features=out_features * 2)
    
    def forward(self, x, condition):
        # Ensure condition has shape B x in_features
        condition = condition.view(-1, self.in_features)
        film_params = self.film_layer(condition) # B x out_features * 2
        weight, bias = film_params.chunk(2, dim=1) # Each B x out_features
        weight = weight.reshape(-1, self.out_features, 1, 1) # Reshape for broadcasting
        bias = bias.reshape(-1, self.out_features, 1, 1)
        x = weight * x + bias # Apply modulation to x
        return x

class TossingbotModel(nn.Module):

    def __init__(self, in_channels):
        """
        Paper: https://arxiv.org/pdf/1903.11239
        Section V. Network Architecture Details.
        """
        super().__init__()
        self.perception_model = PerceptionModel(in_channels=in_channels, out_channels=512)
        self.film_layer = FiLM_Layer(in_features=1, out_features=512)
        self.grasping_model = ActionModel(in_channels=256, out_channels=1)
        self.throwing_model = ActionModel(in_channels=256, out_channels=1)
        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, v):
        # Extract visual features
        x = self.perception_model(x)
        # Condition x on v
        x = self.film_layer(x, condition=v)
        # Compute grasping affordances
        q_grasp = self.grasping_model(x[:,:256])
        # Compute throwing residuals
        v_residual = self.throwing_model(x[:,256:])

        return q_grasp, v_residual