import torch
import torch.nn as nn
import torchvision.models as models

# import BiFPN class from submodule
import sys
sys.path.append('yaep/')
from yaep.efficientdet.model import BiFPN

class OccupancyNet(nn.Module):
    def __init__(self, backbone, embed_dim=64, grid_size=16):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.models = {"resnet" : ["resnet18", models.ResNet18_Weights, [128, 256, 512]], # we should be able to determine the conv_channels from embed_dim!
                        "regnet" : ["regnet_y_400mf", models.RegNet_Y_400MF_Weights, [104, 208, 440]], # we should be able to determine the conv_channels from embed_dim!
                        "efficientnet" : ["efficientnet_b0", models.EfficientNet_B0_Weights, [40, 112, 320]]} # we should be able to determine the conv_channels from embed_dim!
        self.left_feature_extractor = MultiscaleFeatureExtractor(self.models[self.backbone][0], self.models[self.backbone][1], self.embed_dim)
        self.right_feature_extractor = MultiscaleFeatureExtractor(self.models[self.backbone][0], self.models[self.backbone][1], self.embed_dim)
        self.left_bifpn = BiFPN(embed_dim, self.models[self.backbone][2], first_time=True)
        self.right_bifpn = BiFPN(embed_dim, self.models[self.backbone][2], first_time=True)
        self.attn_module = AttentionModule(embed_dim=self.embed_dim, grid_size=self.grid_size)
        self.deconvnet = DeconvNet(self.embed_dim, num_layers=2)

    def forward(self, left_image, right_image):
        # PATCHING AND EMBEDDING
        left_features = self.left_feature_extractor(left_image)
        right_features = self.right_feature_extractor(right_image)

        # BiFPN
        left_multiscale_features = self.left_bifpn(list(left_features.values()))
        right_multiscale_features = self.right_bifpn(list(right_features.values()))

        # Flatten each tensor and swap channel and sequence axes
        left_features_flat = []
        right_features_flat = []
        keys = ["p3_out", "p4_out", "p5_out", "p6_out", "p7_out"]
        for key, value in zip(keys, left_multiscale_features):
            value = value.view(value.size(0), value.size(1), -1)
            left_features_flat.append(value.permute(0, 2, 1))
        for key, value in zip(keys, right_multiscale_features):
            value = value.view(value.size(0), value.size(1), -1)
            right_features_flat.append(value.permute(0, 2, 1))

        # Concatenate along axis=1, sequence
        left_features_flat = torch.cat(left_features_flat, dim=1)
        right_features_flat = torch.cat(right_features_flat, dim=1)

        # ENCODING
        fused_features = self.attn_module(left_features_flat, right_features_flat)

        # 3D DECONVOLUTIONS AND PREDICT OCCUPANCY
        fused_features = fused_features.view(-1, self.embed_dim, self.grid_size, self.grid_size, self.grid_size)  # reshaping to 3D
        output = self.deconvnet(fused_features)

        return output
    

class MultiscaleFeatureExtractor(nn.Module):
    def __init__(self, backbone, weights, num_channels=64):
        super().__init__()
        model = getattr(models, backbone)(weights=weights.DEFAULT)

        if backbone == "resnet18":
            feature_levels = {"layer2" : "p3",
                              "layer3" : "p4",
                              "layer4" : "p5"}
            last_layer_idx = -2
        elif backbone == "efficientnet_b0":
            feature_levels = {"features.3" : "p3",
                              "features.5" : "p4",
                              "features.7" : "p5"}
            last_layer_idx = -2
        elif backbone == "regnet_y_400mf":
            feature_levels = {"trunk_output.block2" : "p3",
                              "trunk_output.block3" : "p4",
                              "trunk_output.block4" : "p5"}
            last_layer_idx = -2
 
        self.features = {}

        # Register hooks for each layer
        for level, module in model.named_modules():
            if level in feature_levels.keys():
                module.register_forward_hook(self.hook_function(feature_levels[level]))

        # Keep layers except for last two layers
        self.feature_extractor = nn.Sequential(*list(model.children())[:last_layer_idx])

    def hook_function(self, level):
        def hook(model, input, output):
            self.features[level] = output
        return hook

    def forward(self, x):
        self.feature_extractor(x)
        return self.features
    

class AttentionModule(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8, mlp_dim=64, max_seq_length=2048, grid_size=16):
        super().__init__()

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(grid_size**3, embed_dim))

        # Class token
        self.class_tokens = nn.Parameter(torch.randn(1, embed_dim))  # 1 for one class token

        # 2D Random Positional Embedding
        # Create a parameter directory with the correct shape
        self.left_pos_embedding = nn.Parameter(torch.randn(max_seq_length, embed_dim))
        self.right_pos_embedding = nn.Parameter(torch.randn(max_seq_length, embed_dim))

        # 3D Random Positional Embedding
        # Create a parameter directly with the correct shape
        self.pos_embedding_3d = nn.Parameter(torch.randn(grid_size**3, embed_dim))

        # Transformer encoder
        self.transformer = TransformerEncoder(embed_dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim)

    def forward(self, left_features, right_features):

        batch_size, left_seq_length, _ = left_features.size()
        _, right_seq_length, _ = right_features.size()

        # Add positional embedding after class token
        left_features = left_features + self.left_pos_embedding[:left_seq_length].unsqueeze(0).expand(batch_size, -1, -1)
        right_features = right_features + self.right_pos_embedding[:right_seq_length].unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate features along the sequence axis
        concat_features = torch.cat((left_features, right_features), dim=1)
        seq_length = concat_features.size(1)

        # Fixed Queries
        fixed_query_length = self.queries.size(0)

        # Expand queries to match the batch size of concat_features
        pos_embedding_3d = self.pos_embedding_3d.unsqueeze(0).expand(batch_size, -1, -1)
        fixed_queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        fixed_queries = fixed_queries + pos_embedding_3d

        # Pad concat_features to match or exceed queries' length
        if concat_features.size(1) < fixed_query_length:
            pad_length = fixed_query_length - concat_features.size(1)
            concat_features = torch.nn.functional.pad(concat_features, (0, 0, 0, pad_length))

        # Create mask for padded positions
        # mask shape should be[batch_size, seq_length] where True indicates positions to be ignored
        mask = torch.zeros(concat_features.size(0), concat_features.size(1), dtype=torch.bool, device=concat_features.device)
        if concat_features.size(1) > seq_length:
            mask[:, seq_length:] = True

        # Transform features using fixed queries
        fused_features = self.transformer(concat_features, fixed_queries, mask)

        return fused_features


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8, mlp_dim=64):
        super().__init__()

        # Transformer layers
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.mlp = MLP(embed_dim=embed_dim, hidden_dim=mlp_dim)
    
    def forward(self, x, fixed_queries=None, mask=None):

        # Normalization steps before attention for stability in training
        x1 = self.norm_1(x)

        if fixed_queries != None:
            q = fixed_queries
        else:
            q = x1
        k = x1
        v = x1
            
        # Attention computation
        # Note: We use batch_first=True, so we don't need to permute keys and values
        attn_output, _ = self.mha(q, k, v, key_padding_mask=mask, need_weights=False)

        # Residual connection
        # We only need to pad concat_features if it's shorter than attn_output which shouldn't happen if we've already padded it
        x = attn_output + x

        # Apply layer norm after the residual connection
        x2 = self.norm_2(x)

        # Apply MLP to further transform features
        mlp_output = self.mlp(x2)

        # Residual connection
        x = mlp_output + x

        return x


class MLP(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class DeconvNet(nn.Module):
    def __init__(self, dim=64, num_layers=4):
        super(DeconvNet, self).__init__()
        
        # Store layers in a list
        self.deconvs = nn.ModuleList()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Initial channel count
        channels = dim
        
        for i in range(num_layers):
            # Determine the number of output channels for this layer
            out_channels = channels // 2 if i < num_layers - 1 else 1  # Last layer outputs 1 channel
            self.deconvs.append(nn.ConvTranspose3d(
                in_channels=channels, 
                out_channels=out_channels, 
                kernel_size=2, 
                stride=2
            ))
            channels = out_channels  # Update channel count for next layer

    def forward(self, x):
        for i, deconv in enumerate(self.deconvs):
            x = deconv(x)
            if i < len(self.deconvs) - 1:  # Apply ReLU to all but the last layer
                x = self.relu(x)
            else:  # Apply sigmoid to the last layer
                x = self.sigmoid(x)
        return x