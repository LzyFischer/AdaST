import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from types import SimpleNamespace
from tqdm import tqdm
import pdb
import torch.nn.init as init



class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)], dim=1)
        else:
            x = x
        
        return x

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        if enable_padding == True:
            self.__padding = (kernel_size - 1) * dilation
        else:
            self.__padding = 0
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=self.__padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[: , : , : -self.__padding]
        
        return result

class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)

        return result

class TemporalConvLayer(nn.Module):

    # Temporal Convolution Layer (GLU)
    #
    #        |--------------------------------| * residual connection *
    #        |                                |
    #        |    |--->--- casualconv2d ----- + -------|       
    # -------|----|                                   ⊙ ------>
    #             |--->--- casualconv2d --- sigmoid ---|                               
    #
    
    #param x: tensor, [bs, c_in, ts, n_vertex]

    def __init__(self, Kt, c_in, c_out, n_vertex, act_func):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.align = Align(c_in, c_out)
        if act_func == 'glu' or act_func == 'gtu':
            self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=2 * c_out, kernel_size=(Kt, 1), enable_padding=False, dilation=1)
        else:
            self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=c_out, kernel_size=(Kt, 1), enable_padding=False, dilation=1)
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.act_func = act_func

    def forward(self, x):   
        x_in = self.align(x)[:, :, self.Kt - 1:, :]
        x_causal_conv = self.causal_conv(x)

        if self.act_func == 'glu' or self.act_func == 'gtu':
            x_p = x_causal_conv[:, : self.c_out, :, :]
            x_q = x_causal_conv[:, -self.c_out:, :, :]

            if self.act_func == 'glu':
                # Explanation of Gated Linear Units (GLU):
                # The concept of GLU was first introduced in the paper 
                # "Language Modeling with Gated Convolutional Networks". 
                # URL: https://arxiv.org/abs/1612.08083
                # In the GLU operation, the input tensor X is divided into two tensors, X_a and X_b, 
                # along a specific dimension.
                # In PyTorch, GLU is computed as the element-wise multiplication of X_a and sigmoid(X_b).
                # More information can be found here: https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.glu
                # The provided code snippet, (x_p + x_in) ⊙ sigmoid(x_q), is an example of GLU operation. 
                x = torch.mul((x_p + x_in), torch.sigmoid(x_q))

            else:
                # tanh(x_p + x_in) ⊙ sigmoid(x_q)
                x = torch.mul(torch.tanh(x_p + x_in), torch.sigmoid(x_q))

        elif self.act_func == 'relu':
            x = self.relu(x_causal_conv + x_in)

        elif self.act_func == 'silu':
            x = self.silu(x_causal_conv + x_in)
        
        else:
            raise NotImplementedError(f'ERROR: The activation function {self.act_func} is not implemented.')
        
        return x


class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, gso, bias):
        super(GraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        #bs, c_in, ts, n_vertex = x.shape
        x = torch.permute(x, (0, 2, 3, 1))

        first_mul = torch.einsum('hi,btij->bthj', self.gso, x)
        second_mul = torch.einsum('bthi,ij->bthj', first_mul, self.weight)

        if self.bias is not None:
            graph_conv = torch.add(second_mul, self.bias)
        else:
            graph_conv = second_mul
        
        return graph_conv

class GraphConvLayer(nn.Module):
    def __init__(self, graph_conv_type, c_in, c_out, Ks, gso, bias):
        super(GraphConvLayer, self).__init__()
        self.graph_conv_type = graph_conv_type
        self.c_in = c_in
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        self.Ks = Ks
        self.gso = gso
        if self.graph_conv_type == 'cheb_graph_conv':
            self.cheb_graph_conv = ChebGraphConv(c_out, c_out, Ks, gso, bias)
        elif self.graph_conv_type == 'graph_conv':
            self.graph_conv = GraphConv(c_out, c_out, gso, bias)

    def forward(self, x):
        x_gc_in = self.align(x)
        if self.graph_conv_type == 'cheb_graph_conv':
            x_gc = self.cheb_graph_conv(x_gc_in)
        elif self.graph_conv_type == 'graph_conv':
            x_gc = self.graph_conv(x_gc_in)
        x_gc = x_gc.permute(0, 3, 1, 2)
        x_gc_out = torch.add(x_gc, x_gc_in)

        return x_gc_out

class STConvBlock(nn.Module):
    # STConv Block contains 'TGTND' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv or GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels, act_func, graph_conv_type, gso, bias, droprate):
        super(STConvBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)
        self.graph_conv = GraphConvLayer(graph_conv_type, channels[0], channels[1], Ks, gso, bias)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex, act_func)
        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]], eps=1e-12)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.graph_conv(x)
        x = self.relu(x)
        x = self.tmp_conv2(x)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)

        return x

class OutputBlock(nn.Module):
    # Output block contains 'TNFF' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, Ko, last_block_channel, channels, end_channel, n_vertex, act_func, bias, droprate):
        super(OutputBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Ko, last_block_channel, channels[0], n_vertex, act_func)
        self.fc1 = nn.Linear(in_features=channels[0], out_features=channels[1], bias=bias)
        self.fc2 = nn.Linear(in_features=channels[1], out_features=end_channel, bias=bias)
        self.tc1_ln = nn.LayerNorm([n_vertex, channels[0]], eps=1e-12)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.tc1_ln(x.permute(0, 2, 3, 1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x).permute(0, 3, 1, 2)

        return x



class STGCNGraphConv(nn.Module):
    def __init__(self, args, blocks, n_vertex):
        super(STGCNGraphConv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(STConvBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], args.act_func, args.graph_conv_type, args.gso, args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.horizon - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.do = nn.Dropout(p=args.droprate)
        self.fc1 = nn.Linear(in_features=blocks[-3][-1]*self.Ko, out_features=blocks[-2][0], bias=args.enable_bias)
        self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=args.horizon, bias=args.enable_bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(1, 0, 2).unsqueeze(1)
        x = self.st_blocks(x)
        # if self.Ko > 1:
        #     x = self.output(x)
        # elif self.Ko == 0:
        #     x = self.fc1(x.permute(0, 2, 3, 1))
        #     x = self.relu(x)
        #     x = self.fc2(x).permute(0, 3, 1, 2)
        x = x.reshape(x.shape[0], -1, x.shape[-1]).permute(0, 2, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x).permute(2, 0, 1)

        return x

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import pickle
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv

# Load the saved graph
with open("location_graph.pkl", "rb") as f:
    G = pickle.load(f)

# Convert graph to adjacency matrix
adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)

# Load measurement data
measurement_data = pd.read_csv("measurement_data.csv", index_col=0)
measurement_data = measurement_data.T  # Transpose to have time as index

# Convert DataFrame to numpy array
data_values = measurement_data.values.astype(np.float32)

# Define train/val/test split
train_ratio, val_ratio, test_ratio = 0.7, 0.1, 0.2
train_size = int(len(data_values) * train_ratio)
val_size = int(len(data_values) * val_ratio)
test_size = len(data_values) - train_size - val_size

train_data, test_data = train_test_split(data_values, train_size=train_size+val_size, test_size=test_size, shuffle=False)
train_data, val_data = train_test_split(train_data, train_size=train_size, test_size=val_size, shuffle=False)

# Create dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size=12, horizon=12):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon

        self.samples = []
        for i in range(len(data) - window_size - horizon + 1):
            x = data[i:i+window_size]
            y = data[i+window_size:i+window_size+horizon]
            x = np.nan_to_num(x, nan=0.0)
            y = np.nan_to_num(y, nan=0.0)
            # if not np.isnan(x).any() and not np.isnan(y).any():  # Ignore missing values
            self.samples.append((x, y))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        output_mask = (y != 0.0).astype(np.float32)
        return torch.tensor(x), torch.tensor(y), torch.tensor(output_mask)

# Create datasets and dataloaders
train_dataset = TimeSeriesDataset(train_data)
val_dataset = TimeSeriesDataset(val_data)
test_dataset = TimeSeriesDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Train size:", len(train_dataset), "Val size:", len(val_dataset), "Test size:", len(test_dataset))


# Prepare adjacency matrix for PyTorch Geometric
edge_index = torch.tensor(np.array(list(nx.to_edgelist(G)))[:, :2].astype(int).T, dtype=torch.long)
num_nodes = adj_matrix.shape[0]

# Initialize model, optimizer, and loss function
Kt = 3  # Kernel size for temporal conv
c_in = 1  # Input channels (e.g., 1 for univariate time series)
c_out = 32  # Hidden layer channels
n_vertex = 55  # Number of nodes (e.g., sensors, locations)
num_timesteps = 12  # Input time steps
horizon = 12  # Prediction horizon
stblock_num = 2
Ko = horizon - (Kt - 1) * 2 * stblock_num
# blocks: settings of channel size in st_conv_blocks and output layer,
# using the bottleneck design in st_conv_blocks
blocks = []
blocks.append([1])
for l in range(stblock_num):
    blocks.append([64, 16, 64])
if Ko == 0:
    blocks.append([128])
elif Ko > 0:
    blocks.append([128, 128])
blocks.append([1])
args = {
    "Kt": Kt,
    "Ko": Ko,
    "Ks": 3,
    "c_in": c_in,
    "c_out": c_out,
    "n_vertex": n_vertex,
    "num_timesteps": num_timesteps,
    "horizon": horizon,
    "graph_conv_type": "graph_conv",
    "gso": adj_matrix.cuda(),
    "enable_bias": True,
    "droprate": 0.2,
    "act_func": "relu"
}
args = SimpleNamespace(**args)

model = STGCNGraphConv(args, blocks, n_vertex).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
def masked_loss_fn(y_pred, y, mask):
    loss = loss_fn(y_pred, y)
    loss = loss * mask  # Apply the mask to ignore missing values (0 values)
    return loss.sum() / mask.sum()



# Test the model
def test_model(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y, mask in test_loader:
            x, y = x.permute(1, 0, 2), y.permute(1, 0, 2)
            x, y = x.cuda(), y.cuda()  # Move to GPU
            mask = mask.cuda()  # Move to GPU
            y_pred = model(x)
            loss = masked_loss_fn(y_pred, y, mask)
            test_loss += loss.item()
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")
# Training loop
def train_model(model, train_loader, val_loader, epochs=50):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y, mask in tqdm(train_loader):
            x, y = x.permute(1, 0, 2), y.permute(1, 0, 2)  # Adjust shape
            x, y = x.cuda(), y.cuda()  # Move to GPU
            mask = mask.cuda()  # Move to GPU
            optimizer.zero_grad()
            y_pred = model(x)
            loss = masked_loss_fn(y_pred, y, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y = x.permute(1, 0, 2), y.permute(1, 0, 2)
                x, y = x.cuda(), y.cuda()  # Move to GPU
                mask = mask.cuda()  # Move to GPU
                y_pred = model(x)
                loss = masked_loss_fn(y_pred, y, mask)
                val_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
        test_model(model, test_loader)

# Train the model
train_model(model, train_loader, val_loader)


# Run testing
test_model(model, test_loader)