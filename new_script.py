import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class NLPEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(NLPEncoder, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        output, _ = self.gru(x)
        return output


class GCN(nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class AttentionFusion(nn.Module):
    def __init__(self, hidden_size, fusion_size):
        super(AttentionFusion, self).__init__()
        self.W = nn.Linear(hidden_size * 2, fusion_size)
        self.V = nn.Linear(fusion_size, 1)

    def forward(self, seq_embed, struct_embed):
        combined = torch.cat((seq_embed, struct_embed), dim=-1)
        energy = torch.tanh(self.W(combined))
        attention = F.softmax(self.V(energy), dim=1)
        fused = torch.sum(attention * combined, dim=1)
        return fused


class NLPDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(NLPDecoder, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.gru(x)
        output = self.fc(output)
        return output

    def generate(self, initial_input, seq_length):
        output_seq = []
        input_step = initial_input.unsqueeze(1)

        for _ in range(seq_length):
            output, _ = self.gru(input_step)
            output = self.fc(output)
            output_step = torch.argmax(output, dim=-1)
            output_seq.append(output_step)
            input_step = F.one_hot(output_step, num_classes=self.fc.out_features).float()

        return torch.cat(output_seq, dim=1)


class AntibodyOptimizationModel(nn.Module):
    def __init__(self, nlp_input_size, nlp_hidden_size, gcn_node_features, gcn_hidden_channels, fusion_size,
                 nlp_output_size, num_layers=1):
        super(AntibodyOptimizationModel, self).__init__()
        self.encoder = NLPEncoder(nlp_input_size, nlp_hidden_size, num_layers)
        self.gcn = GCN(gcn_node_features, gcn_hidden_channels)
        self.attention_fusion = AttentionFusion(nlp_hidden_size, fusion_size)
        self.decoder = NLPDecoder(fusion_size, nlp_hidden_size, num_layers, nlp_output_size)

    def forward(self, seq_input, graph_input, edge_index):
        seq_embed = self.encoder(seq_input)
        struct_embed = self.gcn(graph_input, edge_index)
        fused = self.attention_fusion(seq_embed, struct_embed)
        output = self.decoder(fused.unsqueeze(1))
        return output

def generate_optimized_sequence(model, seq_input, graph_input, edge_index, initial_input, seq_length):
    model.eval()

    with torch.no_grad():
        seq_embed = model.encoder(seq_input)
    struct_embed = model.gcn(graph_input, edge_index)
    fused = model.attention_fusion(seq_embed, struct_embed)
    optimized_seq = model.decoder.generate(fused, seq_length)
    return optimized_seq

nlp_input_size = 21
nlp_hidden_size = 128
gcn_node_features = 3
gcn_hidden_channels = 128
fusion_size = 128
nlp_output_size = 21
num_layers = 1

model = AntibodyOptimizationModel(nlp_input_size, nlp_hidden_size, gcn_node_features, gcn_hidden_channels,
                                  fusion_size, nlp_output_size, num_layers)

seq_input = torch.randn(1, 100, nlp_input_size)
graph_input = torch.randn(100, gcn_node_features)
edge_index = torch.randint(0, 100, (2, 50))
initial_input = torch.zeros(1, nlp_input_size)

optimized_seq = generate_optimized_sequence(model, seq_input, graph_input, edge_index, initial_input, seq_length=100)
print(optimized_seq.shape)
print(optimized_seq)
