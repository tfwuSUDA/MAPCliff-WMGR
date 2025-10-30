import torch
import torch.nn as nn
import torch.nn.functional as F


class fp_Fingerprint(nn.Module):
    def __init__(self, args):
        super(fp_Fingerprint, self).__init__()
        self.atom_fc = nn.Linear(args.input_feature_dim, args.fingerprint_dim)
        self.neighbor_fc = nn.Linear(args.input_feature_dim + args.input_bond_dim, args.fingerprint_dim)
        self.GRUCell = nn.ModuleList([nn.GRUCell(args.fingerprint_dim, args.fingerprint_dim)])
        self.align = nn.ModuleList([nn.Linear(2 * args.fingerprint_dim, 1)])
        self.attend = nn.ModuleList([nn.Linear(args.fingerprint_dim, args.fingerprint_dim)])
        self.dropout = nn.Dropout(p=args.p_dropout)

    def forward(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask):
        atom_mask = atom_mask.unsqueeze(2)
        batch_size, mol_length, num_atom_feat = atom_list.size()
        atom_feature = F.leaky_relu(self.atom_fc(atom_list))

        bond_neighbor = torch.stack([bond_list[i][bond_degree_list[i]] for i in range(batch_size)], dim=0)
        atom_neighbor = torch.stack([atom_list[i][atom_degree_list[i]] for i in range(batch_size)], dim=0)
        neighbor_feature = torch.cat([atom_neighbor, bond_neighbor], dim=-1)
        neighbor_feature = F.leaky_relu(self.neighbor_fc(neighbor_feature))

        attend_mask = atom_degree_list.clone()
        attend_mask[attend_mask != mol_length - 1] = 1
        attend_mask[attend_mask == mol_length - 1] = 0
        attend_mask = attend_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)
        
        softmax_mask = atom_degree_list.clone()
        softmax_mask[softmax_mask != mol_length - 1] = 0
        softmax_mask[softmax_mask == mol_length - 1] = -9e8
        softmax_mask = softmax_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        batch_size, mol_length, max_neighbor_num, fingerprint_dim = neighbor_feature.shape
        atom_feature_expand = atom_feature.unsqueeze(-2).expand(batch_size, mol_length, max_neighbor_num, fingerprint_dim)
        feature_align = torch.cat([atom_feature_expand, neighbor_feature], dim=-1)
        align_score = F.leaky_relu(self.align[0](self.dropout(feature_align)))
        align_score = align_score + softmax_mask
        attention_weight = F.softmax(align_score, -2)
        attention_weight = attention_weight * attend_mask
        neighbor_feature_transform = self.attend[0](self.dropout(neighbor_feature))
        context = torch.sum(torch.mul(attention_weight, neighbor_feature_transform), -2)
        context = F.elu(context)
        context_reshape = context.view(batch_size * mol_length, fingerprint_dim)
        atom_feature_reshape = atom_feature.view(batch_size * mol_length, fingerprint_dim)
        atom_feature_reshape = self.GRUCell[0](context_reshape, atom_feature_reshape)
        atom_feature = atom_feature_reshape.view(batch_size, mol_length, fingerprint_dim)
        activated_features = F.relu(atom_feature)
        return activated_features



class new_pass_FP_GAT_later_layer(nn.Module):
    def __init__(self, args, snn_args, input_dim, output_dim):
        super(new_pass_FP_GAT_later_layer, self).__init__()
        self.fingerprint_dim = args.fingerprint_dim
        self.neighbor_fc = nn.Linear(input_dim, args.fingerprint_dim)
        self.need_initializer = snn_args.need_initializer
        self.attention_combine = snn_args.attention_combine
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = snn_args.use_bias
        self.num_attention_heads = snn_args.num_attention_heads
        self.dropout = nn.Dropout(snn_args.attention_dropout)
        
        self.kernels = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, output_dim)) 
            for _ in range(self.num_attention_heads)
        ])
        self.attention_kernels = nn.ParameterList([
            nn.Parameter(torch.empty(2 * self.fingerprint_dim, 1)) 
            for _ in range(self.num_attention_heads)
        ])
        self.epsilons = nn.ParameterList([
            nn.Parameter(torch.full((1,), 0.6)) 
            for _ in range(self.num_attention_heads)
        ])
        
        if snn_args.use_bias:
            self.biases = nn.ParameterList([
                nn.Parameter(torch.zeros(self.fingerprint_dim)) 
                for _ in range(self.num_attention_heads)
            ])
            self.attention_biases = nn.ParameterList([
                nn.Parameter(torch.zeros(1)) 
                for _ in range(self.num_attention_heads)
            ])
        
        if self.need_initializer:
            for kernel in self.kernels:
                nn.init.xavier_uniform_(kernel)
            for attention_kernel in self.attention_kernels:
                nn.init.xavier_uniform_(attention_kernel)

    def forward(self, atom_neighbor_SNN, x_full_atom_neighbors):
        output = []
        batch_size = atom_neighbor_SNN.size(0)
        
        for i in range(self.num_attention_heads):
            diag_elements = torch.diagonal(x_full_atom_neighbors, dim1=-2, dim2=-1)
            scaled_diag = diag_elements * self.epsilons[i]
            update_diag = torch.diag_embed(scaled_diag)
            temp_mask = torch.eye(x_full_atom_neighbors.size(1), device=x_full_atom_neighbors.device).repeat(batch_size, 1, 1)
            new_x_full_atom_neighbors = temp_mask * update_diag + (1 - temp_mask) * x_full_atom_neighbors
            
            neighbor_feature_SNN = torch.matmul(new_x_full_atom_neighbors, atom_neighbor_SNN)
            neighbor_feature_SNN = F.leaky_relu(self.neighbor_fc(neighbor_feature_SNN))
            neighbor_feature_SNN = torch.matmul(neighbor_feature_SNN, self.kernels[i])
            
            if self.use_bias:
                neighbor_feature_SNN += self.biases[i]
                
            attention_neighbor_feature_self = torch.matmul(neighbor_feature_SNN, self.attention_kernels[i][:self.fingerprint_dim])
            attention_neighbor_feature_neigh = torch.matmul(neighbor_feature_SNN, self.attention_kernels[i][self.fingerprint_dim:])
            
            if self.use_bias:
                attention_neighbor_feature_self += self.attention_biases[i]
                
            attention_matrix = attention_neighbor_feature_self + torch.transpose(attention_neighbor_feature_neigh, -2, -1)
            attention_matrix = F.elu(attention_matrix, alpha=1.0)
            mask_SNN = torch.exp(x_full_atom_neighbors * -10e9) * -10e9
            attention_matrix = attention_matrix + mask_SNN
            attention_matrix = F.softmax(attention_matrix, dim=-1)
            attention_matrix = self.dropout(attention_matrix)
            node_feature_SNN = torch.matmul(attention_matrix, neighbor_feature_SNN)
            output.append(node_feature_SNN)
            
        if self.attention_combine == 'concat':
            output = torch.cat(output, dim=-1)
        elif self.attention_combine == 'mean':
            output = torch.mean(torch.stack(output, dim=0), dim=0)
            
        return output, attention_matrix

    def apply_unit_norm(self):
        for kernel in self.kernels:
            kernel.data = F.normalize(kernel.data, p=2, dim=0)
        for attention_kernel in self.attention_kernels:
            attention_kernel.data = F.normalize(attention_kernel.data, p=2, dim=0)
        if self.use_bias:
            for bias in self.biases:
                bias.data = F.normalize(bias.data, p=2, dim=0)
            for attention_bias in self.attention_biases:
                attention_bias.data = F.normalize(attention_bias.data, p=2, dim=0)



class GraphSNN_GAT(nn.Module):
    def __init__(self, args, snn_args):
        super(GraphSNN_GAT, self).__init__()
        self.num_layers = args.num_layers
        self.dropout = nn.Dropout(p=args.p_dropout)
        self.dropout_linear = nn.Dropout(p=0.2)
        self.Fusion_atom_and_bond = fp_Fingerprint(args)
        self.layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            self.layers.append(
                new_pass_FP_GAT_later_layer(
                    args, snn_args, 
                    snn_args.num_attention_heads * args.fingerprint_dim, 
                    snn_args.num_attention_heads * args.fingerprint_dim
                )
            )
            
        output_input_dim = 39 + args.fingerprint_dim * (args.num_layers + 1 + args.T) * snn_args.num_attention_heads
        output_hidden_dim = args.fingerprint_dim * snn_args.num_attention_heads
        self.output1 = nn.Linear(output_input_dim, output_hidden_dim)
        self.output2 = nn.Linear(output_hidden_dim, args.output_units_num)

    def forward(self, atom_feature, bond_feature, atom_degree_list, bond_degree_list, atom_mask, x_full_atom_neighbors, x_full_bond_neighbors):
        hidden_states = [atom_feature]
        atom_feature = self.Fusion_atom_and_bond(atom_feature, bond_feature, atom_degree_list, bond_degree_list, atom_mask)
        atom_feature = self.dropout(atom_feature)
        hidden_states.append(atom_feature)
        
        attention_matrices_all_layers = []
        for i in range(self.num_layers):
            atom_feature, attention_matrices = self.layers[i](atom_feature, x_full_atom_neighbors)
            atom_feature = F.relu(atom_feature)
            atom_feature = self.dropout(atom_feature)
            hidden_states.append(atom_feature)
            attention_matrices_all_layers.append(attention_matrices)
            
        atom_feature = torch.cat(hidden_states, dim=2).sum(dim=1)
        final_feature = self.dropout_linear(F.relu(self.output1(atom_feature)))
        res = self.output2(final_feature)

        return res, final_feature