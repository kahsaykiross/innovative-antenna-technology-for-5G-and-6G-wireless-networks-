# ============================================================
# 3D Multi-Cell 6G Antenna System Dashboard with RL Control
# ============================================================

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import copy

# -----------------------------
# Simulation parameters
# -----------------------------
num_cells = 3
antennas_per_cell = 4
users_per_cell = 3
ris_per_cell = 5

cell_positions = np.random.rand(num_cells,3)*50
antenna_positions = [cell_positions[i] + np.random.rand(antennas_per_cell,3)*5 for i in range(num_cells)]
user_positions = [cell_positions[i] + np.random.rand(users_per_cell,3)*5 for i in range(num_cells)]
ris_positions = [cell_positions[i] + np.random.rand(ris_per_cell,3)*5 for i in range(num_cells)]

beam_levels = [np.random.rand(antennas_per_cell) for _ in range(num_cells)]
ris_phases = [np.random.rand(ris_per_cell)*2*np.pi for _ in range(num_cells)]

# -----------------------------
# GNN for antenna embeddings
# -----------------------------
class AntennaGNN(nn.Module):
    def __init__(self, in_ch=3, hidden_ch=16, out_ch=4):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden_ch)
        self.conv2 = GCNConv(hidden_ch, out_ch)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

gnn_model = AntennaGNN()

def generate_graph(num_antennas):
    edge_index = []
    for i in range(num_antennas):
        for j in range(num_antennas):
            if i != j:
                edge_index.append([i,j])
    return torch.tensor(edge_index).t().contiguous()

# -----------------------------
# Simple RL Agent for beam + RIS control
# -----------------------------
class SimpleRLAgent:
    def __init__(self, num_antennas, num_ris):
        self.num_antennas = num_antennas
        self.num_ris = num_ris
        self.lr = 0.1
        # Initialize Q-tables for discrete beam/RIS adjustments
        self.q_table_beam = np.zeros((11, num_antennas))
        self.q_table_ris = np.zeros((11, num_ris))
    
    def discretize(self, val):
        return int(np.clip(val*10,0,10))
    
    def select_action(self, beam_levels, ris_phases):
        beam_action = np.array([self.discretize(b) for b in beam_levels])
        ris_action = np.array([self.discretize(p/(2*np.pi)) for p in ris_phases])
        return beam_action, ris_action
    
    def update(self, beam_levels, ris_phases, reward):
        # Simple RL update (dummy for simulation)
        pass  # extend for real Q-learning if desired

rl_agent = SimpleRLAgent(antennas_per_cell, ris_per_cell)

# -----------------------------
# Dash App
# -----------------------------
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("3D Multi-Cell 6G Dashboard with RL-Controlled Beam/RIS"),
    
    html.Div([
        html.Button("Step Simulation", id="step-btn", n_clicks=0),
        html.Div(id="sim-output")
    ]),
    
    dcc.Graph(id="antenna-3d-graph", style={"height":"700px"}),
    dcc.Graph(id="ris-3d-graph", style={"height":"400px"})
])

# -----------------------------
# Callback
# -----------------------------
@app.callback(
    Output("antenna-3d-graph","figure"),
    Output("ris-3d-graph","figure"),
    Output("sim-output","children"),
    Input("step-btn","n_clicks")
)
def update_simulation(n_clicks):
    global antenna_positions, user_positions, ris_positions, beam_levels, ris_phases, gnn_model, rl_agent
    
    # --- Update GNN embeddings for beamforming adaptation ---
    for i in range(num_cells):
        x = torch.tensor(antenna_positions[i],dtype=torch.float32)
        edge_index = generate_graph(antennas_per_cell)
        embeddings = gnn_model(x, edge_index).detach().numpy()
        # Use embeddings mean as beamforming base
        beam_levels[i] = np.clip(np.mean(embeddings, axis=1),0,1)
    
    # --- RL decides final beam + RIS adjustments ---
    for i in range(num_cells):
        beam_action, ris_action = rl_agent.select_action(beam_levels[i], ris_phases[i])
        beam_levels[i] = beam_action / 10.0
        ris_phases[i] = ris_action / 10.0 * 2*np.pi
    
    # --- Simulate small movements for environment dynamics ---
    for i in range(num_cells):
        antenna_positions[i] += (np.random.rand(antennas_per_cell,3)-0.5)*0.2
        user_positions[i] += (np.random.rand(users_per_cell,3)-0.5)*0.2
        ris_positions[i] += (np.random.rand(ris_per_cell,3)-0.5)*0.2
    
    # --- 3D Antenna & Users + MIMO Links ---
    fig_ant = go.Figure()
    colorscale = 'Viridis'
    for i in range(num_cells):
        # Antennas
        fig_ant.add_trace(go.Scatter3d(
            x=antenna_positions[i][:,0], y=antenna_positions[i][:,1], z=antenna_positions[i][:,2],
            mode='markers+text',
            marker=dict(size=8,color=beam_levels[i],colorscale=colorscale,cmin=0,cmax=1),
            text=[f"A{i}_{j}" for j in range(antennas_per_cell)],
            textposition="top center", name=f"Cell {i} Antennas"
        ))
        # Users
        fig_ant.add_trace(go.Scatter3d(
            x=user_positions[i][:,0], y=user_positions[i][:,1], z=user_positions[i][:,2],
            mode='markers+text',
            marker=dict(size=6,color='red'),
            text=[f"U{i}_{j}" for j in range(users_per_cell)],
            textposition="bottom center", name=f"Cell {i} Users"
        ))
        # MIMO links
        for a in range(antennas_per_cell):
            for u in range(users_per_cell):
                fig_ant.add_trace(go.Scatter3d(
                    x=[antenna_positions[i][a,0], user_positions[i][u,0]],
                    y=[antenna_positions[i][a,1], user_positions[i][u,1]],
                    z=[antenna_positions[i][a,2], user_positions[i][u,2]],
                    mode='lines', line=dict(color='blue',width=1), showlegend=False
                ))
    
    fig_ant.update_layout(scene=dict(xaxis_title='X',yaxis_title='Y',zaxis_title='Z'),
                          title="3D Antenna, Users & MIMO Links")
    
    # --- 3D RIS Figure ---
    fig_ris = go.Figure()
    for i in range(num_cells):
        fig_ris.add_trace(go.Scatter3d(
            x=ris_positions[i][:,0], y=ris_positions[i][:,1], z=ris_positions[i][:,2],
            mode='markers+text',
            marker=dict(size=6, color=ris_phases[i], colorscale='Cividis', cmin=0, cmax=2*np.pi),
            text=[f"RIS{i}_{j}" for j in range(ris_per_cell)],
            textposition="top center", name=f"Cell {i} RIS"
        ))
    fig_ris.update_layout(scene=dict(xaxis_title='X',yaxis_title='Y',zaxis_title='Z'),
                          title="3D RIS Elements (Phase in rad)")
    
    return fig_ant, fig_ris, f"Step {n_clicks}: RL-controlled beamforming & RIS updated."

# -----------------------------
# Run server
# -----------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
