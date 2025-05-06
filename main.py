import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import itertools
import argparse

import shapely
from shapely.geometry import Polygon, Point
from matplotlib.path import Path
from skimage.draw import polygon2mask
from mpl_toolkits.axes_grid1 import make_axes_locatable

import imageio
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
import copy
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import MSELoss
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Tuple, Dict
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
import base64

# ======================================================

# Constants for LBM

# D2Q9 Lattice velocities and weights
c = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
)
w = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
opp = [0, 3, 4, 1, 2, 7, 8, 5, 6]  # Opposite directions

SELECTED_TARGETS = np.array(["drag", "lift", "mean_abs_wake_vorticity"])

ALL_TARGETS = [
    "drag",
    "lift",
    "mean_abs_wake_vorticity",
    "max_wake_vorticity",
    "mean_velocity_wake",
    "std_velocity_wake",
    "kinetic_energy_total",
]

TARGET_INDICES = np.array([ALL_TARGETS.index(t) for t in SELECTED_TARGETS])

ALL_TARGETS = np.array(ALL_TARGETS)

# The dataset dir will be created at training data generation if it does not exist
DATASET_DIR = "dataset"

# ======================================================
# FastAPI RESTful interface


app = FastAPI()

# Enable CORS for local testing (optional but useful)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def initialise():
    model, scaler = load_optimised_model_and_scaler()
    app.state.model = model
    app.state.scaler = scaler


class PolygonRequest(BaseModel):
    vertices: List[Tuple[float, float]]


@app.post("/simulate")
async def simulate(request: PolygonRequest):
    """
    This endpoint takes a set of points comprising a user-defined poly and

    1) Performs inference over the polygon using the trained surrogate (instant) and
    2) Runs the full CFD for comparison and visualisation (slow)
    """

    model = app.state.model
    scaler = app.state.scaler

    test_sample = generate_graph_from_vertices(np.array(request.vertices))

    with torch.no_grad():
        out = model(test_sample)
    pred_scaled = out.cpu().numpy()
    pred_targets = scaler.inverse_transform(pred_scaled).flatten()

    reloaded_mask, reloaded_shifted_poly = rasterize_polygon_from_grid_vertices(
        test_sample.x, grid_shape=(128, 256)
    )

    rho, u, f, _ = run_lbm(reloaded_mask, n_steps=1000, u0=0.2, frame_interval=1001)
    omega = compute_vorticity(u)
    wake_bounds = (96, 176, 38, 88)
    cfd_targets = compute_scalar_targets(
        u, f, omega, reloaded_mask, wake_bounds, rho=rho
    )

    fig = visualize_velocity(u, reloaded_mask, scale=0.1, stride=4)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_bytes = buf.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    plt.close(fig)

    results = [
        {"name": name, "surrogate": float(pred), "cfd": float(cfd)}
        for name, pred, cfd in zip(SELECTED_TARGETS, pred_targets, cfd_targets.values())
    ]

    return {
        "image": img_b64,  # base64-encoded image string
        "results": results,
    }


class OptimisationInput(BaseModel):
    targets: Dict[str, float]


@app.post("/optimise")
def optimise_endpoint(input: OptimisationInput):
    """
    Executes the inverse problem; using the surrogate to rapidly optimise a polygon
    such that it exhibits the user's specified drag, lift and wake vorticity
    """

    # Validate input keys
    invalid_keys = [k for k in input.targets.keys() if k not in SELECTED_TARGETS]
    if invalid_keys:
        raise HTTPException(
            status_code=400, detail=f"Invalid target(s): {invalid_keys}"
        )

    # Server-side log
    print(f"Optimising for targets: {input.targets}")

    # Call your actual optimisation function here
    optimised_vertices, optim_fun = optimise(
        [v for v in input.targets.values()],
        app.state.model,
        app.state.scaler,
        max_evals=1000,
        n_vertices=30,
        r_bounds=(5, 30),
        device="cpu",
        grid_shape=(128, 256),
    )

    return {"vertices": [[float(x), float(y)] for x, y in optimised_vertices]}


# Serve HTML/JS/CSS -- PUT IT after defining ROUTES
app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ======================================================


def load_optimised_model_and_scaler():
    """
    Restore previously serialised GAT model
    """
    model = SimpleGNN(
        hidden_dim=128,
        out_dim=len(SELECTED_TARGETS),
        num_layers=3,
        dropout=0.1,  # Enable dropout for uncertainty estimation
        use_gat=True,
        heads=4,
    )

    model.load_state_dict(torch.load("pre_trained_model/gnn_model_v6.pth"))
    model.eval()

    with open("pre_trained_model/gnn_scaler_v6.pkl", "rb") as f:
        scaler = pickle.load(f)

    print(f"Loaded GNN surrogate {model} and scaler")

    return model, scaler


# ======================================================
#
# Generate and manage LBM training examples
#


class ShapeCFDDatasetBuilder:
    def __init__(self, output_dir, grid_shape, n_samples, shape_type, wake_bounds):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.grid_shape = grid_shape
        self.n_samples = n_samples
        self.shape_type = shape_type
        self.wake_bounds = wake_bounds

    def generate_sample(self, i):
        try:
            scale_fraction = np.random.uniform(0.15, 0.35)

            if self.shape_type == "spiky":

                polygon = generate_random_polygon(
                    n_points=np.random.randint(10, 50),
                    radius=20,
                    jitter=np.random.uniform(0.05, 0.45),
                    seed=None,
                    asymmetry_strength=np.random.uniform(0.0, 0.95),
                    angle_of_attack=np.random.uniform(-180, 180),  # degrees clockwise
                )

            elif self.shape_type == "wing":
                polygon = generate_parametric_airfoil(
                    n_points=int(16 + 32 * np.random.rand()),
                    chord_length=15 + 30 * np.random.rand(),
                    max_camber=0.02 + 0.1 * np.random.rand(),
                    camber_position=0.1 + 0.5 * np.random.rand(),
                    thickness=0.03 + 0.4 * np.random.rand(),
                    angle_of_attack_deg=np.pi * 2 * np.random.rand(),
                    grid_shape=(128, 128),
                )
            else:
                raise NotImplementedError

            mask, shaped_poly = rasterize_polygon(
                polygon,
                grid_shape=self.grid_shape,
                target_extent_fraction=scale_fraction,
            )
            rho, u, f, _ = run_lbm(mask, n_steps=1000, u0=0.2, frame_interval=1001)
            omega = compute_vorticity(u)
            targets = compute_scalar_targets(
                u, f, omega, mask, self.wake_bounds, rho=rho
            )
            data = polygon_to_graph(shaped_poly, targets)
            outfile = os.path.join(self.output_dir, f"sample_{i:06d}.pt")
            torch.save(data, outfile)
            return i, None

        except Exception as e:
            print(str(e))
            return i, str(e)

    def build_dataset(self, use_multiprocessing=True, max_workers=None):
        # Find the next available sample index, so we can incrementally add to the dataset
        existing_files = [
            f
            for f in os.listdir(self.output_dir)
            if f.startswith("sample_") and f.endswith(".pt")
        ]
        existing_indices = sorted(
            [
                int(f.split("_")[1].split(".")[0])
                for f in existing_files
                if f.split("_")[1].split(".")[0].isdigit()
            ]
        )
        start_index = (max(existing_indices) + 1) if existing_indices else 0

        print(
            f"Found {len(existing_files)} existing samples. Generating {self.n_samples} more..."
        )
        indices_to_generate = range(start_index, start_index + self.n_samples)

        if use_multiprocessing:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.generate_sample, i): i
                    for i in indices_to_generate
                }
                for future in tqdm.tqdm(as_completed(futures), total=self.n_samples):
                    i, error = future.result()
                    if error:
                        print(f"[!] Sample {i} failed: {error}")
        else:
            for i in tqdm.tqdm(indices_to_generate):
                i, error = self.generate_sample(i)
                if error:
                    print(f"[!] Sample {i} failed: {error}")


class ShapeCFDDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data_list = []
        for dirpath, _, filenames in os.walk(root, followlinks=True):
            for f in sorted(filenames):
                if not f.endswith(".pt"):
                    continue
                full_path = os.path.join(dirpath, f)
                try:
                    data = torch.load(full_path)
                    if (
                        not hasattr(data, "y")
                        or data.y.ndim != 1
                        or data.y.shape[0] < len(ALL_TARGETS)
                    ):
                        print(
                            f"[SKIP] {full_path} has malformed y: {getattr(data, 'y', None)}"
                        )
                        continue
                    self.data_list.append(data)
                except Exception as e:
                    print(f"[ERROR] Failed to load {full_path}: {e}")
                    continue

        self.data, self.slices = self.collate(self.data_list)

        # Validate post-collation retrieval
        for i in range(len(self.data_list)):
            sample = self.get(i)
            y_tensor = sample.y
            if y_tensor.ndim == 0 or y_tensor.shape[0] <= 1:
                print(f"[BAD] Sample {i:06d} has y shape: {y_tensor.shape}")

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


class NormalizedTargetDataset(Dataset):
    def __init__(self, base_dataset, target_indices=None, scaler=None):
        self.base_dataset = base_dataset
        self.target_indices = target_indices[
            target_indices < base_dataset[0].y.shape[0]
        ]

        if scaler is None:
            raise ValueError("Must provide a precomputed StandardScaler.")
        self.scaler = scaler

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.base_dataset[idx])

        y_tensor = data.y
        if y_tensor.ndim == 0:
            y_tensor = y_tensor.unsqueeze(0)  # make it shape [1]

        # Validate size
        if max(self.target_indices) >= y_tensor.shape[0]:
            raise IndexError(
                f"BAD SAMPLE {idx} Requested target index {max(self.target_indices)}, but y only has shape {y_tensor.shape} y={data.y}"
            )

        y = y_tensor[self.target_indices].numpy().reshape(1, -1)
        data.y = torch.tensor(self.scaler.transform(y).squeeze(), dtype=torch.float)

        return data

    def inverse_transform(self, y_tensor):
        y_np = y_tensor.detach().cpu().numpy()
        return self.scaler.inverse_transform(y_np)


def compute_scaler_from_dataset(dataset, target_indices):
    y_all = [data.y[target_indices].numpy() for data in dataset]
    scaler = StandardScaler().fit(y_all)
    return scaler


# ======================================================
#
# Graph Neural Network and Graph Attention Network implementation
#


class SimpleGNN(nn.Module):
    def __init__(
        self,
        in_dim=2,
        hidden_dim=64,
        out_dim=1,
        num_layers=2,
        dropout=0.0,
        use_gat=False,
        heads=1,
    ):
        super().__init__()
        assert num_layers >= 1, "There must be at least one hidden layer"
        self.use_gat = use_gat
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.dropouts = nn.ModuleList()  # Add explicit dropout layers

        # First layer
        if use_gat:
            self.convs.append(
                GATConv((in_dim, in_dim), hidden_dim // heads, heads=heads)
            )
        else:
            self.convs.append(GCNConv(in_dim, hidden_dim))
        self.dropouts.append(nn.Dropout(p=dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            if use_gat:
                self.convs.append(
                    GATConv((hidden_dim, hidden_dim), hidden_dim // heads, heads=heads)
                )
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.dropouts.append(nn.Dropout(p=dropout))

        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, dropout in zip(self.convs, self.dropouts):
            x = F.relu(conv(x, edge_index))
            x = dropout(x)  # Use explicit dropout layer
        x = global_mean_pool(x, batch)
        return self.lin(x)


def enable_dropout(model: nn.Module) -> None:
    """
    Sets all dropout layers to train mode so they keep dropping units
    even when the rest of the model is in eval mode.
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            m.p = 0.1  # Set dropout probability directly on the layer

@torch.no_grad()
def mc_predict(loader, model, n_samples: int = 50):
    """
    Performs MC-dropout inference.
    Returns:
        mean  - tensor [N, out_dim]
        std   - tensor [N, out_dim]  (epistemic uncertainty proxy)
    """
    # Set model to train mode to enable dropout
    model.train()
    preds = []

    for _ in range(n_samples):
        batch_preds = []
        for batch in loader:         # works with any PyG DataLoader
            batch_preds.append(model(batch))
        preds.append(torch.cat(batch_preds, dim=0))

    stacked = torch.stack(preds)     # [S, N, out_dim]
    return stacked.mean(0), stacked.std(0)



# ======================================================
#
# Shape generation routines for a) training set generation and b) optimisation
#


def generate_random_polygon(
    n_points=30,
    radius=10,
    jitter=0.5,
    seed=None,
    asymmetry_strength=0.0,  # 0 = symmetric, 1 = strong vertical bias
    angle_of_attack=0.0,  # degrees, clockwise
):
    """
    Generates a random closed polygon with optional asymmetry and rotation
    """
    if seed is not None:
        np.random.seed(seed)

    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    # Base radii with random jitter
    base_radii = radius * (1 + jitter * (2 * np.random.rand(n_points) - 1))

    # Shift sin so that bottom gets smaller, top gets bigger
    vertical_bias = 1 + (asymmetry_strength / 2) * np.sin(angles)

    radii = base_radii * vertical_bias

    # Convert polar to Cartesian coordinates
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    y *= 1 - asymmetry_strength
    points = np.vstack((x, y)).T

    # Rotate by angle_of_attack
    theta = np.radians(angle_of_attack)
    rot_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    rotated_points = points @ rot_matrix.T

    return Polygon(rotated_points)


def generate_parametric_airfoil(
    n_points=64,
    chord_length=60,
    max_camber=0.08,
    camber_position=0.4,
    thickness=0.12,
    angle_of_attack_deg=0,
    grid_shape=(128, 256),
):
    """
    Generates a wing-like polygon based on parametric camber/thickness airfoil model.

    Returns:
        (n_points, 2) array of coordinates, closed loop.
    """
    cx = grid_shape[1] * 0.25
    cy = grid_shape[0] * 0.5

    x = np.linspace(0, 1, n_points // 2)
    yt = (
        5
        * thickness
        * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )
    )

    # Camber line
    yc = np.where(
        x < camber_position,
        max_camber / (camber_position**2) * (2 * camber_position * x - x**2),
        max_camber
        / ((1 - camber_position) ** 2)
        * ((1 - 2 * camber_position) + 2 * camber_position * x - x**2),
    )
    dyc_dx = np.gradient(yc, x)
    theta = np.arctan(dyc_dx)

    # Upper and lower surface
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Combine into full polygon
    x_full = np.concatenate([xu, xl[::-1]])
    y_full = np.concatenate([yu, yl[::-1]])

    # Rotate for AoA
    # aoa = np.radians(angle_of_attack_deg)
    aoa = angle_of_attack_deg
    coords = np.stack([x_full * chord_length, y_full * chord_length], axis=1)
    rot_matrix = np.array([[np.cos(aoa), -np.sin(aoa)], [np.sin(aoa), np.cos(aoa)]])
    coords = coords @ rot_matrix.T

    # Translate centroid to cx, cy
    centroid = np.mean(coords, axis=0)

    # Translate to domain
    coords[:, 0] += cx - centroid[0]
    coords[:, 1] += cy - centroid[1]
    return Polygon(coords)


def sample_winglike_polys():
    """
    Generate and plot polygon samples to validate the space
    """
    fig, axes = plt.subplots(7, 7, figsize=(12, 12))
    axes = axes.flatten()

    for ax in axes:

        shape = generate_parametric_airfoil(
            n_points=int(16 + 32 * np.random.rand()),
            chord_length=15 + 30 * np.random.rand(),
            max_camber=0.02 + 0.1 * np.random.rand(),
            camber_position=0.1 + 0.5 * np.random.rand(),
            thickness=0.03 + 0.4 * np.random.rand(),
            angle_of_attack_deg=np.pi * 2 * np.random.rand(),
            grid_shape=(128, 128),
        )

        polygon = polygon_to_vertices(shape)
        polygon = np.vstack([polygon, polygon[0]])

        ax.plot(polygon[:, 0], polygon[:, 1], "-o", markersize=2)
        ax.set_aspect("equal")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def polygon_to_graph(polygon, targets):
    """
    Converts a polygon and set of scalar targets to the PyG data format
    """
    coords = np.array(polygon.exterior.coords[:-1])  # remove closing repeat
    n = len(coords)

    x = torch.tensor(coords, dtype=torch.float)  # shape [n, 2]
    edge_index = torch.tensor(
        [list(range(n)), [(i + 1) % n for i in range(n)]], dtype=torch.long
    )  # shape [2, num_edges]

    y = torch.tensor([v for v in targets.values()], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)


def polygon_to_vertices(poly: Polygon) -> np.ndarray:
    """
    Converts a Shapely Polygon to a (N, 2) NumPy array of vertices.
    Ignores interior holes.
    """
    if not poly.is_valid:
        poly = poly.buffer(0)  # fix invalid geometries if needed

    coords = np.array(poly.exterior.coords)
    return (
        coords[:-1] if np.allclose(coords[0], coords[-1]) else coords
    )  # drop duplicate last point if closed


def generate_graph_from_vertices(vertices):
    """
    Converts a 2D polygon into a PyTorch Geometric graph for GNN input.

    This is for inference time only

    Args:
        vertices (np.ndarray): shape (n_vertices, 2), 2D polygon vertex coordinates.

    Returns:
        torch_geometric.data.Data: Graph with cyclic edges and node features.
    """
    num_nodes = vertices.shape[0]
    # Node features (you can add more later if needed)
    x = torch.tensor(vertices, dtype=torch.float)

    # Create cyclic edges
    edge_index = []
    for i in range(num_nodes):
        edge_index.append([i, (i + 1) % num_nodes])
        edge_index.append(
            [(i + 1) % num_nodes, i]
        )  # add both directions for undirected

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create the Data object
    data = Data(
        x=x, edge_index=edge_index, pos=x.clone()
    )  # pos is needed for visualization / GNNs that use it
    return data


def rasterize_polygon(polygon, grid_shape=(128, 256), target_extent_fraction=0.3):
    """
    Takes a polygon, scales it, and centres it in the correct location in the
    'wind tunnel'
    """
    ny, nx = grid_shape

    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx
    height = maxy - miny

    # Target shape size
    target_extent_x = target_extent_fraction * nx
    target_extent_y = target_extent_fraction * ny

    scale = min(target_extent_x / width, target_extent_y / height)

    # Center, scale, and position
    centroid = polygon.centroid
    centered = shapely.affinity.translate(polygon, xoff=-centroid.x, yoff=-centroid.y)
    scaled = shapely.affinity.scale(centered, xfact=scale, yfact=scale, origin=(0, 0))

    # Place object 1/4 into the domain horizontally, centered vertically
    shifted = shapely.affinity.translate(scaled, xoff=nx // 4, yoff=ny // 2)

    # Rasterize
    x = np.arange(nx)
    y = np.arange(ny)
    xv, yv = np.meshgrid(x, y)
    points = np.vstack((xv.ravel(), yv.ravel())).T

    path = Path(np.array(shifted.exterior.coords))
    mask = path.contains_points(points).reshape((ny, nx))

    return ~mask, shifted  # mask: True = fluid


def rasterize_polygon_from_grid_vertices(vertices, grid_shape, padding=0):
    """
    Rasterizes a polygon already defined in grid-space coordinates.
    Assumes no further scaling is needed.
    """
    poly = Polygon(vertices)
    coords = np.array(poly.exterior.coords)

    # Flip y-axis for image space
    coords_flipped = coords.copy()

    # Transpose (x, y) → (row, col)
    coords_for_mask = coords_flipped[:, [1, 0]]

    fluid_mask = ~polygon2mask(grid_shape, coords_for_mask)
    return fluid_mask, poly


def visualize_polygon_and_mask(polygon, mask):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Vector view
    axs[0].plot(*polygon.exterior.xy)
    axs[0].set_title("Polygon (vector view)")
    axs[0].axis("equal")

    # Raster view
    axs[1].imshow(mask, origin="lower", cmap="gray")
    axs[1].set_title("Rasterized Mask")
    plt.tight_layout()


# ======================================================
# Lattice Boltzmann CFD simulation


def initialize_lbm(nx, ny, tau=0.6, u0=0.1):
    rho = np.ones((ny, nx))
    u = np.zeros((ny, nx, 2))
    u[:, :, 0] = u0  # uniform inflow

    f = np.zeros((9, ny, nx))
    for i in range(9):
        cu = u[:, :, 0] * c[i, 0] + u[:, :, 1] * c[i, 1]
        f[i] = (
            w[i]
            * rho
            * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * (u[:, :, 0] ** 2 + u[:, :, 1] ** 2))
        )

    return rho, u, f


def lbm_step(rho, u, f, solid, tau, u0):
    ny, nx = rho.shape

    # Macroscopic variables
    rho = np.sum(f, axis=0)
    u = np.zeros_like(u)
    for i in range(9):
        u[:, :, 0] += f[i] * c[i, 0]
        u[:, :, 1] += f[i] * c[i, 1]
    u /= rho[:, :, None]

    # Enforce no fluid inside solid region
    u[solid] = 0.0
    rho[solid] = 1.0  # or whatever background density you use (typically 1)

    # Equilibrium distribution
    feq = np.zeros_like(f)
    u_sq = u[:, :, 0] ** 2 + u[:, :, 1] ** 2
    for i in range(9):
        cu = u[:, :, 0] * c[i, 0] + u[:, :, 1] * c[i, 1]
        feq[i] = w[i] * rho * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * u_sq)

    # Collision
    f += -(f - feq) / tau

    # Bounce-back
    for i in range(1, 9):
        bounce = f[opp[i]][solid]  # reflect from neighbor into solid
        f[i][solid] = bounce

    # Boundary conditions...

    # Inlet: impose velocity on x = 0 column
    u_in = np.zeros((rho.shape[0], 2))
    u_in[:, 0] = u0  # inflow velocity

    rho_in = rho[:, 0]  # density at inlet column
    for i in range(9):
        cu = u_in[:, 0] * c[i, 0] + u_in[:, 1] * c[i, 1]
        feq = (
            w[i]
            * rho_in
            * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * (u_in[:, 0] ** 2 + u_in[:, 1] ** 2))
        )
        f[i][:, 0] = feq

    # Outlet: zero-gradient approximation (copy from one cell left)
    for i in range(9):
        f[i][:, -1] = f[i][:, -2]

    # Streaming
    for i in range(9):
        f[i] = np.roll(f[i], shift=c[i][::-1], axis=(0, 1))  # note: y,x

    return rho, u, f


def run_lbm(mask, n_steps=1000, tau=0.6, u0=0.1, frame_interval=10):
    """
    Main CFD loop
    """
    ny, nx = mask.shape
    solid = ~mask  # inverse: True = wall

    rho, u, f = initialize_lbm(nx, ny, tau=tau, u0=u0)

    u_frames = []

    for step in range(n_steps):
        rho, u, f = lbm_step(rho, u, f, solid, tau, u0)

        if step % frame_interval == 0:
            u_frames.append(np.copy(u))

    return rho, u, f, u_frames


# ======================================================
# CFD output metrics and visualisation


def visualize_velocity(u, mask, scale=1.0, stride=4):
    """
    Visualizes the 2D velocity field.

    Parameters:
        u: (ny, nx, 2) array of velocity vectors
        mask: boolean (ny, nx) fluid mask
        scale: multiplier for quiver arrow length
        stride: spacing between quiver arrows
    """
    mag = np.linalg.norm(u, axis=2)
    ny, nx = mag.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(mag, origin="lower", cmap="viridis", interpolation="bilinear")

    # Create manually sized colorbar using axes divider
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Velocity Magnitude")

    # Add quiver arrows (sampled by stride)
    ax.quiver(
        X[::stride, ::stride],
        Y[::stride, ::stride],
        u[::stride, ::stride, 0],
        u[::stride, ::stride, 1],
        scale=1 / scale,
        color="white",
        pivot="middle",
    )

    ax.set_title("Velocity Field")
    ax.set_aspect("equal")
    plt.tight_layout()

    return fig


def save_velocity_frames(u_frames, mask, output_dir="frames", stride=4, scale=0.2):
    """
    Save (downsampled) frames of the LBM simulation
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, u in enumerate(u_frames):
        mag = np.linalg.norm(u, axis=2)
        ny, nx = mag.shape
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(mag, origin="lower", cmap="viridis")
        ax.quiver(
            X[::stride, ::stride],
            Y[::stride, ::stride],
            u[::stride, ::stride, 0],
            u[::stride, ::stride, 1],
            scale=1 / scale,
            color="white",
            pivot="middle",
        )
        ax.set_title(f"Velocity Field - Frame {i}")
        ax.set_axis_off()
        plt.tight_layout()
        fig.savefig(f"{output_dir}/frame_{i:04d}.png")
        plt.close(fig)


def create_gif_from_frames(frame_dir="frames", output_file="velocity.gif", fps=10):
    """
    Create an animated gif from stored frames of the LBM simulation
    """
    filenames = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])
    images = [imageio.imread(os.path.join(frame_dir, f)) for f in filenames]
    imageio.mimsave(output_file, images, fps=fps)


def compute_vorticity(u):
    """
    Compute 2D vorticity from velocity field.

    Parameters:
        u: (ny, nx, 2) array of velocity vectors

    Returns:
        omega: (ny, nx) vorticity field
    """
    uy = u[:, :, 1]
    ux = u[:, :, 0]

    dudy = np.gradient(uy, axis=0)  # du_y/dy
    dudx = np.gradient(ux, axis=1)  # du_x/dx

    omega = dudy - dudx
    return omega


def visualize_vorticity_with_wake(omega, mask, wake_bounds=None):
    """
    Visualize vorticity with an optional wake region overlay.

    Parameters:
        omega: (ny, nx) vorticity field
        mask: boolean (ny, nx) fluid mask
        wake_bounds: (xmin, xmax, ymin, ymax) tuple or None
    """
    omega_masked = np.copy(omega)
    omega_masked[~mask] = np.nan  # mask solid regions

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        omega_masked, origin="lower", cmap="coolwarm", interpolation="bilinear"
    )
    plt.colorbar(im, ax=ax, label="Vorticity")
    ax.set_title("Vorticity Field with Wake Region")
    ax.set_aspect("equal")

    if wake_bounds:
        xmin, xmax, ymin, ymax = wake_bounds
        rect = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor="yellow",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)

    plt.tight_layout()


def compute_mean_wake_vorticity_manual(omega, mask, wake_bounds):
    xmin, xmax, ymin, ymax = wake_bounds
    wake_region = omega[ymin:ymax, xmin:xmax]
    fluid_mask = mask[ymin:ymax, xmin:xmax]

    if np.sum(fluid_mask) == 0:
        return np.nan

    return np.nanmean(np.abs(wake_region[fluid_mask]))


def compute_drag(f, solid_mask):
    """
    Approximate drag force based on bounce-back momentum exchange.

    Parameters:
        f: (9, ny, nx) post-collision distribution functions
        solid_mask: boolean (ny, nx) array where True = solid cell

    Returns:
        drag_force: scalar estimate of drag in x-direction
    """
    ny, nx = solid_mask.shape
    drag = 0.0

    for i in range(1, 9):  # skip rest particle i=0
        cx, cy = c[i]
        shifted_solid = np.roll(np.roll(solid_mask, -cy, axis=0), -cx, axis=1)
        interface = shifted_solid & ~solid_mask  # fluid neighbor to solid cell

        # Add momentum exchange (only x-component for drag)
        drag += 2 * c[i, 0] * np.sum(f[i][interface])

    return drag


def compute_scalar_targets(u, f, omega, mask, wake_bounds, rho=None):
    """
    Compute a dictionary of scalar targets for training or analysis.

    Parameters:
        u: (ny, nx, 2) velocity field
        f: (9, ny, nx) post-collision distribution functions
        omega: (ny, nx) vorticity field
        mask: boolean (ny, nx) fluid mask (True = fluid)
        wake_bounds: (xmin, xmax, ymin, ymax)
        rho: optional (ny, nx) density field (used for kinetic energy)

    Returns:
        dict of scalar targets
    """
    xmin, xmax, ymin, ymax = wake_bounds
    wake_mask = mask[ymin:ymax, xmin:xmax]
    u_wake = u[ymin:ymax, xmin:xmax]
    omega_wake = omega[ymin:ymax, xmin:xmax]

    # Kinematic features
    u_mag = np.linalg.norm(u, axis=2)
    u_mag_wake = np.linalg.norm(u_wake, axis=2)

    # Kinetic energy
    if rho is None:
        rho = np.ones_like(u_mag)  # assume unit density
    ke_total = 0.5 * np.sum(rho * u_mag**2)

    # Drag and lift (momentum transfer via bounce-back)
    drag, lift = 0.0, 0.0
    solid = ~mask
    for i in range(1, 9):
        cx, cy = c[i]
        shifted_solid = np.roll(np.roll(solid, -cy, axis=0), -cx, axis=1)
        interface = shifted_solid & ~solid
        drag += 2 * c[i, 0] * np.sum(f[i][interface])
        lift += 2 * c[i, 1] * np.sum(f[i][interface])

    return {
        "drag": float(drag),
        "lift": float(lift),
        "mean_abs_wake_vorticity": float(np.nanmean(np.abs(omega_wake[wake_mask]))),
        "max_wake_vorticity": float(np.nanmax(np.abs(omega_wake[wake_mask]))),
        "mean_velocity_wake": float(np.nanmean(u_mag_wake[wake_mask])),
        "std_velocity_wake": float(np.nanstd(u_mag_wake[wake_mask])),
        "kinetic_energy_total": float(ke_total),
    }


# ======================================================
# Running the CFD simulation standalone (for visual debugging and exploration)


def run_single(gif_filename=None, shape_bias="wing", wind_tunnel_walls=False):

    if shape_bias == "wing":
        poly = generate_parametric_airfoil(
            n_points=int(16 + 32 * np.random.rand()),
            chord_length=15 + 30 * np.random.rand(),
            max_camber=0.02 + 0.1 * np.random.rand(),
            camber_position=0.1 + 0.5 * np.random.rand(),
            thickness=0.03 + 0.4 * np.random.rand(),
            angle_of_attack_deg=-0.1,
            grid_shape=(128, 128),
        )

    elif shape_bias == "spiky":
        poly = generate_random_polygon(
            n_points=40,
            radius=20,
            jitter=0.005,
            seed=42,
            asymmetry_strength=0.85,
            angle_of_attack=0,
        )
    else:
        raise NotImplementedError

    mask, _ = rasterize_polygon(
        poly, grid_shape=(128, 256), target_extent_fraction=0.35
    )

    if wind_tunnel_walls:
        # Add no-slip boundary conditions (slip might be better)
        # If we don't do this, we have cylindrical space
        mask[0, :] = False  # Top wall
        mask[-1, :] = False  # Bottom wall

    # CFD simulation for T=1000 steps etc (expensive)
    # Example usage with mask from earlier step
    # mask is a boolean grid where True = fluid, False = solid
    # u will be a (ny, nx, 2) array of final velocities

    rho, u, f, u_frames = run_lbm(
        mask, n_steps=1000, tau=0.6, u0=0.2, frame_interval=10
    )

    if gif_filename is not None:
        save_velocity_frames(u_frames, mask, output_dir="frames")
        create_gif_from_frames("frames", gif_filename, fps=10)
        print(f'Wrote animated gif to {gif_filename}')

    visualize_velocity(u, mask, scale=0.1, stride=4)

    omega = compute_vorticity(u)

    wake_bounds = (96, 176, 38, 88)  # (xmin, xmax, ymin, ymax)

    mean_vort = compute_mean_wake_vorticity_manual(omega, mask, wake_bounds)

    targets = compute_scalar_targets(u, f, omega, mask, wake_bounds)
    print(targets)

    visualize_vorticity_with_wake(omega, mask, wake_bounds)

    drag = compute_drag(f, ~mask)
    print(f"Drag: {drag:.4f}, Mean wake vorticity: {mean_vort:.4f}")

    plt.show()


def create_training_data_from_full_cfd(
    use_multiprocessing, n_samples, shape_type, wake_bounds
):
    builder = ShapeCFDDatasetBuilder(
        output_dir=DATASET_DIR,
        grid_shape=(128, 256),
        n_samples=n_samples,
        shape_type=shape_type,
        wake_bounds=wake_bounds,
    )
    builder.build_dataset(
        use_multiprocessing=use_multiprocessing, max_workers=None
    )  # or None for auto


def check_device_consistency(model, batch, verbose=True):
    """
    Checks that the model and all relevant tensors in the batch are on the same device.
    Prints warnings if mismatches are found.
    """
    model_device = next(model.parameters()).device
    issues = []

    if hasattr(batch, "x") and batch.x.device != model_device:
        issues.append(f"batch.x is on {batch.x.device}, expected {model_device}")
    if hasattr(batch, "edge_index") and batch.edge_index.device != model_device:
        issues.append(
            f"batch.edge_index is on {batch.edge_index.device}, expected {model_device}"
        )
    if hasattr(batch, "y") and batch.y.device != model_device:
        issues.append(f"batch.y is on {batch.y.device}, expected {model_device}")

    if issues and verbose:
        print("[Device Warning] Mismatched devices detected:")
        for issue in issues:
            print(" -", issue)
    elif verbose:
        print(f"All tensors and model are on {model_device}")

    return issues == []  # returns True if all OK


def sample_hypers_random(n_evals):
    """
    Hyperparameter schedule for random search approach
    """
    hypers = []
    for i in range(n_evals):
        hypers.append(
            {
                "hidden_dim": np.random.choice([32, 64, 128]),
                "num_layers": np.random.choice([2, 3]),
                "dropout": np.random.choice([0.0, 0.2, 0.4]),
                "lr": 10 ** np.random.uniform(-4, -2),
                "batch_size": np.random.choice([16, 32, 64]),
                "use_gat": np.random.choice([False, True]),
                "heads": np.random.choice([1, 2, 4, 8]),
            }
        )
    return hypers


def sample_hypers_grid():
    """
    Hyperparameter schedule for grid search approach
    """
    hidden_dim_values = np.array([64, 128])
    num_layers_values = np.array([2, 3])
    dropout_values = np.array([0.0, 0.2])
    lr_values = np.power(10.0, np.array([-4, -3, -2]))
    batch_size_values = np.array([32])  # [16, 32, 64)
    use_gat_values = np.array([True])
    heads_values = np.array([1, 2, 4, 8])

    hypers = []

    for (
        hidden_dim,
        num_layers,
        dropout,
        lr,
        batch_size,
        use_gat,
        heads,
    ) in itertools.product(
        hidden_dim_values,
        num_layers_values,
        dropout_values,
        lr_values,
        batch_size_values,
        use_gat_values,
        heads_values,
    ):
        hypers.append(
            {
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "lr": lr,
                "batch_size": batch_size,
                "use_gat": use_gat,
                "heads": heads,
            }
        )

    return hypers


def train_gnn(hyperopt, search_method, force_cpu=True):
    """
    Train the GNN with optional hyperparameter optimisation (random or grid)
    """

    dataset = ShapeCFDDataset(DATASET_DIR)

    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    # Compute scaler from training set only
    target_indices = TARGET_INDICES
    scaler = compute_scaler_from_dataset(train_set, target_indices)

    # Wrap datasets with normalized targets
    train_norm = NormalizedTargetDataset(train_set, target_indices, scaler=scaler)
    val_norm = NormalizedTargetDataset(val_set, target_indices, scaler=scaler)
    test_norm = NormalizedTargetDataset(test_set, target_indices, scaler=scaler)

    # Determine pytorch device (CPU / GPU)
    if force_cpu:
        device = "cpu"
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # TODO CUDA support could be added here
    print(f"Device = {device}")

    if hyperopt > 0:

        # Random search hyperopt to get best model hypers (and the best model itself)

        if search_method == "random":
            n_evals = hyperopt
            hyper_space = sample_hypers_random(n_evals)
        elif search_method == "grid":
            hyper_space = sample_hypers_grid()
        else:
            raise NotImplementedError

        best_loss = float("inf")
        best_model = None
        best_hparams = None

        for i, hypers in enumerate(hyper_space):

            print(f"\n[Trial {i+1} / {len(hyper_space)}] Hypers: {hypers}")

            train_loader = DataLoader(
                train_norm, batch_size=int(hypers["batch_size"]), shuffle=True
            )
            val_loader = DataLoader(val_norm, batch_size=int(hypers["batch_size"]))

            model = SimpleGNN(
                hidden_dim=hypers["hidden_dim"],
                out_dim=len(SELECTED_TARGETS),
                num_layers=hypers["num_layers"],
                dropout=hypers["dropout"],
                use_gat=hypers["use_gat"],
                heads=hypers["heads"],
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=hypers["lr"])
            loss_fn = nn.MSELoss()

            model, val_loss = train_model_with_validation(
                model,
                device,
                train_loader,
                val_loader,
                optimizer,
                loss_fn,
                max_epochs=100,
                patience=10,
            )

            print(f"Val Loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model
                best_hparams = hypers

            print(f"Current best Loss: {best_loss:.4f} with hypers: {best_hparams}")

        print(f"\nBest Loss: {best_loss:.4f} with hypers: {best_hparams}")

        model = best_model

    else:

        train_loader = DataLoader(train_norm, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_norm, batch_size=32)

        model = SimpleGNN(
            hidden_dim=128,
            out_dim=len(SELECTED_TARGETS),
            num_layers=3,
            dropout=0.1,  # Enable dropout for uncertainty estimation
            use_gat=True,
            heads=4,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        loss_fn = nn.MSELoss()

        model, val_loss = train_model_with_validation(
            model,
            device,
            train_loader,
            val_loader,
            optimizer,
            loss_fn,
            max_epochs=100,
            patience=10,
        )

    test_loader = DataLoader(test_norm, batch_size=32)
    test_loss = evaluate(model, device, test_loader, nn.MSELoss())
    print(f"Test Loss: {test_loss:.4f}")

    plot_predictions_vs_targets(
        model=model,
        device=device,
        test_loader=test_loader,
        scaler=scaler,
        target_names=SELECTED_TARGETS,
    )

    return model, scaler


def train_model_with_validation(
    model,
    device,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    max_epochs=100,
    patience=10,
    check_device=False,
):
    """
    Inner training loop for a model with a fixed set of hyperparameters
    Applies early stopping based on validation loss
    """
    best_val_loss = float("inf")
    best_model = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0

        target_indices = torch.tensor(TARGET_INDICES)

        for batch in train_loader:

            if check_device:
                check_device_consistency(model, batch, verbose=False)

            batch = batch.to(device)
            pred = model(batch)
            batch_y = batch.y.view(batch.num_graphs, -1)
            loss = loss_fn(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                batch_y = batch.y.view(batch.num_graphs, -1)
                val_loss += loss_fn(pred, batch_y).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch+1} epochs.")
                break

    # Restore best model
    model.load_state_dict(best_model)
    return model, best_val_loss


def evaluate(model, device, test_loader, loss_fn):
    """
    Evaluate model performance on the test dataset
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            batch_y = batch.y.view(batch.num_graphs, -1)
            total_loss += loss_fn(pred, batch_y).item()
    return total_loss / len(test_loader)


def plot_predictions_vs_targets(model, device, test_loader, scaler, target_names):
    """
    Plot predicted vs actual values (in physical units) for each target in test set.
    Computes and displays R² for each subplot.

    Parameters:
        model: trained PyTorch model
        test_loader: DataLoader for test dataset
        scaler: fitted StandardScaler used during training
        target_names: list of target names (e.g., ["drag", "lift", "vorticity"])
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            batch_y = batch.y.view(batch.num_graphs, -1)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    # Inverse transform to physical units
    preds_physical = scaler.inverse_transform(preds)
    targets_physical = scaler.inverse_transform(targets)

    # Plot
    n_targets = len(target_names)
    fig, axes = plt.subplots(
        1, n_targets, figsize=(5 * n_targets, 5), constrained_layout=True
    )

    if n_targets == 1:
        axes = [axes]

    for i in range(n_targets):
        ax = axes[i]
        ax.scatter(targets_physical[:, i], preds_physical[:, i], alpha=0.7)
        ax.plot(
            [targets_physical[:, i].min(), targets_physical[:, i].max()],
            [targets_physical[:, i].min(), targets_physical[:, i].max()],
            "k--",
            lw=1,
        )

        r2 = r2_score(targets_physical[:, i], preds_physical[:, i])
        ax.set_title(f"{target_names[i]} (R-squared = {r2:.3f})")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.grid(True)

    plt.suptitle("Model Predictions vs True Values (in physical units)", fontsize=16)
    plt.show()


def optimise_targets(desired_targets, model=None, grid_shape=(128, 256)):
    """
    Run optimisation (using the surrogate to optimise geometry) and visualise
    """

    if model is None:

        model, scaler = load_optimised_model_and_scaler()

    optimised_vertices, optim_fun = optimise(
        desired_targets,
        model,
        scaler,
        max_evals=100,
        n_vertices=30,
        r_bounds=(5, 30),
        device="cpu",
        grid_shape=grid_shape,
    )

    # Run this set of best vertices through the surrogate again and store the (unscaled) results,
    # and also run the LBM and generate the actual results
    test_sample = generate_graph_from_vertices(optimised_vertices)
    with torch.no_grad():
        out = model(test_sample)
    pred_scaled = out.cpu().numpy()
    pred_targets = scaler.inverse_transform(pred_scaled).flatten()

    reloaded_mask, reloaded_shifted_poly = rasterize_polygon_from_grid_vertices(
        optimised_vertices, grid_shape=grid_shape
    )

    rho, u, f, _ = run_lbm(reloaded_mask, n_steps=1000, u0=0.2, frame_interval=1001)
    omega = compute_vorticity(u)
    wake_bounds = (96, 176, 38, 88)
    cfd_targets = compute_scalar_targets(
        u, f, omega, reloaded_mask, wake_bounds, rho=rho
    )

    print(f"{desired_targets=}\n{pred_targets=}\n{cfd_targets=}")

    visualize_polygon_and_mask(reloaded_shifted_poly, reloaded_mask)
    visualize_velocity(u, reloaded_mask, scale=0.1, stride=4)

    plt.show()


def optimise(
    desired_targets,
    model,
    scaler,
    max_evals=100,
    n_vertices=8,
    r_bounds=(5, 40),
    device="cpu",
    grid_shape=(128, 256),
):
    """
    Optimises polygon to match CFD target values using the trained GNN surrogate.
    Polygon is centered in physical space according to grid_shape.

    Args:
        desired_targets (np.ndarray): shape (D,), physical target values.
        model (nn.Module): trained GNN surrogate model.
        scaler (sklearn-style): fitted scaler for target normalization.
        max_evals (int): surrogate evaluations budget.
        n_vertices (int): number of polygon vertices.
        r_bounds (tuple): (min_radius, max_radius), in pixels for "spiky" shapes
        device (str): 'cpu' or 'cuda'.
        grid_shape (tuple): (height, width) of CFD grid.

    Returns:
        best_vertices (np.ndarray): (n_vertices, 2) array of 2D polygon points.
        loss (float): surrogate MSE loss on best shape.
    """
    model.to(device)

    desired_scaled = scaler.transform([desired_targets])[0]

    shape_bias = "wings"

    if shape_bias == "spiky":

        bounds = [r_bounds] * n_vertices
        angles = 2 * np.pi * np.arange(n_vertices) / n_vertices

        cx = grid_shape[1] * 0.25  # 1/4 width
        cy = grid_shape[0] * 0.5  # centered vertically

        def fitness(radii):
            x = cx + radii * np.cos(angles)
            y = cy + radii * np.sin(angles)
            vertices = np.stack([x, y], axis=1)

            try:
                data = generate_graph_from_vertices(vertices)
                data = data.to(device)
                with torch.no_grad():
                    out = model(data)
                pred_scaled = out.cpu().numpy().flatten()
            except Exception as e:
                return 1e6

            return np.mean((pred_scaled - desired_scaled) ** 2)

        result = differential_evolution(
            fitness, bounds, maxiter=max_evals, polish=True, popsize=32
        )

        best_radii = result.x
        x = cx + best_radii * np.cos(angles)
        y = cy + best_radii * np.sin(angles)
        best_vertices = np.stack([x, y], axis=1)

    elif shape_bias == "wings":

        bounds = [
            [30.0, 80.0],  # chord len
            [0.02, 0.25],  # max_camber
            [0.1, 0.8],  # camber_position
            [0.03, 0.73],  # thickness
            [0.0, 2 * np.pi],  # angle of attack
        ]

        def vertices_from_genotype(genotype):
            return polygon_to_vertices(
                generate_parametric_airfoil(
                    n_points=32,
                    chord_length=genotype[0],
                    max_camber=genotype[1],
                    camber_position=genotype[2],
                    thickness=genotype[3],
                    angle_of_attack_deg=genotype[4],
                    grid_shape=(128, 256),
                )
            )

        def fitness(genotype):

            vertices = vertices_from_genotype(genotype)
            try:
                data = generate_graph_from_vertices(vertices)
                data = data.to(device)
                with torch.no_grad():
                    out = model(data)
                pred_scaled = out.cpu().numpy().flatten()
            except Exception as e:
                return 1e6
            return np.mean((pred_scaled - desired_scaled) ** 2)

        result = differential_evolution(
            fitness,
            bounds,
            maxiter=max_evals,
            polish=False,
            popsize=64,
            recombination=0.25,
            mutation=(0.65, 1),
            tol=0.001,
        )

        best_vertices = vertices_from_genotype(result.x)
    else:
        raise NotImplementedError

    print(f"Optimisation converged after {result.nfev} evaluations {result}")

    return best_vertices, result.fun


def plot_target_distributions(dataset_dir, target_names=None):
    """
    Loads .pt files from dataset_dir, extracts `y` values, and plots KDE distributions for each target.

    Args:
        dataset_dir (str): Path to directory containing .pt PyG Data files.
        target_names (list): Optional list of target names for labeling plots.
    """
    ys = []

    for fname in sorted(os.listdir(dataset_dir)):
        if fname.endswith(".pt"):
            path = os.path.join(dataset_dir, fname)
            try:
                data = torch.load(path)
                y = data.y
                if y.ndim == 0:
                    y = y.unsqueeze(0)
                ys.append(y.numpy())
            except Exception as e:
                print(f"Skipping {fname} due to error: {e}")

    if not ys:
        print("No valid .pt files found with 'y' targets.")
        return

    y_matrix = pd.DataFrame(ys)
    n_targets = y_matrix.shape[1]

    if target_names and len(target_names) == n_targets:
        y_matrix.columns = target_names
    else:
        y_matrix.columns = [f"target_{i}" for i in range(n_targets)]

    # Plot KDE for each target
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(n_targets, 1, figsize=(6, 3 * n_targets))

    if n_targets == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        sns.kdeplot(data=y_matrix.iloc[:, i], ax=ax)
        ax.set_title(f"Distribution of {y_matrix.columns[i]}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

    plt.tight_layout()
    plt.show()


def compute_polygon_features(vertices):
    """
    Compute features from a polygon:
    - asymmetry: std of distance from centroid
    - area: shoelace formula
    - vertex_std: std of angular positions
    """
    centroid = np.mean(vertices, axis=0)
    centered = vertices - centroid
    radii = np.linalg.norm(centered, axis=1)

    asymmetry = np.std(radii)

    x, y = vertices[:, 0], vertices[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    angles = np.arctan2(centered[:, 1], centered[:, 0])
    angles = np.unwrap(angles)
    vertex_std = np.std(angles)

    return asymmetry, area, vertex_std


def track_surrogate_errors_training_data():
    model, scaler = load_optimised_model_and_scaler()
    dataset = ShapeCFDDataset(DATASET_DIR)  # physical units on targets

    track_surrogate_errors(
        model, dataset, scaler, device="cpu", target_indices=TARGET_INDICES, n_mc=50
    )


def track_surrogate_errors(
    model,
    dataset,
    scaler,
    device: str = "cpu",
    target_indices=None,
    n_mc: int = 50,
):
    """
    Adds MC-dropout uncertainty to the existing diagnostics.
    """
    model.to(device)

    # 1) One-shot MC-dropout over the whole pool (batch_size>1 for speed).
    pool_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    mean, std = (t.cpu() for t in mc_predict(pool_loader, model, n_samples=n_mc))

    # Convert to np for easy indexing
    mean_np, std_np = mean.numpy(), std.numpy()

    if target_indices is not None:
        mean_np = mean_np[:, target_indices]
        std_np = std_np[:, target_indices]

    # Optional: bring predictions *and* std into physical units
    mean_phys = scaler.inverse_transform(mean_np)
    std_phys = std_np * scaler.scale_[target_indices] if target_indices is not None else std_np * scaler.scale_

    # 2) Per-sample feature / error bookkeeping
    records = []
    single_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for idx, data in enumerate(single_loader):
        true = data.y.cpu().numpy().flatten()
        if target_indices is not None:
            true = true[target_indices]

        error = mean_squared_error(true, mean_phys[idx])

        vertices = data.x.cpu().numpy()
        asymmetry, area, vertex_std = compute_polygon_features(vertices)

        records.append(
            {
                "asymmetry": asymmetry,
                "area": area,
                "vertex_std": vertex_std,
                "error": error,
                "uncertainty": float(std_phys[idx].mean()),  # collapse dim if needed
            }
        )

    df = pd.DataFrame(records)

    # 3) Parallel coordinates (colour by error or uncertainty as you wish)
    scaled_df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)

    fig = px.parallel_coordinates(
        scaled_df,
        dimensions=["asymmetry", "area", "vertex_std", "error", "uncertainty"],
        color="uncertainty",
        color_continuous_scale=px.colors.sequential.Viridis,
        title="Surrogate Error and Uncertainty vs. Polygon Shape Features",
    )
    fig.show()

    render_top_error_shapes(df, dataset)

    # Plot error vs uncertainty scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(df['uncertainty'], df['error'], alpha=0.6)
    plt.xlabel('Uncertainty (MC-dropout std)')
    plt.ylabel('Error (MSE)')
    plt.title('Model Error vs Uncertainty')
    plt.grid(True)
    # Compute and add R-squared between uncertainty and error
    import statsmodels.api as sm
    X = sm.add_constant(df['uncertainty'])
    model = sm.OLS(df['error'], X).fit()
    r2 = model.rsquared
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', 
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    return df


def render_top_error_shapes(
    df,
    dataset,
    top_n = 49,
    grid_shape = (7, 7),
    figsize = (12, 12),
):
    """
    Now also annotates each tile with its MC‑dropout std.
    """
    top_idx = df["error"].nlargest(top_n).index
    fig, axes = plt.subplots(*grid_shape, figsize=figsize)
    axes = axes.flatten()

    for ax, i in zip(axes, top_idx):
        data = dataset[i]
        verts = data.x.cpu().numpy()
        poly = np.vstack([verts, verts[0]])

        ax.plot(poly[:, 0], poly[:, 1], "-o", markersize=2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            f"Err={df.iloc[i]['error']:.2f}\nσ={df.iloc[i]['uncertainty']:.3f}",
            fontsize=8,
        )

    plt.tight_layout()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-single",
        action="store_true",
        default=False,
        help="Run the LBM standalone",
    )
    parser.add_argument(
        "--generate-training-data",
        action="store_true",
        default=False,
        help="Generate training samples",
    )
    parser.add_argument(
        "--train-gnn",
        action="store_true",
        default=False,
        help="Train the Graph Neural Network",
    )
    parser.add_argument(
        "--optimise-targets",
        action="store_true",
        default=False,
        help="Used trained surrogate to optimise geometry",
    )
    parser.add_argument(
        "--plot-target-distributions",
        action="store_true",
        default=False,
        help="Diagnostic analysis of training dataset",
    )
    parser.add_argument(
        "--track-surrogate-errors",
        action="store_true",
        default=False,
        help="Diagnostic analysis of surrogate model errors",
    )
    parser.add_argument(
        "--sample-winglike-polys",
        action="store_true",
        default=False,
        help="Diagnostic exploration of geometric priors",
    )
    return parser.parse_args()


def main():

    args = parse_args()

    if args.run_single:
        run_single(gif_filename="animations/lbm_anim_v0.gif")

    if args.generate_training_data:
        create_training_data_from_full_cfd(
            use_multiprocessing=True,
            n_samples=1000,
            shape_type="wing",
            wake_bounds=(96, 176, 38, 88),
        )

    if args.train_gnn:
        model, scaler = train_gnn(
            hyperopt=1, search_method="grid"
        )  # <= 0 to not do hyperopt
        torch.save(model.state_dict(), "gnn_model_v7.pth")
        pickle.dump(scaler, open("gnn_scaler_v7.pkl", "wb"))
        print(f"Wrote gnn_model_v7.pth and gnn_scaler_v7.pkl")

    if args.optimise_targets:
        test_targets = {
            "drag": 0.011159762859833,
            "lift": 3.0,
            "mean_abs_wake_vorticity": 0.0031619607853942866,
            # 'max_wake_vorticity': 0.027121600561166997,
            # 'mean_velocity_wake': 0.12720911680224747,
            # 'std_velocity_wake': 0.09038716502351885,
            # 'kinetic_energy_total': 610.4620099763782
        }

        optimise_targets(
            desired_targets=[v for v in test_targets.values()],
            model=None,
            grid_shape=(128, 256),
        )

    if args.plot_target_distributions:
        plot_target_distributions(DATASET_DIR, target_names=None)

    if args.track_surrogate_errors:
        track_surrogate_errors_training_data()

    if args.sample_winglike_polys:
        sample_winglike_polys()

    plt.show()


if __name__ == "__main__":
    main()
