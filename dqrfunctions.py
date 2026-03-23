import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

"""
--------------------------------------------------------------
DATA STRUCTURES AND MODEL DEFINITION (CLASS AND LOSS FUNCTION)
--------------------------------------------------------------
""" 

# Build training dataset for NN model in PyTorch
def build_dataset(df, use_hour=False, use_last_event=False):
    data = {
        "q": torch.tensor(df["q"].values, dtype=torch.float32),
        "y": torch.tensor(df["event_id"].values, dtype=torch.long),
        "dt":torch.tensor(df["dtk_l"].values, dtype=torch.float32)    
    }
    if use_last_event:
        data["last_event"] = torch.tensor(df["last_event_id"].values, dtype=torch.long)
    if use_hour:
        data["hour_id"] = torch.tensor(df["hour_last_event"].values, dtype=torch.long)
    return data


# Neural network class used for DQR models
class DQRNet(nn.Module):
    # Declaration of neural network class
    def __init__(
        self,
        hidden_layers=(128,32),
        use_hour=False,
        use_last_event=False,
        hour_num_classes=None,
        emb_dim=2
    ):
        super().__init__()
        
        self.use_hour = use_hour
        self.use_last_event = use_last_event
        self.emb_dim = emb_dim
        self.hidden_layers = hidden_layers
        
        input_dim = 1 # qk only
        
        # Embedding the last event features in dimension 2
        if self.use_last_event:
            self.last_event_emb = nn.Embedding(num_embeddings=3, embedding_dim=emb_dim)
            input_dim += emb_dim
        
        # Embedding the hours features in dimension 2
        if self.use_hour:
            if hour_num_classes is None:
                raise ValueError("hour_num_classes must be provided when use_hour=True")
            self.hour_emb = nn.Embedding(num_embeddings=hour_num_classes, embedding_dim=emb_dim)
            input_dim += emb_dim
        
        # Neural network implementation
        self.net = nn.Sequential(
            # First hidden layer input -> 128
            nn.Linear(input_dim, hidden_layers[0]),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_layers[0]),
            
            # Second hidden layer 128 -> 32
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_layers[1]),
            
            # Output layer
            nn.Linear(hidden_layers[1], 3),
            nn.ReLU()
        )
        
        self.input_dim = input_dim
        self.history = {
            "train_loss": [],
            "val_loss": []
        }
    
    # Forward pass
    def forward(self, q, last_event=None, hour=None):
        x_parts = [q.unsqueeze(1).float()] # (batch,1)
        
        if self.use_last_event:
            if last_event is None:
                raise ValueError("last_event must be provided when use_last_event=True")
            x_parts.append(self.last_event_emb(last_event))
        
        if self.use_hour:
            if hour is None:
                raise ValueError("hour must be provided when use_hour=True")
            x_parts.append(self.hour_emb(hour))
            
        x = torch.cat(x_parts, dim=1)
        return self.net(x)
    
    def plot_history(self, model_name="Default"):
        if len(self.history["train_loss"]) == 0 and len(self.history["val_loss"]) == 0:
            print("No training history stored in this model.")
            return
        
        plt.figure(figsize=(6, 4))
        plt.plot(self.history["train_loss"], ".-b", label="Training loss")
        plt.plot(self.history["val_loss"], ".-r", label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training history : DQR {model_name}")
        plt.grid()
        plt.legend()
        plt.show()


# Negative log likelihood derivation of loss function
def dqr_loss(lambdas, event_id, dt):
    # lambdas shape (batch, 3); Lambda = lambda_L + lambda_C + lambda_M
    total_intensity = lambdas.sum(dim=1)
    idx = torch.arange(event_id.size(0), device=event_id.device) # choix du lambda de l'évènement
    chosen_lambda = lambdas[idx, event_id] # shape (batch, )
    # Est ce qu'on prend la moyenne sur le batch ou la somme => je pense ça change rien
    loss = (total_intensity * dt - torch.log(chosen_lambda + 1e-8)).sum()
    return loss


"""
-----------------------------------------------------
MODEL TRAINING ROUTINES AND SAVING/RECOVERY FUNCTIONS
-----------------------------------------------------
""" 

# Helper functions used to assess the performance across 1 epoch
@torch.no_grad()
def eval_epoch(model: DQRNet, loader):
    # Function to evaluate loss across 
    model.eval()
    total_loss, n = 0.0, 0
    
    # iterate across batches
    for batch in loader:
        # Hour + Last Event model
        if model.use_hour and model.use_last_event:
            qb, eb, hb, yb, dtb = batch
            lambdas = model(qb, last_event=eb, hour=hb)
            bs = qb.size(0)
        
        # Last Event model
        elif model.use_last_event:
            qb, eb, yb, dtb = batch
            lambdas = model(qb, last_event=eb)
            bs = qb.size(0)
            
        # Hour model
        elif model.use_hour:
            qb, hb, yb, dtb = batch
            lambdas = model(qb, hour=hb)
            bs = qb.size(0)
        
        # Vanilla model qk only
        else:
            qb, yb, dtb = batch
            lambdas = model(qb)
            bs = qb.size(0)
            
        loss = dqr_loss(lambdas, yb, dtb)
        total_loss += loss.item()
        n += bs
    
    return total_loss / max(n,1)


# Training routine for the DQR Model
def train_model(
    data,
    epochs=300,
    batch_size=256,
    val_frac=0.2,
    patience=30,
    min_delta=0.0,
    seed=None,
    use_hour=False,
    use_last_event=False,
    hour_num_classes=None
):
    n = data["q"].size(0)

    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g)
    else:
        idx = torch.randperm(n)

    n_val = int(val_frac * n)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    # Build TensorDataset depending on model type
    if use_hour and use_last_event:
        train_dataset = TensorDataset(
            data["q"][tr_idx],
            data["last_event"][tr_idx],
            data["hour_id"][tr_idx],
            data["y"][tr_idx],
            data["dt"][tr_idx]
        )
        val_dataset = TensorDataset(
            data["q"][val_idx],
            data["last_event"][val_idx],
            data["hour_id"][val_idx],
            data["y"][val_idx],
            data["dt"][val_idx]
        )

    elif use_last_event:
        train_dataset = TensorDataset(
            data["q"][tr_idx],
            data["last_event"][tr_idx],
            data["y"][tr_idx],
            data["dt"][tr_idx]
        )
        val_dataset = TensorDataset(
            data["q"][val_idx],
            data["last_event"][val_idx],
            data["y"][val_idx],
            data["dt"][val_idx]
        )

    elif use_hour:
        train_dataset = TensorDataset(
            data["q"][tr_idx],
            data["hour_id"][tr_idx],
            data["y"][tr_idx],
            data["dt"][tr_idx]
        )
        val_dataset = TensorDataset(
            data["q"][val_idx],
            data["hour_id"][val_idx],
            data["y"][val_idx],
            data["dt"][val_idx]
        )

    else:
        train_dataset = TensorDataset(
            data["q"][tr_idx],
            data["y"][tr_idx],
            data["dt"][tr_idx]
        )
        val_dataset = TensorDataset(
            data["q"][val_idx],
            data["y"][val_idx],
            data["dt"][val_idx]
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = DQRNet(
        hidden_layers=(128, 32),
        use_hour=use_hour,
        use_last_event=use_last_event,
        hour_num_classes=hour_num_classes,
        emb_dim=2
    )

    # Optimizer and learning rate parameters to fit paper's settings
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    n_train = len(tr_idx)
    steps_per_epoch = math.ceil(n_train / batch_size)
    step_size_up = 4 * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=1e-5,
        max_lr=1e-3,
        step_size_up=step_size_up,
        mode="triangular",
        cycle_momentum=False
    )

    # Save model training history as attribute of class
    model.history = {"train_loss": [], "val_loss": []}

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    # Training and propagation routine
    for epoch in range(epochs):
        model.train()
        total_loss, count = 0.0, 0

        for batch in train_loader:
            optimizer.zero_grad()

            if use_hour and use_last_event:
                qb, eb, hb, yb, dtb = batch
                lambdas = model(qb, last_event=eb, hour=hb)
                bs = qb.size(0)

            elif use_last_event:
                qb, eb, yb, dtb = batch
                lambdas = model(qb, last_event=eb)
                bs = qb.size(0)

            elif use_hour:
                qb, hb, yb, dtb = batch
                lambdas = model(qb, hour=hb)
                bs = qb.size(0)

            else:
                qb, yb, dtb = batch
                lambdas = model(qb)
                bs = qb.size(0)

            loss = dqr_loss(lambdas, yb, dtb)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            count += bs

        train_loss = total_loss / max(count, 1)
        val_loss = eval_epoch(model, val_loader)

        model.history["train_loss"].append(train_loss)
        model.history["val_loss"].append(val_loss)

        # early stopping
        if best_val - val_loss > min_delta:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


from pathlib import Path
# Function to train model or recover a saved model from previous session
def get_or_train_model(
    model_path,
    data,
    compute_model=False,
    epochs=300,
    batch_size=None,
    val_frac=0.2,
    patience=200,
    min_delta=0.0,
    seed=None,
    use_hour=False,
    use_last_event=False,
    hour_num_classes=None
):
    
    model_path = Path(model_path)
    
    if batch_size is None:
        N = len(data["q"])
        batch_size = max(32, min(2048, N//100))

    if compute_model or not model_path.exists():

        model = train_model(
            data=data,
            epochs=epochs,
            batch_size=batch_size,
            val_frac=val_frac,
            patience=patience,
            min_delta=min_delta,
            seed=seed,
            use_hour=use_hour,
            use_last_event=use_last_event,
            hour_num_classes=hour_num_classes
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "history": model.history,
            "input_dim": model.input_dim,
            "hidden_layers": model.hidden_layers,
            "use_hour": model.use_hour,
            "use_last_event": model.use_last_event,
            "emb_dim": model.emb_dim,
            "hour_num_classes": hour_num_classes
        }

        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, model_path)

    else:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

        model = DQRNet(
            hidden_layers=tuple(ckpt["hidden_layers"]),
            use_hour=ckpt["use_hour"],
            use_last_event=ckpt["use_last_event"],
            hour_num_classes=ckpt["hour_num_classes"],
            emb_dim=ckpt["emb_dim"]
        )

        model.load_state_dict(ckpt["model_state_dict"])
        model.history = ckpt["history"]

    model.eval()
    return model



"""
-------------------------------
ANALYSIS AND PLOTTING FUNCTIONS
-------------------------------
""" 

# Transition matrices functions
def transition_real(event_id, last_event_id, n_events=3):
    T = np.zeros((n_events, n_events), dtype=float)
    for i in range(n_events):
        mask = (last_event_id == i)
        if mask.sum() == 0:
            continue
        counts = np.bincount(event_id[mask], minlength=n_events).astype(float)
        T[i, :] = counts / counts.sum()
    return T

@torch.no_grad()
def transition_dqr(model, data, last_event_id, n_events=3, batch_size=65536):
    model.eval()

    N = data["q"].shape[0]
    probs = torch.empty((N, n_events), dtype=torch.float32)

    for s in range(0, N, batch_size):
        q_batch = data["q"][s:s+batch_size]

        if model.use_hour and model.use_last_event:
            hour_batch = data["hour_id"][s:s+batch_size]
            last_event_batch = data["last_event"][s:s+batch_size]
            lambdas = model(q_batch, last_event=last_event_batch, hour=hour_batch)

        elif model.use_last_event:
            last_event_batch = data["last_event"][s:s+batch_size]
            lambdas = model(q_batch, last_event=last_event_batch)

        elif model.use_hour:
            hour_batch = data["hour_id"][s:s+batch_size]
            lambdas = model(q_batch, hour=hour_batch)

        else:
            lambdas = model(q_batch)

        p = lambdas / (lambdas.sum(dim=1, keepdim=True) + 1e-12)
        probs[s:s+batch_size] = p.cpu()

    probs = probs.numpy()
    last = last_event_id.cpu().numpy() if torch.is_tensor(last_event_id) else np.asarray(last_event_id)

    T = np.zeros((n_events, n_events), dtype=float)

    for i in range(n_events):
        mask = (last == i)
        if mask.sum() == 0:
            continue
        T[i, :] = probs[mask].mean(axis=0)

    return T


import matplotlib.patheffects as pe
def plot_two_heatmaps(T_real, T_dqr, labels=None, dqr_title=r"DQR"):
    if labels is None:
        labels = [str(i) for i in range(T_real.shape[0])]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    titles = [
        r"Real : $\mathbb{P}(\eta_k=j \mid \eta_{k-1}=i)$",
        dqr_title
    ]

    for ax, T, title in zip(axes, [T_real, T_dqr], titles):
        im = ax.imshow(T, aspect="equal")
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)

        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                ax.text(
                    j, i, f"{T[i,j]:.2f}",
                    ha="center", va="center",
                    color="white", fontweight="bold", fontsize=9,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="black")]
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return fig
import matplotlib.patheffects as pe

def plot_three_heatmaps(T_real, T_qr, T_dqr, labels=None, dqr_title=r"DQR"):

    if labels is None:
        labels = [str(i) for i in range(T_real.shape[0])]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    titles = [
        r"Real",
        r"QR",
        dqr_title
    ]
    
    vmin = min([T_real.flatten().min(), T_qr.flatten().min(), T_dqr.flatten().min()])
    vmax = max([T_real.flatten().max(), T_qr.flatten().max(), T_dqr.flatten().max()])
    
    for ax, T, title in zip(axes, [T_real, T_qr, T_dqr], titles):

        im = ax.imshow(T, aspect="equal", vmin=vmin, vmax=vmax)

        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)

        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                ax.text(
                    j, i, f"{T[i,j]:.2f}",
                    ha="center", va="center",
                    color="white", fontweight="bold", fontsize=9,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="black")]
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return fig


@torch.no_grad()
def compute_hourly_intensity(model, data, hour_values):
    model.eval()

    # Forward according to model type
    if model.use_hour and model.use_last_event:
        lambdas = model(
            data["q"],
            last_event=data["last_event"],
            hour=data["hour_id"]
        )
    elif model.use_hour:
        lambdas = model(
            data["q"],
            hour=data["hour_id"]
        )
    elif model.use_last_event:
        lambdas = model(
            data["q"],
            last_event=data["last_event"]
        )
    else:
        lambdas = model(data["q"])

    lambdas = lambdas.cpu().numpy()
    
    # Il faut récupérer les dt pour pondérer l'intensité
    dts = data["dt"].cpu().numpy()

    df_plot = pd.DataFrame({
        "hour_id": hour_values,
        "lambda_limit": lambdas[:, 0],
        "lambda_cancel": lambdas[:, 1],
        "lambda_trade": lambdas[:, 2],
        "dt": dts  # Ajout de l'intervalle de temps
    })

    # Calcul du nombre d'événements attendus par le modèle (lambda * dt)
    df_plot["expected_trades"] = df_plot["lambda_trade"] * df_plot["dt"]
    df_plot["expected_limits"] = df_plot["lambda_limit"] * df_plot["dt"]
    df_plot["expected_cancels"] = df_plot["lambda_cancel"] * df_plot["dt"]

    # Agrégation pondérée par le temps
    grouped = df_plot.groupby("hour_id")
    
    # Intensité horaire = (Somme des événements attendus) / (Somme des dt)
    hourly = grouped["expected_trades"].sum() / grouped["dt"].sum()

    return hourly

def compute_real_hourly_intensity(df, trade_id=2):
    df_real = df.copy()

    # total observation time per hour
    total_time = df_real.groupby("hour_last_event")["dtk_l"].sum()

    # number of trade events per hour
    n_trades = (df_real["event_id"] == trade_id).groupby(df_real["hour_last_event"]).sum()

    # empirical intensity = count / total time
    lambda_real = n_trades / total_time

    return lambda_real


"""
COMPARISON OF MODEL 'S FUNCTIONS
"""

@torch.no_grad()
def predict_lambdas(model, data, batch_size=4096):
    model.eval()
    N = data["q"].shape[0]
    out = torch.empty((N, 3), dtype=torch.float32)

    for s in range(0, N, batch_size):
        q_batch = data["q"][s:s+batch_size]

        if model.use_hour and model.use_last_event:
            hour_batch = data["hour_id"][s:s+batch_size]
            last_event_batch = data["last_event"][s:s+batch_size]
            lambdas = model(q_batch, last_event=last_event_batch, hour=hour_batch)

        elif model.use_last_event:
            last_event_batch = data["last_event"][s:s+batch_size]
            lambdas = model(q_batch, last_event=last_event_batch)

        elif model.use_hour:
            hour_batch = data["hour_id"][s:s+batch_size]
            lambdas = model(q_batch, hour=hour_batch)

        else:
            lambdas = model(q_batch)

        out[s:s+batch_size] = lambdas.cpu()

    return out.numpy()


def balanced_accuracy_numpy(y_true, y_pred, n_classes=3):
    recalls = []
    for c in range(n_classes):
        mask = (y_true == c)
        if mask.sum() == 0:
            continue
        recalls.append((y_pred[mask] == c).mean())
    return float(np.mean(recalls))


def evaluate_dqr_model(model, data, eps=1e-12):
    lambdas = predict_lambdas(model, data)                 # shape (N,3)
    y_true = data["y"].cpu().numpy()
    dt_true = data["dt"].cpu().numpy()

    Lambda = lambdas.sum(axis=1)                          # total intensity
    chosen_lambda = lambdas[np.arange(len(y_true)), y_true]

    # 1) mean log-likelihood per observation (higher is better)
    loglik = np.mean(np.log(chosen_lambda + eps) - Lambda * dt_true)

    # 2) next-event prediction
    y_pred = np.argmax(lambdas, axis=1)
    bal_acc = balanced_accuracy_numpy(y_true, y_pred, n_classes=3)

    # 3) time-to-next-event relative difference (WMAPE)
    dt_pred = 1.0 / np.maximum(Lambda, eps)
    
    # Calcul de la Mean Absolute Error (MAE)
    mae = np.mean(np.abs(dt_pred - dt_true))
    # Division par le temps d'attente moyen réel
    mean_true = np.mean(dt_true)
    
    rel_diff = (mae / np.maximum(mean_true, eps)) * 100.0
    return {
        "loglik": float(loglik),
        "bal_acc": float(bal_acc),
        "rel_diff_pct": float(rel_diff),
    }
    
def plot_model_comparison(results):
    model_names = list(results.keys())
    model_colors = {
        "Vanilla": "#1f77b4",
        "Hour": "#ff7f0e",
        "Last event": "#2ca02c",
        "Hour + Last event": "#d62728",
    }
    colors= [model_colors[m] for m in model_names]

    loglik = [results[m]["loglik"] for m in model_names]
    balacc = [results[m]["bal_acc"] for m in model_names]
    reldiff = [results[m]["rel_diff_pct"] for m in model_names]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

    axes[0].bar(model_names, loglik, color=colors)
    axes[0].set_title("Log-likelihood")
    axes[0].set_ylabel("Mean log-likelihood")
    axes[0].set_xlabel("(Higher is better)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(model_names, balacc, color=colors)
    axes[1].set_title("Next event prediction\nBalanced accuracy")
    axes[1].set_ylabel("Balanced accuracy")
    axes[1].set_xlabel("(Higher is better)")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(model_names, reldiff, color=colors)
    axes[2].set_title("Time to next event\n(Relative Difference (%))")
    axes[2].set_ylabel("Relative difference (%)")
    axes[2].set_ylim([min(reldiff)-5, max(reldiff)+5])
    axes[2].set_xlabel("(Lower is better)")
    axes[2].grid(axis="y", alpha=0.3)

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)

    plt.show()