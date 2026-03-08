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
        "q": torch.tensor(df["q_event"].values, dtype=torch.float32),
        "y": torch.tensor(df["event_id"].values, dtype=torch.long),
        "dt":torch.tensor(df["dtk_l"].values, dtype=torch.float32)    
    }
    if use_last_event:
        data["last_event"] = torch.tensor(df["last_event_id"].values, dtype=torch.long)
    if use_hour:
        data["hour_id"] = torch.tensor(df["hour_id"].values, dtype=torch.long)
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