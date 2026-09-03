

def prepare_ml_data(ds, scaler=None, fit_scaler=False):
    """
    Prepare data for machine learning: stack & clean (scaler-aware).
    This function prepares the data in such a way that we can:
    * Fit scaler on training data only
    * Reuse it for validation & inference later
    * Preserve xarray → NumPy → torch logic
    
    Args:
        ds (xr.Dataset): Input dataset with BVB terms and bvb_regime.
        scaler (sklearn.preprocessing.StandardScaler, optional): Scaler to use. Defaults to None.
        fit_scaler (bool, optional): Whether to fit the scaler. Defaults to False.
    Returns:
        tuple: (Xv, yv, valid, stacked) where:
            Xv (np.ndarray): Feature matrix, shape (n_samples, n_features).
            yv (np.ndarray): Target vector, shape (n_samples,).
            valid (np.ndarray): Boolean mask of valid samples.
            stacked (xr.Dataset): Stacked dataset for reference.
    """
    stacked = ds.stack(sample=("time", "lat", "lon"))

    X = stacked[BVB_TERMS].to_array("feature").transpose("sample", "feature")
    y = stacked["bvb_regime"]

    valid = np.isfinite(X).all("feature") & np.isfinite(y)

    Xv = X[valid].values.astype(np.float32)
    yv = y[valid].values.astype(np.int64)

    if scaler is not None:
        if fit_scaler:
            Xv = scaler.fit_transform(Xv)
        else:
            Xv = scaler.transform(Xv)

    return Xv, yv, valid, stacked


def time_based_split(ds, train_frac=0.7):
    """
    Train-validation split (time-aware): since this is geophysical time series (time, lat, lon),
    this function splits by time (month), not randomly.
    
    Args:
        ds (xr.Dataset): Input dataset.
        train_frac (float): Fraction of data to use for training.
    
    Returns:
        tuple: (train_ds, val_ds) where:
            train_ds (xr.Dataset): Training dataset.
            val_ds (xr.Dataset): Validation dataset.
    """
    ntime = ds.sizes["time"]
    ntrain = int(train_frac * ntime)

    train_ds = ds.isel(time=slice(0, ntrain))
    val_ds   = ds.isel(time=slice(ntrain, None))

    return train_ds, val_ds


def compute_regime_weights(y, n_regimes):
    """
    Compute regime weights to handle class imbalance.
    
    Args:
        y (np.ndarray): Target vector.
        n_regimes (int): Number of regimes.
    
    Returns:
        torch.Tensor: Regime weights.
    """
    counts = np.bincount(y, minlength=n_regimes)
    weights = counts.sum() / (counts + 1e-8)
    weights = weights / weights.mean()
    
    return torch.tensor(weights, dtype=torch.float32)


def curriculum_regime_weights(y, n_regimes, epoch, warmup=10):
    """
    Compute regime weights with a curriculum learning approach.
    
    Args:
        y (np.ndarray): Target vector.
        n_regimes (int): Number of regimes.
        epoch (int): Current epoch.
        warmup (int): Number of epochs to warm up.
    
    Returns:
        torch.Tensor: Curriculum regime weights.
    """
    counts = np.bincount(y, minlength=n_regimes)
    base = counts.sum() / (counts + 1e-8)
    base /= base.mean()

    alpha = min(1.0, epoch / warmup)
    weights = (1 - alpha) + alpha * base

    return torch.tensor(weights, dtype=torch.float32)


class BVDataset(Dataset):
    """
    Dataset for BVB data.
    """
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loaders(X_train, y_train, X_val, y_val, batch_size=4096):
    """
    Create data loaders for training and validation datasets.
    
    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training targets.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation targets.
        batch_size (int): Batch size for the data loaders.
    
    Returns:
        tuple: (train_loader, val_loader) where:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
    """
    train_loader = DataLoader(BVDataset(X_train, y_train),
                              batch_size=batch_size,
                              shuffle=True)

    val_loader = DataLoader(BVDataset(X_val, y_val),
                            batch_size=batch_size,
                            shuffle=False)

    return train_loader, val_loader



class BVBMLP(nn.Module):
    """
    BVB MLP model.
    """
    def __init__(self, n_features=9, n_regimes=15, hidden=(64, 32, 16), rare_regimes=True):
        super().__init__()

        # Setep intializers
        layers = []
        in_dim = n_features
        if len(hidden) >= 3:
            if rare_regimes:
                Activation_Function = nn.SiLU
            else:
                Activation_Function = nn.GELU
        else:
            Activation_Function = nn.Tanh

        # Create the model 
        for h in hidden:
            layers += [nn.Linear(in_dim, h),
                       Activation_Function(),
                       nn.BatchNorm1d(h)]
            in_dim = h

        layers.append(nn.Linear(in_dim, n_regimes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x) # Softmax handled by loss


class EarlyStopping:
    """
    Early stopping for training. It's reusable and can be used in 
    any training loop.
    """
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


class DualCriterionScheduler:
    """
    Dual-criterion scheduler (loss + entropy): a scheduler that combines 
    validation loss and entropy for learning rate adjustment.
    """
    def __init__(self, 
                 scheduler, 
                 lambda_entropy=0.3):
        """
        scheduler: e.g. ReduceLROnPlateau
        lambda_entropy: weight of entropy penalty
        """
        self.scheduler = scheduler
        self.lambda_entropy = lambda_entropy

    def step(self, val_loss, val_entropy):
        metric = val_loss + self.lambda_entropy * val_entropy # Combined metric
        self.scheduler.step(metric)
        
        return metric


class TrainingController:
    """
    Scheduler-Early-Stopping controller: training controller for managing early stopping and model restoration.
    """
    def __init__(self, 
                 patience=20,
                 min_delta=1e-4,
                 restore_best=True,):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best

        self.best_score = np.inf
        self.best_state = None
        self.counter = 0
        self.should_stop = False

    def step(self, score, model):
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
            if self.restore_best:
                self.best_state = {k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore(self, model):
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    args:
        model: the model to train
        loader: the training data loader
        optimizer: the optimizer
        criterion: the loss function
        device: the device to run on
    returns:
        train_loss: the average training loss
        train_entropy: the average training entropy
    """
    model.train()
    train_loss, train_entropy, n_train = 0.0, 0.0, 0
    

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1).mean()
            K = probs.shape[1]
            entropy = entropy / np.log2(K) # Normalized entropy

        bs = X.size(0)
        train_loss += loss.item() * bs
        train_entropy += entropy.item() * bs
        n_train += bs

    train_loss /= n_train
    train_entropy /= n_train

    return train_loss, train_entropy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on the validation set.
    args:
        model: the model to evaluate
        loader: the validation data loader
        criterion: the loss function
        device: the device to run on
    returns:
        val_loss: the average validation loss
        val_entropy: the average validation entropy
        n_val: the number of validation samples
    """
    model.eval()
    val_loss, val_entropy, n_val = 0.0, 0.0, 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)

        logits = model(X)
        loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1).mean()
        K = probs.shape[1]
        entropy = entropy / np.log2(K) # Normalized entropy

        bs = X.size(0)
        val_loss += loss.item() * bs
        val_entropy += entropy.item() * bs
        n_val += bs

    val_loss /= n_val
    val_entropy /= n_val

    return val_loss, val_entropy


def init_history():
    """
    Initialize the training history dictionary.
    returns:
        history: a dictionary to store training metrics
    """
    history = {"epoch": [],
               "train_loss": [],
               "val_loss": [],
               "train_entropy": [],
               "val_entropy": [],
               "lr": [],}
    return history



@torch.no_grad()
def predict_probabilistic_maps(model, ds_new, scaler, device, entropy_unit="fraction"):
    """
    Predict probabilistic maps and entropy maps for a new dataset.
    entropy_unit: "fraction" -> [0, 1] or "percent" -> [0, 100]
    Args:
        model: the trained model
        ds_new: the new dataset to predict on
        scaler: the scaler used for preprocessing
        device: the device to run on
        entropy_unit: "fraction" or "percent" for entropy output
    Returns:
        pred_ds: a dataset containing probabilistic maps and entropy maps
    """

    # Stack spatial + temporal dimensions
    stacked = ds_new.stack(sample=("time", "lat", "lon"))

    X = (stacked[BVB_TERMS]
         .to_array("feature")
         .transpose("sample", "feature"))

    valid = np.isfinite(X).all("feature")

    # Scale + move to device
    Xv = scaler.transform(X[valid].values.astype(np.float32))
    Xt = torch.from_numpy(Xv).to(device)

    # Forward pass
    model = model.to(device)
    logits = model(Xt)
    probs = torch.softmax(logits, dim=1)

    # Normalized entropy
    K = probs.shape[1]
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
    entropy = entropy / np.log2(K)

    if entropy_unit == "percent":
        entropy = 100.0 * entropy

    # Allocate output arrays (preserve land mask)
    nsample = stacked.sizes["sample"]

    prob_map = np.full((nsample, K), np.nan, dtype=np.float32)
    ent_map  = np.full(nsample, np.nan, dtype=np.float32)

    prob_map[valid.values] = probs.cpu().numpy()
    ent_map[valid.values]  = entropy.cpu().numpy()

    # Unstack back to (time, lat, lon)
    prob_da = (xr.DataArray(data=prob_map,
                            dims=("sample", "regime"),
                            coords={"sample": stacked["sample"],
                                    "regime": np.arange(K),},
                            name="prob_map",attrs={'Description': ('Predictions of probability distribution over all BV regimes for each grid '
                                                                   'point and time')})
               .unstack("sample"))

    ent_da = (xr.DataArray(data=ent_map,
                           dims=("sample",),
                           coords={"sample": stacked["sample"]},
                           name="ent_map", attrs={'Description': ('Measures of probabilistic calibration and regime ambiguity, with: '
                                                                  '(1) low entropy → one regime dominates → high confidence, stable classification, '
                                                                  'and (2) igh entropy → probabilities are spread across regimes → ambiguous or transitional dynamics.')})
              .unstack("sample"))

    pred_ds = xr.merge([prob_da, ent_da])
    pred_ds.attrs = {'Description': ('In this BV regime classification problem, a probabilistic inference has been used. '
                                     'This means the model does not assign a single regime outright. Instead, for each grid '
                                     'point and time, it outputs a probability distribution over all BV regimes. From these probabilities, '
                                     'we compute entropy, which summarize how confident the model is. So, together, (1) Probabilistic Maps tell '
                                     'which regimes are likely, and (2) Entropy Maps tell how reliable that assignment is. This combination allows '
                                     'us to distinguish well-defined BV regimes from regions or times of dynamical transition, making the classification '
                                     'both informative and physically interpretable.'), }

    return pred_ds



class BVBRegimeMLP:
    """
    A class for managing the training and prediction of a neural network for BV regime classification.
    """
    def __init__(self, n_features=9, hidden=(64, 32, 16), n_regimes=15, device=None, rare_regimes=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"The Pipeline is now using the device: {self.device}")
        
        self.model = BVBMLP(n_features, n_regimes, hidden, rare_regimes)
        self.history = init_history()
        self.scaler = pp.StandardScaler()
        self.n_regimes = n_regimes

    def fit(self, ds, 
            n_epochs=50, batch_size=4096, 
            lr=1e-3, weight_decay=1e-5, 
            patience=16, min_delta=1e-4):
        
        train_ds, val_ds = time_based_split(ds)

        Xtrn, ytrn, _, _ = prepare_ml_data(train_ds, self.scaler, fit_scaler=True)
        Xval, yval, _, _ = prepare_ml_data(val_ds, self.scaler, fit_scaler=False)

        regime_weights = compute_regime_weights(ytrn, self.n_regimes).to(self.device)

        train_loader, val_loader = make_loaders(Xtrn, ytrn, Xval, yval, batch_size)

        

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(weight=regime_weights)
    
        # Instantiate the Dual Scheduler and the Controller
        base_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                    mode="min",            # we minimize val_loss
                                                                    factor=0.5,            # lr ← lr * 0.5
                                                                    patience=8,            # wait 10 epochs before reducing
                                                                    threshold=1e-3,        # minimum improvement to count
                                                                    cooldown=5,            # wait after lr reduction
                                                                    min_lr=1e-6,           # lower bound
                                                                    )
        scheduler = DualCriterionScheduler(base_scheduler, lambda_entropy=0.25,) 
        controller = TrainingController(patience=patience, min_delta=min_delta, restore_best=True)

        self.model.to(self.device)
        train_model(model=self.model, 
                    train_loader=train_loader, val_loader=val_loader, 
                    optimizer=optimizer, criterion=criterion, 
                    scheduler=scheduler, controller=controller,
                    device=self.device, history=self.history, n_epochs=n_epochs)

    def predict(self, ds_new, entropy_unit='fraction'):
        device = next(self.model.parameters()).device
        self.model.eval()
        pred_ds = predict_probabilistic_maps(model=self.model, ds_new=ds_new, scaler=self.scaler, 
                                             device=device, entropy_unit=entropy_unit)
        return pred_ds
    
    def save(self, path):
        torch.save(obj={"model_state": self.model.state_dict(), 
                        "scaler": self.scaler,
                        "config": {"hidden": self.model.net,
                                   "n_regimes": self.n_regimes}, 
                        "history": self.history, },
                   f=path)


    def load(self, path):
        chk = torch.load(f=path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(chk["model_state"])
        self.scaler = chk["scaler"]
        self.history = chk["history"]
        






