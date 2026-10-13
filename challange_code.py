
"""EP55 - Neural Architecture Search: AutoKeras designs a CNN for the digits dataset, no hand-tuning.
    python ep55_autokeras.py     # search 12 architectures (about 3-4 minutes on a CPU), test the best one
pip install tensorflow-cpu==2.20.0 keras==3.10.0 autokeras==3.0.0 scikit-learn numpy
"""
import numpy as np
import keras
import autokeras as ak
from sklearn.datasets import load_digits
# ── 1. CONFIG ─────────────────────────────────────────────────────────────
SEED = 0
TRIALS = 12                          # architectures AutoKeras may try
EPOCHS = 15                          # training epochs for EACH trial
VAL = 0.2                            # share of the training images held out to score a trial
# ── 2. DATA: 8x8 digits, a fixed 1,437 / 360 split ────────────────────────
def load():
    x, y = load_digits(return_X_y=True)
    x = (x / 16.0).reshape(-1, 8, 8, 1).astype("float32")
    perm = np.random.RandomState(0).permutation(len(x))
    cut = int(0.8 * len(x))
    return x[perm[:cut]], y[perm[:cut]], x[perm[cut:]], y[perm[cut:]]

# ── 3. SEARCH SPACE: we choose WHAT may be built, AutoKeras chooses HOW ───
def search(xtr, ytr, seed):
    inp = ak.ImageInput()
    out = ak.ImageBlock(block_type="vanilla", normalize=True, augment=False)(inp)   # plain conv nets only
    out = ak.ClassificationHead()(out)
    nas = ak.AutoModel(inputs=inp, outputs=out, max_trials=TRIALS, objective="val_accuracy",
                       tuner="bayesian", seed=seed, overwrite=True, directory=f"ak_seed{seed}")
    nas.fit(xtr, ytr, epochs=EPOCHS, validation_split=VAL, verbose=0)
    return nas

# ── 4. READ THE SEARCH: one row per architecture that was tried ───────────
def trials(nas):
    rows = []
    for t in sorted(nas.tuner.oracle.trials.values(), key=lambda t: int(t.trial_id)):
        hp = {k.split("/")[-1]: v for k, v in t.hyperparameters.values.items()}
        size = nas.tuner.load_model(t).count_params()             # reload that trial's model, count its weights
        rows.append(dict(val=t.score, blocks=hp["num_blocks"], layers=hp["num_layers"], kernel=hp["kernel_size"],
                         separable=hp["separable"], opt=hp["optimizer"], params=size))
    return rows

# ── 5. BASELINE: the CNN a person would write by hand, same data and budget ──
def by_hand(xtr, ytr):
    m = keras.Sequential([keras.layers.Input((8, 8, 1)),
                          keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
                          keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
                          keras.layers.GlobalMaxPooling2D(), keras.layers.Dense(10, activation="softmax")])
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    m.fit(xtr, ytr, epochs=EPOCHS, validation_split=VAL, verbose=0)
    return m

def accuracy(model, x, y):
    return float((model.predict(x, verbose=0).argmax(1) == y).mean())

def run(seed=SEED):
    keras.utils.set_random_seed(seed)
    xtr, ytr, xte, yte = load()
    nas = search(xtr, ytr, seed)
    best = nas.export_model()
    hand = by_hand(xtr, ytr)
    return dict(nas=accuracy(best, xte, yte), hand=accuracy(hand, xte, yte), nas_params=best.count_params(),
                hand_params=hand.count_params(), trials=trials(nas))

if __name__ == "__main__":
    r = run()
    for i, t in enumerate(r["trials"]):
        print(f"trial {i:2d}  val acc {t['val']:.3f}  blocks {t['blocks']} x layers {t['layers']}  "
              f"kernel {t['kernel']}  separable {t['separable']}  optimizer {t['opt']}  params {t['params']:,}")
    print(f"test accuracy | AutoKeras best: {r['nas']:.3f} ({r['nas_params']:,} params)"
          f" | hand-made CNN: {r['hand']:.3f} ({r['hand_params']:,} params)")
