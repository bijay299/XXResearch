"""FROZEN protocol constants for cpr_direct_agreement_controls_v1 (committed before any numerical
loss test, detector inference or FLUX output). Nothing here is tuned from outcomes."""
import hashlib

from cprd2.common import PROJECT_ID

ARMS = ["native_gem", "cpr", "centered_radial", "truncated_anchor", "anchor_only", "axial_margin"]
NON_CPR_COMPARATORS = [a for a in ARMS if a != "cpr"]          # the 5 comparators

DEV_TARGET = "refrigerator"
HELDOUT_TARGETS = ["toilet", "bed"]
PANELS = {
    "refrigerator": ["oven", "sink", "bus", "train"],
    "toilet": ["couch", "potted plant", "bus", "train"],
    "bed": ["couch", "potted plant", "bus", "train"],
}
SCORED_CLASSES = ["refrigerator", "toilet", "bed", "oven", "sink", "bus", "train", "couch", "potted plant"]

# ---- native settings (recorded; trainer defaults preserved unless listed here) ----
SETTINGS = {
    "adapter": {"adapter_mode": "lora", "subset_name": "qk_dual", "r": 16, "lora_alpha": 16, "lora_dropout": 0.1},
    "eta": 1.0, "gamma": 100.0, "optimizer": "torch.optim.AdamW (native default betas/eps/weight_decay)",
    "learning_rates": {"lr_1e-3": 1e-3, "lr_5e-4": 5e-4},
    "max_step_budget": 250, "max_parallel_timesteps": 1, "backbone_dtype": "bfloat16", "loss_dtype": "float32",
    "partial_denoising_end_at_timestep": 10, "partial_denoising_mode": "student",
    "partial_denoising_guidance_scale": 3.0, "use_inference_timesteps": True,
    "partial_denoising_inference_timesteps": None, "anchors": [""], "empty_concept": "",
    "gradient_checkpointing": True, "use_wandb": False, "prompt_augmentation": None,
    "evaluation": {"height": 512, "width": 512, "num_inference_steps": 28, "guidance_scale": 3.5,
                   "max_sequence_length": 512, "batch_size": 1, "output_type": "pil"},
    "n_latents_per_window": 11,
}
# processed-state checkpoints: first complete optimizer update reaching 25% / 50% of the 250-state
# cap, plus the terminal endpoint. 11-state windows -> 66 (update 6), 132 (update 12), 250 (update 23).
CAP = 250
CKPT_FRACTIONS = {"pct25": 0.25, "pct50": 0.50}
CHECKPOINT_NAMES = ["pct25", "pct50", "terminal"]
CHECKPOINT_STATES = {"pct25": 66, "pct50": 132, "terminal": 250}
CHECKPOINT_UPDATES = {"pct25": 6, "pct50": 12, "terminal": 23}

# ---- seeds ----
TRAIN_SEED_DEV = 5101
TRAIN_SEEDS_HELDOUT = [5201, 5202]
EVAL_BASE = 9_000_000
TECH_BASE = 9_900_000
BOOTSTRAP_SEED = int(hashlib.sha256(PROJECT_ID.encode()).hexdigest()[:8], 16)
N_BOOT = 10000
BOOT_PCT = [2.5, 97.5]

# ---- gate capture: predeclared processed-step indices on the first planned native dev trajectory
CAPTURE_PROCESSED_INDICES = [0, 7]
GATE_NATIVE_TRAJ = "dev.native_gem.lr_1e-3"

# ---- tolerances (direct native-agreement gate) ----
TOL = {
    "loss": {"rtol": 1e-5, "atol": 1e-6},
    "grad": {"rtol": 1e-4, "atol": 1e-7},
}

# ---- detectors ----
THRESHOLD = 0.30
IOU_MATCH = 0.50
QUAL = {"n_pos": 20, "n_neg": 20, "min_pos_boxmatch": 16, "min_neg_clean": 16, "n_confounders": 10,
        "min_box_frac": 0.05}
DETECTORS = {
    "rtdetr_r50vd": {"repo": "PekingU/rtdetr_r50vd", "revision": "df939e661d8c52e80608d1ec566561aabd25a4e7",
                     "role": "primary"},
    "detr_resnet50_no_timm": {"repo": "facebook/detr-resnet-50", "revision": "70120ba84d68ca1211e007c4fb61d0cd5424be54",
                              "role": "secondary"},
}
PRIMARY_DET, SECONDARY_DET = "rtdetr_r50vd", "detr_resnet50_no_timm"

# ---- eligibility / selection / decision constants ----
DEV_ELIG = {"target_min": 10, "allowed_min": 10, "per_allowed_class_min": 2, "n_target": 12, "n_per_allowed": 3}
HO_ELIG = {"target_min": 16, "per_allowed_class_min": 4, "n_target": 20, "n_per_allowed": 5}
S_FLOOR = 0.80
DELTA_S_MARGIN = -0.05          # tradeoff: DeltaS >= -0.05 (CPR vs comparator)
DELTA_R_PROMISING = 0.10
NA_DELTA_S_MAX = 0.05
NA_DELTA_R_MAX = 0.10           # strict <
COLLAPSE_STD_THRESHOLD = 1.0 / 255.0

# ---- image budgets ----
IMAGES = {"primary_total": 1928, "technical_allowance": 272, "total_cap": 2200,
          "dev_edited": 864, "dev_m0": 24, "ho_edited": 960, "ho_m0": 80}

# ---- frozen prompt templates ----
TEMPLATES = {
    "refrigerator": [
        "a photo of a refrigerator in a kitchen", "a stainless steel refrigerator next to a counter",
        "a white refrigerator in a small apartment kitchen", "a refrigerator with magnets on the door",
        "a close-up photo of a refrigerator", "a tall refrigerator beside wooden cabinets",
        "a refrigerator in a modern kitchen", "an old refrigerator in a farmhouse kitchen",
        "a refrigerator with its door closed", "a refrigerator in a bright kitchen",
        "a black refrigerator in a clean kitchen", "a refrigerator next to a dining area",
    ],
    "toilet": [
        "a photo of a toilet in a bathroom", "a white toilet beside a tiled wall",
        "a toilet in a small restroom", "a clean toilet in a modern bathroom",
        "a close-up photo of a toilet", "a toilet next to a sink",
        "a toilet in a public restroom", "an old toilet in a rustic bathroom",
        "a toilet with the lid closed", "a toilet in a bright bathroom",
        "a toilet beside a bathroom window", "a toilet in a narrow bathroom",
        "a toilet on a tiled floor", "a toilet in a hotel bathroom",
        "a toilet with a wooden seat", "a toilet next to a towel rack",
        "a toilet in a quiet restroom", "a toilet in a renovated bathroom",
        "a toilet against a white wall", "a toilet in a cozy bathroom",
    ],
    "bed": [
        "a photo of a bed in a bedroom", "a neatly made bed with white sheets",
        "a bed beside a window", "a wooden bed in a small room",
        "a close-up photo of a bed", "a bed with a patterned blanket",
        "a bed in a hotel room", "an unmade bed in morning light",
        "a bed with several pillows", "a bed in a bright bedroom",
        "a double bed against a wall", "a bed with a wooden headboard",
        "a bed in a guest room", "a bed next to a nightstand",
        "a bed in a cozy attic room", "a bed with a gray comforter",
        "a bed in a modern bedroom", "a bed under a skylight",
        "a bed in a rustic cabin", "a bed with a quilt",
    ],
    "oven": [
        "a photo of an oven in a kitchen", "a stainless steel oven with a glass door",
        "a built-in oven beside cabinets",
    ],
    "sink": [
        "a photo of a sink in a kitchen", "a stainless steel sink under a window",
        "a bathroom sink with a mirror above it",
    ],
    "couch": [
        "a photo of a couch in a living room", "a gray couch beside a lamp",
        "a leather couch in a bright room", "a couch with colorful cushions",
        "a couch against a white wall",
    ],
    "potted plant": [
        "a photo of a potted plant on a table", "a potted plant beside a window",
        "a large potted plant in a living room", "a small potted plant on a shelf",
        "a potted plant on a wooden floor",
    ],
    # 8 prompts: indices 0-2 used in development, 3-7 in held-out (disjoint across stages)
    "bus": [
        "a photo of a bus on a city street", "a red bus at a bus stop",
        "a bus driving down a road", "a yellow school bus parked by a field",
        "a bus on a rainy street", "a double-decker bus in traffic",
        "a bus in front of a building", "a white bus on a highway",
    ],
    "train": [
        "a photo of a train at a station", "a train on a railway track",
        "a passenger train crossing a bridge", "a freight train in the countryside",
        "a train arriving at a platform", "a commuter train on a sunny day",
        "a train beside a platform", "a train passing through a small town",
    ],
}
# prompt index offsets for classes shared between dev and held-out panels
PROMPT_OFFSET = {("dev", "bus"): 0, ("dev", "train"): 0, ("ho", "bus"): 3, ("ho", "train"): 3}


def prompt_index(stage, cls, j):
    return PROMPT_OFFSET.get((stage, cls), 0) + j


def ev_seed(*parts, base=EVAL_BASE):
    return base + (int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) % 900000)
