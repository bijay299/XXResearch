#!/bin/bash
# Repair the memorize_v2.py provenance break by REPRODUCTION rather than by hash.
#
# The recorded source hash 951970d0... resolves to no file on disk; the on-disk script is
# 39a307f8... and no pre-change copy survives (searched all of /data). A three-edit
# reconstruction of the pre-change file hashed 57cc1fb6..., so at least one further change
# exists that cannot be identified. The hash link is therefore UNRECOVERABLE.
#
# What can still be established, and is stronger than a hash: whether the script now on disk
# reproduces the delivered adapter BITWISE at the recorded seed and hyperparameters. If it
# does, the recorded script reproduces the artifact and provenance is restored in substance.
# If it does not, the three delivered backbones must carry a provenance caveat permanently.
#
# Writes to a NEW adapter dir. The delivered backbone is never touched.
set -u
cd /data/bijaypandey/mdlm_unlearn

# wait for GPU 2 to be released by the pool arms
until ! ps -eo args | grep -q '[t]rain_ord_pool.py'; do sleep 30; done

./scripts/launch_py.sh 2 D prov_repro 7200 memorize_v2.py \
  --lr 0.0002 --steps 1200 --rank 32 --seed 20261101 --batch 4 \
  --t-low 0.0 --answer-weight 1.0 --norm pinned --canvas genlen \
  --grad-clip 1.0 --threshold 0.95 --mode qualify \
  --out /data/bijaypandey/mdlm_unlearn/adapters/prov_repro_s20261101 \
  --report /data/bijaypandey/mdlm_unlearn/outputs/prov_repro_s20261101_report.json \
  > outputs/log_prov_repro.txt 2>&1

/data/bijaypandey/llada_smoke/venv_llada/bin/python - <<'EOF' >> outputs/log_prov_repro.txt 2>&1
import hashlib, json, os
P = "/data/bijaypandey/mdlm_unlearn"
old = json.load(open(f"{P}/outputs/memorize_v2_s20261101_report.json"))
rec = old["adapter"]["sha256_per_file"]["adapter_model.safetensors"]
new = f"{P}/adapters/prov_repro_s20261101/adapter_model.safetensors"
if not os.path.exists(new):
    print("REPRODUCTION DID NOT PRODUCE AN ADAPTER"); raise SystemExit(1)
h = hashlib.sha256(open(new, "rb").read()).hexdigest()
print("recorded adapter sha256    :", rec)
print("reproduced adapter sha256  :", h)
print("VERDICT:", "BITWISE REPRODUCED -- provenance restored in substance"
      if h == rec else "NOT REPRODUCED -- provenance caveat is permanent")
json.dump({"study": "memorize_v2_provenance_repair_by_reproduction",
           "recorded_source_sha256": "951970d094c4bbfab6bf7721c4a293dd219ef836c25fa1f51b06e1351f45d1db",
           "on_disk_source_sha256": hashlib.sha256(open(f"{P}/scripts/memorize_v2.py","rb").read()).hexdigest(),
           "recorded_adapter_sha256": rec, "reproduced_adapter_sha256": h,
           "bitwise_reproduced": h == rec},
          open(f"{P}/outputs/prov_repair.json", "w"), indent=2)
EOF
echo "[queue_provenance] done"
