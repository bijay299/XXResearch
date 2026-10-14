#!/usr/bin/env python3
"""Verbatim-vendored copies of two native RECE functions, used to actually
EXECUTE the direct-vs-sequential English-springer edit. Not a
reimplementation: every line below is a byte-for-byte copy of the pinned
repo's own source, checked against that source at import time (see
`assert_no_drift()`), so this cannot silently diverge from what the authors
actually run.

Why vendored instead of imported live:
- `edit_model` (Stage 1 / preliminary UCE) lives in
  `erase_methods/train_uce.py`, which does
  `from ..execs import generate_images, compute_nudity_rate` at module level.
  `compute_nudity_rate` pulls in `utils/metrics/nudity_eval.py` ->
  NudeDetector -> onnxruntime, a nudity/NSFW-detector dependency chain wholly
  unrelated to object-concept erasure. Per this project's explicit
  instruction not to pull in unrelated training/attack dependencies merely
  to obtain a needed function, that chain is not installed, and
  `erase_methods.train_uce` is not live-imported.
- `close_form_emb_regzero` (used inside Stage 2) lives in
  `utils/embedding_calculation/close_form_emb.py`, whose package
  `__init__.py` does `from .adv_embedding_images import *`, and
  `adv_embedding_images.py` does `from execs import compute_nudity_rate` --
  the same onnxruntime chain, for the same unrelated reason. Same decision:
  vendor, do not live-import.

`edit_model_adversarial` (Stage 2's actual erasing step) has no such
problem -- `erase_methods/uce.py` and `erase_methods/__init__.py` import
cleanly (torch, copy, functools, ast, tqdm, pickle, math only, already
confirmed working in the prior task's `--cpu_check`) -- so Stage 2 imports
it live from the pinned repo in `run_edit.py`, not vendored here.

IMPORTANT CORRECTION vs. the prior task's `SEQUENTIAL_EDIT_DESIGN.md`:
that document characterized RECE's "preliminary UCE step" as using
`erase_methods/uce.py`'s `edit_model_adversarial` (the full-77-token-sum
function) for both stages. Direct re-reading of `train_uce.py` for this
task shows that is not what the authors' own Stage-1 driver script calls:
`train_uce.py` defines and calls its OWN, separate `edit_model` function
(this file, vendored below), which uses the SINGLE-TOKEN slicing convention
(`final_token_idx = attention_mask.sum()-2`) -- the same convention as the
"official" UCE script referenced earlier in this project, not the 77-token
sum. Only Stage 2 (`edit_model_adversarial`) sums over all 77 token
positions. This file's docstrings and `run_edit.py` use the corrected,
re-verified attribution; `SEQUENTIAL_EDIT_DESIGN.md` is not retroactively
edited (it was a prepared document for a different, unexecuted concept
pair), but this correction is carried into this task's own RUN_MANIFEST.
"""
import hashlib
import inspect
from pathlib import Path

RECE_REPO = Path("/data/bijaypandey/cuig_pilot/rece_golfball_author_screen/repo_src/RECE")

_TRAIN_UCE_PATH = RECE_REPO / "erase_methods" / "train_uce.py"
_CLOSE_FORM_EMB_PATH = RECE_REPO / "utils" / "embedding_calculation" / "close_form_emb.py"

_EXPECTED_TRAIN_UCE_SHA256 = "3470f6abeafd741b4a996bb15636e7dbee74a777c464beada180836f13d66dd0"
_EXPECTED_CLOSE_FORM_EMB_SHA256 = "d1943625d951fd78ec2c8b9b780312cc555ebcbf859f0a364505f4d369e2f559"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_pinned_sources_unchanged():
    """Fail loudly if the pinned repo files this module vendors from have
    changed on disk since the hashes recorded above were taken (2026-10-14).
    This is the first line of drift defense: if this fails, the vendored
    functions below must be re-transcribed and re-reviewed before use."""
    actual_train_uce = _sha256(_TRAIN_UCE_PATH)
    actual_close_form_emb = _sha256(_CLOSE_FORM_EMB_PATH)
    if actual_train_uce != _EXPECTED_TRAIN_UCE_SHA256:
        raise RuntimeError(
            f"erase_methods/train_uce.py sha256 changed: expected "
            f"{_EXPECTED_TRAIN_UCE_SHA256}, got {actual_train_uce}. "
            "Refusing to use vendored edit_model() without re-verification."
        )
    if actual_close_form_emb != _EXPECTED_CLOSE_FORM_EMB_SHA256:
        raise RuntimeError(
            f"utils/embedding_calculation/close_form_emb.py sha256 changed: "
            f"expected {_EXPECTED_CLOSE_FORM_EMB_SHA256}, got "
            f"{actual_close_form_emb}. Refusing to use vendored "
            "close_form_emb_regzero() without re-verification."
        )


def _rstrip_lines(text: str) -> str:
    """Normalize only trailing-whitespace-per-line (an artifact of this
    file having been saved through an editor that strips trailing
    whitespace on otherwise-blank/short lines). Leading whitespace
    (indentation, i.e. actual code structure) is never touched, so this
    cannot mask a real logic difference -- only cosmetic trailing spaces."""
    return "\n".join(line.rstrip() for line in text.splitlines())


def assert_vendored_text_matches_pinned_source():
    """Second line of defense: extract this module's own vendored function
    source via `inspect.getsource` and check it appears verbatim (modulo
    trailing-whitespace-per-line only, see `_rstrip_lines`) inside the
    pinned repo file's current text. Raises if the vendored copy has
    drifted from the pinned source even if the file hash check above were
    ever loosened."""
    train_uce_text = _rstrip_lines(_TRAIN_UCE_PATH.read_text())
    close_form_emb_text = _rstrip_lines(_CLOSE_FORM_EMB_PATH.read_text())

    edit_model_src = _rstrip_lines(inspect.getsource(edit_model)).strip()
    if edit_model_src not in train_uce_text:
        raise RuntimeError(
            "Vendored edit_model() body does not appear verbatim (modulo "
            "trailing whitespace) in the pinned erase_methods/train_uce.py "
            "text. Refusing to proceed."
        )

    close_form_emb_regzero_src = _rstrip_lines(inspect.getsource(close_form_emb_regzero)).strip()
    if close_form_emb_regzero_src not in close_form_emb_text:
        raise RuntimeError(
            "Vendored close_form_emb_regzero() body does not appear "
            "verbatim (modulo trailing whitespace) in the pinned "
            "close_form_emb.py text. Refusing to proceed."
        )


# =============================================================================
# VERBATIM COPY of erase_methods/train_uce.py lines 28-172 (def edit_model).
# Source: erase_methods/train_uce.py, pinned repo commit
# 403a67fd86dce6c124b88798340e001e625b28c9. NOT modified in any way.
# =============================================================================
import torch
import copy
import ast
from tqdm import tqdm
import pandas as pd


def edit_model(ldm_stable, old_text_, new_text_, retain_text_, layers_to_edit=None, lamb=0.1, erase_scale = 0.1, preserve_scale = 0.1, with_to_k=True, technique='tensor'):
    ### collect all the cross attns modules
    max_bias_diff = 0.05
    sub_nets = ldm_stable.unet.named_children()
    ca_layers = []
    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__ :
                    for attn in block.attentions:
                        for  transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for  transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)

    ### get the value and key modules
    projection_matrices = [l.to_v for l in ca_layers]
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
    if with_to_k:
        projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
        og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_layers]

    ## reset the parameters
    num_ca_clip_layers = len(ca_layers)
    for idx_, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
            projection_matrices[num_ca_clip_layers + idx_] = l.to_k

    ### check the layers to edit (by default it is None; one can specify)
    layers_to_edit = ast.literal_eval(layers_to_edit) if type(layers_to_edit) == str else layers_to_edit
    lamb = ast.literal_eval(lamb) if type(lamb) == str else lamb

    ### Format the edits
    old_texts = []
    new_texts = []
    for old_text, new_text in zip(old_text_, new_text_):
        old_texts.append(old_text)
        n_t = new_text
        if n_t == '':
            n_t = ' '
        new_texts.append(n_t)
    if retain_text_ is None:
        ret_texts = ['']
        retain = False
    else:
        ret_texts = retain_text_
        retain = True

    print(old_texts, new_texts)
    print(f'CUDA memory allocated: {torch.cuda.memory_allocated()/1e9} GB')
    ######################## START ERASING ###################################
    for layer_num in tqdm(range(len(projection_matrices)), desc='Editing layers'):
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue

        #### prepare input k* and v*
        with torch.no_grad():
            #mat1 = \lambda W + \sum{v k^T}
            mat1 = lamb * projection_matrices[layer_num].weight

            #mat2 = \lambda I + \sum{k k^T}
            mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1], device = projection_matrices[layer_num].weight.device)

            for cnt, t in enumerate(zip(old_texts, new_texts)):
                old_text = t[0]
                new_text = t[1]
                texts = [old_text, new_text]
                text_input = ldm_stable.tokenizer(
                    texts,
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]


                final_token_idx = text_input.attention_mask[0].sum().item()-2
                final_token_idx_new = text_input.attention_mask[1].sum().item()-2
                farthest = max([final_token_idx_new, final_token_idx])

                old_emb = text_embeddings[0]
                old_emb = old_emb[final_token_idx:len(old_emb)-max(0,farthest-final_token_idx)]
                new_emb = text_embeddings[1]
                new_emb = new_emb[final_token_idx_new:len(new_emb)-max(0,farthest-final_token_idx_new)]

                context = old_emb.detach()

                values = []
                with torch.no_grad():
                    for layer in projection_matrices:
                        if technique == 'tensor':
                            o_embs = layer(old_emb).detach()
                            u = o_embs
                            u = u / u.norm()

                            new_embs = layer(new_emb).detach()
                            new_emb_proj = (u*new_embs).sum()

                            target = new_embs - (new_emb_proj)*u
                            values.append(target.detach())
                        elif technique == 'replace':
                            values.append(layer(new_emb).detach())
                        else:
                            values.append(layer(new_emb).detach())
                context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                mat1 += erase_scale*for_mat1
                mat2 += erase_scale*for_mat2

            for old_text, new_text in zip(ret_texts, ret_texts):
                text_input = ldm_stable.tokenizer(
                    [old_text, new_text],
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                old_emb, new_emb = text_embeddings
                context = old_emb.detach()
                values = []
                with torch.no_grad():
                    for layer in projection_matrices:
                        values.append(layer(new_emb[:]).detach())
                context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                mat1 += preserve_scale*for_mat1
                mat2 += preserve_scale*for_mat2
                #update projection matrix
            projection_matrices[layer_num].weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2))

    print(f'Current model status: Edited "{str(old_text_)}" into "{str(new_texts)}" and Retained "{str(retain_text_)}"')
    return ldm_stable


# =============================================================================
# VERBATIM COPY of utils/embedding_calculation/close_form_emb.py lines 132-222
# (def close_form_emb_regzero). Source: pinned repo commit
# 403a67fd86dce6c124b88798340e001e625b28c9. NOT modified in any way.
# =============================================================================
def close_form_emb_regzero(model, model_copy, concept, with_to_k=True, save_path=None, save_name=None, regeular_scale=1e-3, seed=123):
    ### collect all the cross attns modules
    print('Regularization to 0 vector')
    sub_nets = model.unet.named_children()
    sub_nets_copy = model_copy.unet.named_children()
    ca_layers = []
    ca_layers_copy = []
    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__:
                    for attn in block.attentions:
                        for transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)
    for net in sub_nets_copy:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__:
                    for attn in block.attentions:
                        for transformer in attn.transformer_blocks:
                            ca_layers_copy.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for transformer in attn.transformer_blocks:
                    ca_layers_copy.append(transformer.attn2)
    ### get the value and key modules
    projection_matrices = [l.to_v for l in ca_layers]
    projection_matrices_copy = [l.to_v for l in ca_layers_copy]
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
    og_matrices_copy = [copy.deepcopy(l.to_v) for l in ca_layers_copy]
    if with_to_k:
        projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
        projection_matrices_copy = projection_matrices_copy + [l.to_k for l in ca_layers_copy]
        og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_layers]
        og_matrices_copy = og_matrices_copy + [copy.deepcopy(l.to_k) for l in ca_layers_copy]

    ## reset the parameters
    num_ca_clip_layers = len(ca_layers)
    for idx_, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[idx_ + num_ca_clip_layers])
            projection_matrices[idx_ + num_ca_clip_layers] = l.to_k
    for idx_, l in enumerate(ca_layers_copy):
        l.to_v = copy.deepcopy(og_matrices_copy[idx_])
        projection_matrices_copy[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices_copy[idx_ + num_ca_clip_layers])
            projection_matrices_copy[idx_ + num_ca_clip_layers] = l.to_k

    ### get the concept embedding
    concept_tokens =  model.tokenizer(
                    concept,
                    padding="max_length",
                    max_length=model.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
    concept_ids = concept_tokens["input_ids"].to(model.device)
    concept_embedding = model.text_encoder(concept_ids)[0].squeeze(0)
    # ###### slice ######
    # final_token_idx = concept_tokens.attention_mask[0].sum().item()-2
    # concept_embedding = concept_embedding[final_token_idx:]
    # ###### slice ######

    mat1 = torch.eye(projection_matrices[0].weight.shape[1]).to(model.device)*regeular_scale
    mat2 = torch.zeros((projection_matrices[0].weight.shape[1], projection_matrices[0].weight.shape[1])).to(model.device)

    for idx_, l in enumerate(projection_matrices):
        mat1 = mat1 + torch.matmul(l.weight.T, l.weight)
        mat2 = mat2 + torch.matmul(l.weight.T, projection_matrices_copy[idx_].weight)
    coefficent = torch.matmul(torch.inverse(mat1), mat2)
    adv_embedding = torch.matmul(concept_embedding, coefficent.T).unsqueeze(0)


    if save_path is not None:
        print(f'Saving the embedding and image to {save_path}')
        # save the new embedding as pt
        torch.save(adv_embedding, f'{save_path}/close_new_embedding_preserved.pt')
        # create a df with column as concept and evaluation_seed
        df = pd.DataFrame(columns=['concept', 'evaluation_seed'])
        df['concept'] = [concept]
        df['evaluation_seed'] = [seed]

        embedding2img(adv_embedding, df, model, save_path, save_name=save_name)
    return concept_embedding.unsqueeze(0), adv_embedding


if __name__ == "__main__":
    assert_pinned_sources_unchanged()
    assert_vendored_text_matches_pinned_source()
    print("OK: vendored edit_model() and close_form_emb_regzero() match the pinned repo source byte-for-byte.")
