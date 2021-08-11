from util import load_models_from_artifact
import wandb
from pathlib import Path
import torch
import models
import fed

api = wandb.Api()

def model_artifact_from_id(run_id, allow_global=True, project="maschm/master-fed"):
    if not "/" in run_id:
        run_id = f"{project}/{run_id}"
    run = api.run(run_id)

    run_art, is_global = None, False
    for art in run.logged_artifacts():
        if art.name.startswith("final"):
            is_global = False
            run_art = art
            break
        elif allow_global and art.name.startswith("global"):
            is_global = True
            run_art = art
            break
    return run_art, run, is_global

def load_model_from_artifact(art, run=None, is_global=None, project="maschm/master-fed"):
    if isinstance(art, str):
        if not "/" in art:
            art = f"{project}/{art}"
        art = api.artifact(art)
    if run is None:
        run = art.logged_by()
    if is_global is None:
        is_global = art.name.startswith("global")
    art_path = Path(art.download())
    model_path = next((c for c in art_path.glob("*pth") if not "optim" in str(c)))

    variant = getattr(models, run.config['model_variant'])
    mapping = run.config['global_model_mapping'] if is_global \
        else run.config['model_mapping'][0]
    # mapping = int(mapping)
    projection = run.config['projection_head']

    model = variant(*variant.hyper[mapping], projection=projection)
    errors = model.load_state_dict(
        torch.load(model_path, map_location=torch.device('cpu')),
        strict=False)
    assert errors.missing_keys == []
    return model

def load_model_from_id(run_id, allow_global=True):
    art, run, is_global = model_artifact_from_id(run_id, allow_global)
    return load_model_from_artifact(art, run, is_global)

def create_embeding_image_for_run(run_id, data, train_data, allow_global=True):
    art, run, is_global = model_artifact_from_id(run_id, allow_global)
    model = load_model_from_artifact(art, run, is_global)
    cfg = run.config
    cfg['rank'] = "-1" if is_global else "0"
    worker = fed.FedWorker(cfg, model)
