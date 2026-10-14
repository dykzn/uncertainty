"""Prepare reviewed, meaning-preserving SPUQ instructions for four LIBERO suites."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
from runtime_config import configure_runtime
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

# Locations, object identities, conjunctions and action order are retained.
GOAL_REWRITES = {
    "open the middle drawer of the cabinet": [
        "pull open the middle drawer of the cabinet",
        "open the cabinet's middle drawer",
        "slide the middle drawer of the cabinet open",
        "please open the middle drawer of the cabinet"],
    "put the bowl on the stove": [
        "place the bowl on the stove", "set the bowl on the stove",
        "move the bowl onto the stove", "please put the bowl on the stove"],
    "put the wine bottle on top of the cabinet": [
        "place the wine bottle on top of the cabinet", "set the wine bottle on top of the cabinet",
        "move the wine bottle onto the top of the cabinet", "please put the wine bottle on top of the cabinet"],
    "open the top drawer and put the bowl inside": [
        "open the top drawer and place the bowl inside it",
        "pull open the top drawer and put the bowl inside",
        "open the top drawer, then put the bowl inside it",
        "please open the top drawer and put the bowl inside"],
    "put the bowl on top of the cabinet": [
        "place the bowl on top of the cabinet", "set the bowl on top of the cabinet",
        "move the bowl onto the top of the cabinet", "please put the bowl on top of the cabinet"],
    "push the plate to the front of the stove": [
        "push the plate toward the front of the stove", "slide the plate to the front of the stove by pushing it",
        "move the plate to the front of the stove with a push", "please push the plate to the front of the stove"],
    "put the cream cheese in the bowl": [
        "place the cream cheese in the bowl", "put the cream cheese into the bowl",
        "move the cream cheese into the bowl", "please put the cream cheese in the bowl"],
    "turn on the stove": [
        "switch on the stove", "turn the stove on", "switch the stove on", "please turn on the stove"],
    "put the bowl on the plate": [
        "place the bowl on the plate", "set the bowl on the plate",
        "move the bowl onto the plate", "please put the bowl on the plate"],
    "put the wine bottle on the rack": [
        "place the wine bottle on the rack", "set the wine bottle on the rack",
        "move the wine bottle onto the rack", "please put the wine bottle on the rack"],
}

LONG_REWRITES = {
    "put both the alphabet soup and the tomato sauce in the basket": [
        "place both the alphabet soup and the tomato sauce in the basket",
        "put the alphabet soup and the tomato sauce into the basket",
        "move both the alphabet soup and the tomato sauce into the basket",
        "please put both the alphabet soup and the tomato sauce in the basket"],
    "put both the cream cheese box and the butter in the basket": [
        "place both the cream cheese box and the butter in the basket",
        "put the cream cheese box and the butter into the basket",
        "move both the cream cheese box and the butter into the basket",
        "please put both the cream cheese box and the butter in the basket"],
    "turn on the stove and put the moka pot on it": [
        "switch on the stove and place the moka pot on it",
        "turn the stove on and put the moka pot on the stove",
        "turn on the stove, then place the moka pot on it",
        "please turn on the stove and put the moka pot on it"],
    "put the black bowl in the bottom drawer of the cabinet and close it": [
        "place the black bowl in the cabinet's bottom drawer and close the drawer",
        "put the black bowl into the bottom drawer of the cabinet and close that drawer",
        "put the black bowl in the bottom drawer of the cabinet, then close the drawer",
        "please put the black bowl in the bottom drawer of the cabinet and close it"],
    "put the white mug on the left plate and put the yellow and white mug on the right plate": [
        "place the white mug on the left plate and place the yellow and white mug on the right plate",
        "set the white mug on the left plate and set the yellow and white mug on the right plate",
        "put the white mug on the left plate, then put the yellow and white mug on the right plate",
        "please put the white mug on the left plate and put the yellow and white mug on the right plate"],
    "pick up the book and place it in the back compartment of the caddy": [
        "lift the book and put it in the back compartment of the caddy",
        "take the book and place it in the caddy's back compartment",
        "grasp the book, then place it in the back compartment of the caddy",
        "please pick up the book and place it in the back compartment of the caddy"],
    "put the white mug on the plate and put the chocolate pudding to the right of the plate": [
        "place the white mug on the plate and place the chocolate pudding to the right of the plate",
        "set the white mug on the plate and put the chocolate pudding to the right of the plate",
        "put the white mug on the plate, then put the chocolate pudding to the right of the plate",
        "please put the white mug on the plate and put the chocolate pudding to the right of the plate"],
    "put both the alphabet soup and the cream cheese box in the basket": [
        "place both the alphabet soup and the cream cheese box in the basket",
        "put the alphabet soup and the cream cheese box into the basket",
        "move both the alphabet soup and the cream cheese box into the basket",
        "please put both the alphabet soup and the cream cheese box in the basket"],
    "put both moka pots on the stove": [
        "place both moka pots on the stove", "set both moka pots on the stove",
        "move both moka pots onto the stove", "please put both moka pots on the stove"],
    "put the yellow and white mug in the microwave and close it": [
        "place the yellow and white mug in the microwave and close the microwave",
        "put the yellow and white mug into the microwave and close it",
        "put the yellow and white mug in the microwave, then close the microwave",
        "please put the yellow and white mug in the microwave and close it"],
}


def main():
    configure_runtime()
    import torch
    import transformers.utils
    import transformers.utils.import_utils
    transformers.utils.is_flash_attn_2_available = lambda: False
    transformers.utils.import_utils.is_flash_attn_2_available = lambda: False
    from transformers import AutoModel, AutoTokenizer
    from libero.libero import benchmark
    from reflect_benchmark import DEFAULT_ROOT, make_suite, safe_initial_states, file_sha256
    import yaml

    torch.set_num_threads(4)
    destination = ROOT / "assets/all-MiniLM-L6-v2"
    manifest = json.loads((destination / "download_manifest.json").read_text())
    assert manifest["revision"] == REVISION
    for entry in manifest["files"]:
        assert file_sha256(destination / entry["path"]) == entry["sha256"]
    tokenizer = AutoTokenizer.from_pretrained(destination, local_files_only=True)
    model = AutoModel.from_pretrained(destination, local_files_only=True).eval()
    paths = yaml.safe_load((Path(__import__("os").environ["LIBERO_CONFIG_PATH"]) / "config.yaml").read_text())
    inventory = {}
    spatial = json.loads((ROOT / "assets/spatial_instructions.json").read_text())
    for name in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        suite = make_suite(benchmark, name, "libero")
        reflect = make_suite(benchmark, name, "libero_reflect", DEFAULT_ROOT)
        prepared, assets = [], []
        for index in range(10):
            task = suite.get_task(index)
            original = task.language
            if name == "libero_spatial":
                row = spatial["tasks"][index].copy()
                assert row["original"] == original
            else:
                if name == "libero_object":
                    prefix, suffix = "pick up ", " and place it in the basket"
                    assert original.startswith(prefix) and original.endswith(suffix)
                    obj = original[len(prefix):-len(suffix)]
                    rewrites = [f"lift {obj} and put it in the basket",
                                f"take {obj} and set it in the basket",
                                f"grasp {obj}, then place it in the basket",
                                f"pick up {obj}, then put it in the basket"]
                else:
                    rewrites = (GOAL_REWRITES if name == "libero_goal" else LONG_REWRITES)[original]
                texts = [original] + rewrites
                assert len(set(texts)) == 5
                inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
                with torch.inference_mode():
                    hidden = model(**inputs).last_hidden_state
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    vectors = torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1).clamp(min=1), p=2, dim=1)
                    weights = (vectors @ vectors[0]).cpu().numpy()
                assert weights.min() > 0 and abs(float(weights[0]) - 1) < 1e-5
                row = {"task_id": index + 1, "original": original, "perturbations": texts,
                       "instruction_weights": weights.tolist(), "instruction_embeddings": vectors.tolist()}
            row["task_name"] = task.name
            prepared.append(row)
            initial = Path(paths["init_states"]) / task.problem_folder / task.init_states_file
            states = safe_initial_states(initial)
            assert states.shape[0] == 50 and reflect.reflect_initial_states[index].shape[0] == 50
            bddl = Path(paths["bddl_files"]) / task.problem_folder / task.bddl_file
            assets.append({"task_id": index + 1, "task_name": task.name,
                           "standard_bddl_sha256": file_sha256(bddl), "standard_initial_states_sha256": file_sha256(initial),
                           "reflect_bddl_sha256": file_sha256(reflect.reflect_asset_paths[index]["bddl"]),
                           "reflect_initial_states_sha256": file_sha256(reflect.reflect_asset_paths[index]["initial_states"]),
                           "initial_states_each_split": 50})
        if name != "libero_spatial":
            result = {"suite": name, "embedding_model": MODEL, "embedding_revision": REVISION,
                      "paraphrase_source": "reviewed templates preserving objects, locations, action conjunctions and order",
                      "pooling": spatial["pooling"], "tasks": prepared,
                      "preparation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            (ROOT / "assets" / f"{name.removeprefix('libero_')}_instructions.json").write_text(json.dumps(result, indent=2) + "\n")
        inventory[name] = {"tasks": assets, "episodes_per_split": 500,
                           "reflect_variant": reflect.reflect_variant,
                           "instruction_file_sha256": file_sha256(ROOT / "assets" / f"{name.removeprefix('libero_')}_instructions.json")}
        print("SUITE_ASSETS_READY", name, "1000 episodes", flush=True)
    (ROOT / "assets/formal_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")


if __name__ == "__main__":
    main()
