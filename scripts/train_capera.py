"""Project-side training/eval entry for xmodaler on CapERA.

Replicates third_party/xmodaler/train_net.py, plus project registrations:
- CapERADataset (xmodaler.datasets.videos.capera)
- SelectorTransformerEncoderDecoder + SELECTOR config schema (models.selector_enc_dec)
- CapEraTrainer: empty-caption guard, expected-selection-ratio budget regularizer,
  selection statistics logging, selection trace dump.
Keeps upstream xmodaler files 100% unmodified.
"""
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XMODALER_DIR = os.path.join(PROJECT_ROOT, "third_party", "xmodaler")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, XMODALER_DIR)

# register CapERADataset into xmodaler.datasets.DATASETS_REGISTRY
import xmodaler.datasets.videos.capera  # noqa: F401

# register SelectorTransformerEncoderDecoder + SELECTOR config schema +
# frame selectors (must happen before get_cfg/merge_from_file)
import models.selector_enc_dec  # noqa: F401

# Environment-level patch: disable SPICE in pycocoevalcap.
# pycocoevalcap's COCOEvalCap constructs Spice() unconditionally, which downloads
# stanford-corenlp-3.6.0 (384 MB, very slow from this host) and needs a JVM at
# scoring time. We only report BLEU-4 / METEOR / ROUGE-L / CIDEr, so replace the
# Spice scorer with a no-op before any COCOEvalCap is instantiated.
# (Records in docs/THIRD_PARTY_PATCHES.md)
import pycocoevalcap.eval as _coco_eval


class _DummySpice:
    def __init__(self):
        pass

    def method(self):
        return "SPICE"

    def compute_score(self, *args, **kwargs):
        return 0.0, []


# evaluate() builds its scorer list locally and references the module-level
# `Spice` name; replacing that name with a no-op is sufficient (SPICE will show
# up in the output dict as 0.0 and can be ignored).
_coco_eval.Spice = _DummySpice

import torch
import tqdm
import xmodaler.utils.comm as comm
from xmodaler.config import get_cfg
from xmodaler.config import kfg
from xmodaler.engine import (DefaultTrainer, build_engine, default_argument_parser,
                             default_setup, launch)
from xmodaler.engine.build import ENGINE_REGISTRY
from xmodaler.modeling import add_config
from xmodaler.utils.events import get_event_storage

from models.selectors.base_selector import (SEL_INDICES, SEL_NUM_CANDIDATES,
                                            SEL_NUM_SELECTED, SEL_PROBS, SEL_STATS)

_ID_MAP = None


def _video_name(int_id):
    """int id -> original CapERA video_id string (lazy-loaded map)."""
    global _ID_MAP
    if _ID_MAP is None:
        with open(os.path.join(PROJECT_ROOT, "data", "CapERA", "video_id_map.json")) as f:
            _ID_MAP = json.load(f)["ids"]
    return _ID_MAP.get(str(int_id), {}).get("video_id", str(int_id))


@ENGINE_REGISTRY.register()
class CapEraTrainer(DefaultTrainer):
    """DefaultTrainer + CapERA-specific additions.

    - empty generated captions are replaced with a minimal fallback so that
      pycocoevalcap's len(hypo)==1 assertion holds in early epochs;
    - a differentiable expected-selection-ratio budget regularizer steers the
      selector away from the trivial all-pick solution (PickNet-style v1;
      NOT the original RL reward);
    - per-step selection statistics go to the event storage;
    - eval dumps a selection trace JSON per epoch.
    """

    def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        try:
            data = next(self._train_data_loader_iter)
        except StopIteration:
            if comm.get_world_size() > 1:
                self.train_data_loader.sampler.set_epoch(self.iter // self.iters_per_epoch)
            self._train_data_loader_iter = iter(self.train_data_loader)
            data = next(self._train_data_loader_iter)
        data_time = time.perf_counter() - start

        data = comm.unwrap_model(self.model).preprocess_batch(data)
        data[kfg.SS_PROB] = self.ss_prob
        outputs_dict = self.model(data)

        losses_dict = {}
        for loss in self.losses:
            loss_dict = loss(outputs_dict)
            losses_dict.update(loss_dict)
        losses = [losses_dict[k] for k in losses_dict if 'acc' not in k]

        # ---- project: expected-selection-ratio budget regularizer ----
        sel = getattr(self.cfg, "SELECTOR", None)
        budget_loss = None
        if sel is not None and sel.LAMBDA_BUDGET > 0 and SEL_PROBS in data:
            probs = data[SEL_PROBS]                                  # (B, T)
            num_cand = data[SEL_NUM_CANDIDATES].float().clamp(min=1)  # (B,)
            ratio = probs.sum(1) / num_cand                          # expected ratio
            budget_loss = sel.LAMBDA_BUDGET * ((ratio - sel.TARGET_RATIO) ** 2).mean()
            losses.append(budget_loss)

        losses = sum(losses)

        self.optimizer.zero_grad()
        losses.backward()

        self._write_metrics(losses_dict, data_time)

        # ---- project: selection stats + budget into event storage ----
        storage = get_event_storage()
        if budget_loss is not None:
            storage.put_scalar("select/budget_loss", float(budget_loss.detach().item()))
        if SEL_STATS in data:
            for k, v in data[SEL_STATS].items():
                storage.put_scalar(f"select/{k}", float(v))

        self.optimizer.step()
        if self.ema is not None:
            self.ema.update(self.model)

    @classmethod
    def test(cls, cfg, model, test_data_loader, evaluator, epoch):
        model.eval()
        results = []
        traces = []
        with torch.no_grad():
            for data in tqdm.tqdm(test_data_loader):
                data = comm.unwrap_model(model).preprocess_batch(data)
                ids = data[kfg.IDS]

                if cfg.INFERENCE.GENERATION_MODE == True:
                    res = model(data, use_beam_search=True, output_sents=True)
                else:
                    res = model(data)

                outputs = res[kfg.OUTPUT]
                for id, output in zip(ids, outputs):
                    if not isinstance(output, str) or output.strip() == "":
                        output = "."
                    results.append({cfg.INFERENCE.ID_KEY: int(id), cfg.INFERENCE.VALUE: output})

                # ---- project: selection trace ----
                if SEL_INDICES in data:
                    probs = data[SEL_PROBS].cpu().tolist()
                    for i, (id, idxs) in enumerate(zip(ids, data[SEL_INDICES])):
                        traces.append({
                            "video_id": _video_name(int(id)),
                            "int_id": int(id),
                            "candidate_indices": list(range(int(data[SEL_NUM_CANDIDATES][i].item()))),
                            "selected_indices": list(idxs),
                            "selection_probs": [round(p, 4) for p in probs[i]],
                            "num_selected": len(idxs),
                        })

        if evaluator is not None:
            eval_res = evaluator.eval(results, epoch)
        else:
            eval_res = ''

        # ---- project: dump selection trace ----
        if traces and cfg.OUTPUT_DIR:
            res_dir = os.path.join(cfg.OUTPUT_DIR, "results")
            os.makedirs(res_dir, exist_ok=True)
            out = os.path.join(res_dir, f"{epoch}_selection_trace.json")
            ns = [t["num_selected"] for t in traces]
            summary = {
                "n_videos": len(ns),
                "avg_selected": round(float(np_mean(ns)), 3),
                "min_selected": int(min(ns)),
                "max_selected": int(max(ns)),
                "median_selected": float(np_median(ns)),
                "selection_ratio": round(float(np_mean(ns)) / 10.0, 3),
            }
            with open(out, "w") as f:
                json.dump({"summary": summary, "traces": traces}, f, indent=1)
            print(f"[selection trace] {out}")
            print(f"[selection summary] {summary}")

        model.train()
        return eval_res


def np_mean(xs):
    import numpy as np
    return np.mean(xs)


def np_median(xs):
    import numpy as np
    return np.median(xs)


def setup(args):
    cfg = get_cfg()
    tmp_cfg = cfg.load_from_file_tmp(args.config_file)
    add_config(cfg, tmp_cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    trainer = build_engine(cfg)
    trainer.resume_or_load(resume=args.resume)

    if args.eval_only:
        if trainer.val_data_loader is not None:
            res = trainer.test(trainer.cfg, trainer.model, trainer.val_data_loader,
                               trainer.val_evaluator, epoch=-1)
            if comm.is_main_process():
                print(res)
        if trainer.test_data_loader is not None:
            res = trainer.test(trainer.cfg, trainer.model, trainer.test_data_loader,
                               trainer.test_evaluator, epoch=-1)
            if comm.is_main_process():
                print(res)
        return

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
