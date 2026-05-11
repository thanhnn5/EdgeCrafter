"""
Thin wandb adapter exposing the same surface (add_scalar / add_text / close)
as torch.utils.tensorboard.SummaryWriter so existing solver call-sites do
not need to change.

Selection rules (see BaseConfig.writer):
- Default backend is wandb if importable and not explicitly disabled.
- WANDB_DISABLED=true (or WANDB_MODE=disabled) skips wandb and uses TB.
- If wandb import or init fails for any reason, TB is used.
"""
import os
from pathlib import Path
from typing import Optional


def _wandb_disabled() -> bool:
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        return True
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        return True
    return False


def wandb_available() -> bool:
    if _wandb_disabled():
        return False
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False
    return True


class WandbWriter:
    """Drop-in replacement for SummaryWriter that logs to Weights & Biases.

    Only the subset of the SummaryWriter API actually used by this codebase
    is implemented: add_scalar, add_text, close. Add new methods here as
    new call-sites appear.
    """

    def __init__(
        self,
        log_dir: str,
        *,
        project: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        import wandb

        self._wandb = wandb
        self.log_dir = str(log_dir)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        project = project or os.environ.get("WANDB_PROJECT", "edgecrafter")
        run_name = run_name or os.environ.get("WANDB_NAME") or Path(self.log_dir).parent.name

        # If a run is already active in this process (e.g. user pre-initialized
        # via env), reuse it. Otherwise start one.
        if wandb.run is not None:
            self._run = wandb.run
        else:
            self._run = wandb.init(
                project=project,
                name=run_name,
                dir=self.log_dir,
                config=config or {},
                reinit=False,
            )

    # ---- SummaryWriter surface ---------------------------------------------

    def add_scalar(self, tag: str, scalar_value, global_step=None, walltime=None):
        # wandb.log uses dict semantics; step is a top-level kwarg.
        payload = {tag: scalar_value}
        if global_step is not None:
            self._wandb.log(payload, step=int(global_step))
        else:
            self._wandb.log(payload)

    def add_text(self, tag: str, text_string: str, global_step=None, walltime=None):
        # Use wandb.config for static config; for arbitrary tagged text use Html.
        if tag == "config":
            try:
                self._run.config.update({"raw_config": text_string}, allow_val_change=True)
            except Exception:
                # Non-fatal: config dump is informational.
                self._wandb.log({tag: self._wandb.Html(f"<pre>{text_string}</pre>")})
            return
        payload = {tag: self._wandb.Html(f"<pre>{text_string}</pre>")}
        if global_step is not None:
            self._wandb.log(payload, step=int(global_step))
        else:
            self._wandb.log(payload)

    def close(self):
        try:
            self._wandb.finish()
        except Exception:
            pass
