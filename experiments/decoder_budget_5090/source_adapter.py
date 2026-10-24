"""Explicit source-fork adapter for decoder-budget research.

The caller creates a new pin after reviewing an expected Git ancestor. This is
an experiment adapter, never evidence of the upstream wheel's RECORD identity.
Only native eager/eval RT-DETR with encoder-feature query initialization is
supported. Query count may change; learned query embeddings are out of scope.
"""
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import subprocess

import detrscope
from detrscope.adapters.base import ModelAdapter, UnsupportedCapability
from detrscope.adapters.ultralytics_rtdetr import UltralyticsRTDETRAdapter, _verify_native_forwards
from detrscope.core.io import sha256_file


PIN_SCHEMA = "decoder-budget-5090.source-pin.v1"


def _git(repo, *args):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if result.returncode:
        raise UnsupportedCapability(f"Source Git identity check failed: {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


def _source_hashes(root):
    root = Path(root).resolve()
    paths = sorted(root.rglob("*.py"))
    if not paths or not (root / "__init__.py").is_file():
        raise UnsupportedCapability(f"Expected a Python package source tree: {root}")
    if any(not path.resolve().is_relative_to(root) for path in paths):
        raise UnsupportedCapability(f"Python source escapes its pinned root: {root}")
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}


def create_source_pin(repo, output, *, expected_commit):
    """Freeze explicit local source evidence, refusing to overwrite an old pin.

    ``expected_commit`` is mandatory: it is the caller-reviewed ancestor, not an
    automatically chosen HEAD. Creating a pin authorizes these source bytes for
    this experiment; it does not certify compatibility (the adapter checks it).
    Both the fork and the actually imported SDK are pinned, including added and
    removed Python files. Relocating either tree requires an explicit new pin.
    """
    repo, output = Path(repo).resolve(), Path(output)
    if not isinstance(expected_commit, str) or not expected_commit.strip():
        raise ValueError("An explicit expected_commit is required")
    if Path(_git(repo, "rev-parse", "--show-toplevel")).resolve() != repo:
        raise UnsupportedCapability("repo must be the source Git repository root")
    ancestor = _git(repo, "rev-parse", "--verify", "--end-of-options", expected_commit + "^{commit}")
    _git(repo, "merge-base", "--is-ancestor", ancestor, "HEAD")
    sdk = Path(detrscope.__file__).resolve().parent
    pin = {"schema": PIN_SCHEMA, "repo_root": str(repo),
           "expected_ancestor_commit": ancestor, "commit": _git(repo, "rev-parse", "HEAD"),
           "branch_at_creation": _git(repo, "branch", "--show-current"),
           "source_sha256": _source_hashes(repo / "ultralytics"),
           "source_worktree_status": _git(repo, "status", "--porcelain", "--untracked-files=all", "--", "ultralytics"),
           "sdk_root": str(sdk), "sdk_version": str(detrscope.__version__),
           "sdk_source_sha256": _source_hashes(sdk),
           "trust": "caller_created_explicit_local_source_pin; not upstream wheel verification"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(pin, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return pin


def _verify_pin(repo, pin):
    if pin.get("schema") != PIN_SCHEMA:
        raise UnsupportedCapability("Unknown decoder-budget source pin schema")
    if pin.get("repo_root") != str(repo):
        raise UnsupportedCapability("Source location differs from the explicit pin")
    if Path(_git(repo, "rev-parse", "--show-toplevel")).resolve() != repo:
        raise UnsupportedCapability("repo must be the source Git repository root")
    for field in ("expected_ancestor_commit", "commit"):
        commit = pin.get(field)
        if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise UnsupportedCapability(f"Pin requires a complete Git commit SHA: {field}")
        _git(repo, "merge-base", "--is-ancestor", commit, "HEAD")
    _git(repo, "merge-base", "--is-ancestor", pin["expected_ancestor_commit"], pin["commit"])
    if _source_hashes(repo / "ultralytics") != pin.get("source_sha256"):
        raise UnsupportedCapability("Ultralytics source tree differs from the frozen experiment pin")
    sdk = Path(detrscope.__file__).resolve().parent
    if str(sdk) != pin.get("sdk_root") or _source_hashes(sdk) != pin.get("sdk_source_sha256"):
        raise UnsupportedCapability("Imported detrscope SDK differs from the frozen source pin")


class PinnedRTDETRAdapter(UltralyticsRTDETRAdapter):
    """Reuse the SDK collector with a separate, explicit source-fork gate."""

    def __init__(self, model, *, repo, pin, class_names, category_ids):
        ModelAdapter.__init__(self, model)
        self.repo = Path(repo).resolve()
        self.pin_path = Path(pin).resolve()
        self.pin = json.loads(self.pin_path.read_text(encoding="utf-8"))
        _verify_pin(self.repo, self.pin)
        from ultralytics.nn.modules.head import RTDETRDecoder
        from ultralytics.nn.tasks import RTDETRDetectionModel
        if type(model) is not RTDETRDetectionModel:
            raise UnsupportedCapability("Expected the native source-fork RTDETRDetectionModel")
        heads = [module for module in model.modules() if type(module) is RTDETRDecoder]
        if len(heads) != 1:
            raise UnsupportedCapability("Expected exactly one native RTDETRDecoder")
        self.head = heads[0]
        self._validate_model()
        self.class_names = list(class_names)
        self.category_ids = list(category_ids)
        if (len(self.class_names) != self.head.nc or
                any(not isinstance(name, str) or not name.strip() for name in self.class_names) or
                len(set(self.class_names)) != self.head.nc):
            raise ValueError("class_names must contain one unique nonempty semantic name per output class")
        if (len(self.category_ids) != self.head.nc or
                any(not isinstance(value, int) or isinstance(value, bool) for value in self.category_ids) or
                len(set(self.category_ids)) != self.head.nc):
            raise ValueError("category_ids must contain one unique integer ID per output class")
        self.upstream = {"installation": "explicit_sha256_pinned_source_fork",
                         "source_commit": self.pin["commit"], "source_current_commit": _git(self.repo, "rev-parse", "HEAD"),
                         "expected_ancestor_commit": self.pin["expected_ancestor_commit"],
                         "source_root": str(self.repo), "source_sha256": self.pin["source_sha256"],
                         "sdk_root": self.pin["sdk_root"], "sdk_source_sha256": self.pin["sdk_source_sha256"],
                         "source_pin_sha256": sha256_file(self.pin_path), "wheel_record_verified": False}
        self._postprocessed_output = False
        self._observing = False

    def _validate_model(self):
        head = self.head
        _verify_native_forwards(self.model)
        package = self.repo / "ultralytics"
        for module in self.model.modules():
            cls = type(module)
            if cls.__module__.startswith("ultralytics."):
                source = inspect.getsourcefile(cls)
                if source is None or not Path(source).resolve().is_relative_to(package):
                    raise UnsupportedCapability(f"Wrong source origin for {cls.__qualname__}: {source}")
                function_source = inspect.getsourcefile(cls.forward)
                if function_source is None or not Path(function_source).resolve().is_relative_to(package):
                    raise UnsupportedCapability(f"Non-native class forward: {cls.__qualname__}")
        for method in (type(head)._get_decoder_input, type(head).forward, type(head.decoder).forward):
            if not Path(inspect.getsourcefile(method) or "").resolve().is_relative_to(package):
                raise UnsupportedCapability("Non-native decoder/query-selection implementation")
        if any(module.training for module in self.model.modules()) or head.export:
            raise UnsupportedCapability("Only already-eval, eager, non-export inference is supported")
        if head.learnt_init_query:
            raise UnsupportedCapability("Query-budget research requires learnt_init_query=False; learned embeddings are unsupported")
        if not isinstance(head.num_queries, int) or isinstance(head.num_queries, bool) or head.num_queries < 1:
            raise UnsupportedCapability("num_queries must be a positive integer")
        if not 0 <= head.decoder.eval_idx < head.decoder.num_layers:
            raise UnsupportedCapability("Invalid native decoder eval_idx")
        source = inspect.getsource(type(head.decoder).forward)
        for token in ("refer_bbox = refer_bbox.sigmoid()", "i == self.eval_idx", "refined_bbox", "score_head[i]"):
            if token not in source:
                raise UnsupportedCapability(f"Incompatible decoder semantics: {token}")
        selection = inspect.getsource(type(head)._get_decoder_input)
        if "torch.topk" not in selection or "enc_outputs_scores.max(-1).values" not in selection:
            raise UnsupportedCapability("Incompatible native encoder query-selection semantics")
        forward = inspect.getsource(type(head).forward)
        if "_debug_recorder" in source + forward or "self.postprocess(" in forward:
            raise UnsupportedCapability("Only the pinned native query-output path is supported")
        for layer in head.decoder.layers:
            attention = layer.cross_attn
            if not isinstance(attention.n_points, int) or attention.n_points < 1:
                raise UnsupportedCapability("Only uniform integer sampling-point counts are supported in this phase")

    def describe(self):
        result = super().describe()
        result.update(adapter="decoder_budget_5090_pinned_source",
                      adapter_scope="Explicit experiment source plugin; wheel RECORD gate remains unchanged",
                      query_initialization="selected_encoder_features; learnt_init_query=False",
                      query_count_semantics="exact native Top-K num_queries; requires enough encoder tokens")
        return result

    @contextmanager
    def observe(self, recorder):
        # Detect edits, changed imports, runtime mode/config changes and forward
        # overrides even when they occur after construction of this adapter.
        if sha256_file(self.pin_path) != self.upstream["source_pin_sha256"]:
            raise UnsupportedCapability("Source pin changed after adapter construction")
        _verify_pin(self.repo, self.pin)
        self._validate_model()
        with super().observe(recorder):
            yield self
