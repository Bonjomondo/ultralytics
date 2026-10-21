"""Experiment adapter for a SHA-pinned source fork; never claims wheel identity.

Reuses detrscope's native hook collector and all model-independent algorithms.
The wheel adapter's source/RECORD gate remains unchanged.
"""
import inspect
import json
from pathlib import Path
import subprocess

from detrscope.adapters.base import ModelAdapter, UnsupportedCapability
from detrscope.adapters.ultralytics_rtdetr import UltralyticsRTDETRAdapter, _verify_native_forwards
from detrscope.core.io import sha256_file


class PinnedR18NAAdapter(UltralyticsRTDETRAdapter):
    def __init__(self, model, *, repo, pin):
        ModelAdapter.__init__(self, model)
        repo = Path(repo).resolve()
        pin = json.loads(Path(pin).read_text())
        if pin['schema'] != 'r18-na.source-pin.v1':
            raise UnsupportedCapability('Unknown source pin schema')
        subprocess.run(['git', '-C', str(repo), 'merge-base', '--is-ancestor', pin['commit'], 'HEAD'], check=True)
        actual = {p.relative_to(repo).as_posix(): sha256_file(p) for p in sorted((repo/'ultralytics').rglob('*.py'))}
        if actual != pin['source_sha256']:
            raise UnsupportedCapability('Ultralytics source tree differs from the frozen experiment pin')
        from ultralytics.nn.modules.head import RTDETRDecoder
        heads = [m for m in model.modules() if type(m) is RTDETRDecoder]
        if len(heads) != 1:
            raise UnsupportedCapability('Expected exactly one native RTDETRDecoder')
        self.head = head = heads[0]
        for cls in (type(head), type(head.decoder), type(head.decoder.layers[0].cross_attn)):
            source = Path(inspect.getsourcefile(cls)).resolve()
            if not source.is_relative_to(repo/'ultralytics'):
                raise UnsupportedCapability(f'Wrong source origin: {source}')
        _verify_native_forwards(head)
        source = inspect.getsource(type(head.decoder).forward)
        for token in ('refer_bbox = refer_bbox.sigmoid()', 'i == self.eval_idx', 'refined_bbox', 'score_head[i]'):
            if token not in source:
                raise UnsupportedCapability(f'Incompatible decoder: {token}')
        selection = inspect.getsource(type(head)._get_decoder_input)
        if 'torch.topk' not in selection or 'enc_outputs_scores.max(-1).values' not in selection:
            raise UnsupportedCapability('Incompatible encoder query selection')
        if '_debug_recorder' in source or '_debug_recorder' in inspect.getsource(type(head).forward):
            raise UnsupportedCapability('SO-DETR recorder instrumentation requires its dedicated adapter')
        if (head.nc, head.num_queries, head.decoder.num_layers, head.decoder.eval_idx) != (10, 300, 3, 2):
            raise UnsupportedCapability('This experiment pin expects 10 classes, 300 queries, 3 executed decoder layers')
        self.upstream = {'installation': 'explicit_sha256_pinned_source_fork', 'source_commit': pin['commit'],
                         'source_sha256': actual, 'wheel_record_verified': False}
        self._postprocessed_output = 'self.postprocess(' in inspect.getsource(type(head).forward)
        self._observing = False
        self.class_names = [str(model.names[i]) for i in range(head.nc)]
        self.category_ids = list(range(1, head.nc+1))
        if len(set(self.class_names)) != head.nc:
            raise UnsupportedCapability('Class names must be unique')

    def describe(self):
        result = super().describe()
        result['adapter'] = 'r18_na_experiment_pinned_source'
        result['adapter_scope'] = 'Explicit experiment plugin; not the unmodified upstream wheel adapter'
        return result
