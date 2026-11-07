"""Stride-aware anchors and semantic initialization mapping for three P2 ablations."""
import sys
from copy import deepcopy

from common import BACKEND, CONFIGS
sys.path.insert(0, str(BACKEND))

import torch
from ultralytics.nn.modules.head import RTDETRDecoder as AuthorDecoder
from ultralytics.nn import tasks


class StrideAwareDecoder(AuthorDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.nl in (3, 4)
        self.feature_strides = [8, 16, 32] if self.nl == 3 else [4, 8, 16, 32]
        self.capture_query_stats = False
        self.last_query_counts = []
        self.last_query_levels = []

    def _generate_anchors(self, shapes, grid_size=.05, dtype=torch.float32, device='cpu', eps=1e-2):
        # Preserve the author's P3/P4/P5 priors when inserting stride-4 P2 first.
        return super()._generate_anchors(shapes, grid_size * self.feature_strides[0] / 8,
                                         dtype, device, eps)

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        result = super()._get_decoder_input(feats, shapes, dn_embed, dn_bbox)
        if self.capture_query_stats:
            with torch.no_grad():
                _, valid = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
                scores = self.enc_score_head(self.enc_output(valid * feats.detach())).max(-1).values
                selected = scores.topk(self.num_queries, dim=1).indices
                boundaries = torch.tensor([h*w for h,w in shapes], device=feats.device).cumsum(0)
                levels = torch.bucketize(selected, boundaries, right=True)
                self.last_query_levels = levels.cpu().tolist()
                self.last_query_counts = torch.bincount(levels.flatten(), minlength=self.nl).cpu().tolist()
        return result


# The author's parser resolves this symbol from its own module namespace.
tasks.RTDETRDecoder = StrideAwareDecoder


def state_sha(state):
    import hashlib
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode() + str(value.dtype).encode() + str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def paired(arm, seed):
    with torch.random.fork_rng(devices=[0] if torch.cuda.is_available() else []):
        torch.manual_seed(seed)
        reference = tasks.RTDETRDetectionModel(str(CONFIGS['b0']), nc=10, verbose=False)
        if arm == 'b0':
            model = deepcopy(reference)
        else:
            torch.manual_seed(seed + 1000)
            model = tasks.RTDETRDetectionModel(str(CONFIGS[arm]), nc=10, verbose=False)
        ref, current = reference.state_dict(), model.state_dict()
        if arm == 'full':
            layer_map = {i:i for i in range(20)}
            layer_map.update({28:20, 30:22, 31:23, 33:25, 34:26})
        elif arm == 'c2':
            # P3 fusion inputs changed meaning, so do not copy B0's layer17 or layer19.
            layer_map = {i:i for i in [*range(17), *range(20, 27)]}
        else:
            layer_map = {i:i for i in range(27)}
        copied, extended = [], []
        for target, value in current.items():
            fields = target.split('.')
            layer = int(fields[1])
            if layer not in layer_map:
                continue
            source_fields = fields.copy()
            source_fields[1] = str(layer_map[layer])
            if arm == 'full' and layer == 34 and fields[2] == 'input_proj':
                level = int(fields[3])
                if level == 0:
                    continue
                source_fields[3] = str(level - 1)
            source = '.'.join(source_fields)
            if source not in ref:
                continue
            original = ref[source]
            if value.shape == original.shape:
                current[target] = original.clone()
                copied.append((target, source))
            elif arm == 'full' and layer == 34 and any(
                    part in target for part in ['sampling_offsets.', 'attention_weights.']):
                # Match existing scale rows by semantic level, including attention heads/points.
                tail = tuple(value.shape[1:])
                points = 8 if 'sampling_offsets.' in target else 4
                mapped = value.clone().reshape(8, 4, points, *tail)
                mapped[:, 1:] = original.reshape(8, 3, points, *tail)
                current[target] = mapped.reshape_as(value)
                extended.append((target, source))
        model.load_state_dict(current, strict=True)
        assert all(torch.equal(model.state_dict()[t], ref[s]) for t,s in copied)
        backbone = {k:v for k,v in model.state_dict().items() if int(k.split('.')[1]) < 8}
        model.initialization_receipt = {
            'arm':arm, 'seed':seed, 'backbone_sha256':state_sha(backbone),
            'copied_pairs':copied, 'extended_scale_pairs':extended,
            'new_or_changed_keys':sorted(set(current)-{t for t,s in copied}-{t for t,s in extended}),
            'no_trained_checkpoint':True,
            'limit':'Architectures differ; only semantically compatible tensors are paired; outputs need not match.'}
    return model
