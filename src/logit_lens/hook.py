from torch import nn
from torch.utils.hooks import RemovableHandle

from logit_lens.models import HIDDEN_STATE_EXTRACTORS


class PostHook:
    hook: RemovableHandle
    result = None

    def __init__(self, module: nn.Module):
        self.hook = module.register_forward_hook(self.hook_fn, with_kwargs=True)
        self.post_process_fn = lambda x: x

    def hook_fn(self, module, args, kwargs, output) -> None:
        self.result = self.post_process_fn(output) if self.post_process_fn else output

    def remove(self):
        self.hook.remove()


class PostLayerHookForHiddenStates(PostHook):
    def __init__(self, module: nn.Module):
        super().__init__(module)
        self.post_process_fn = HIDDEN_STATE_EXTRACTORS.get(
            module.__class__.__name__, lambda x: x
        )
