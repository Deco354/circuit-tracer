from typing import Callable
from circuit_tracer.transcoder import load_transcoder_set
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder, TranscoderSettings
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn.modules.module import _global_module_registration_hooks
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder
from circuit_tracer.replacement_model import ReplacementMLP
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

# Load the model
device = "cuda" if torch.cuda.is_available() else "mps"
model = HookedTransformer.from_pretrained("google/gemma-2-2b", fold_ln=False, center_writing_weights=False, center_unembed=False, device=device)
feature_output_hook = "mlp.hook_out"
feature_input_hook = "mlp.hook_in"

# Load the transcoders from hugging face
transcoder_settings = load_transcoder_set("gemma")
transcoders: OrderedDict[int, SingleLayerTranscoder] = transcoder_settings.transcoders
feature_input_hook: str = transcoder_settings.feature_input_hook
feature_output_hook: str = transcoder_settings.feature_output_hook
scan: str | list[str] = transcoder_settings.scan

print(transcoders.keys())
type(transcoders[0])
for transcoder in transcoders.values():
    transcoder.to(device)
    
transcoders_module = nn.ModuleList([transcoders[i] for i in range(model.cfg.n_layers)])
print(transcoders_module)
model.add_module("transcoders", transcoders_module)
# model.d_transcoder = transcoders[-1].d_transcoder # This is part of the subclass!
# model.original_feature_output_hook = 
# model.feature_input_hook = feature_input_hook
# model.feature_output_hook = feature_output_hook
# model.skip_transcoder = transcoders[-1].W_skip is not None
# model.scan = scan

# Add skip connections
for layer, transcoder in enumerate(transcoders.values()):
    transformer_block = model.blocks[layer]
    cache = {}
    mlp_block = getattr(transformer_block, "mlp")
    output_hookpoint: HookPoint = getattr(mlp_block, "hook_out")
    # hook_string.add_hook(cache_activations, is_permanent=True)
    # print(hook_string)
    # hook_out_grad = getattr(mlp_block, f"{hook_string}_grad")
    # print(hook_out_grad.name)
    # mlp_block.add_hook(hook_out_grad, is_permanent=True)
    
    
    

for transformer_block in model.blocks:
    transformer_block.mlp = ReplacementMLP(transformer_block.mlp)
    print(transformer_block.mlp)
    
# model.unembed = ReplacementUnembed(model.unembed)
# model._configure_gradient_flow()
# model._deduplicate_attention_buffers()
# model.setup()

# # 2. Replace MLPs with ReplacementMLP-like structure
# class SimpleReplacementMLP(nn.Module):
#     """Simplified version of ReplacementMLP that just adds hook points"""
    
#     def __init__(self, old_mlp):
#         super().__init__()
#         self.old_mlp = old_mlp
#         # Add hook points like ReplacementMLP does
#         self.hook_in = nn.Identity()  # Simplified - in real ReplacementMLP this would be a HookPoint
#         self.hook_out = nn.Identity()
    
#     def forward(self, x):
#         x = self.hook_in(x)
#         mlp_out = self.old_mlp(x)
#         return self.hook_out(mlp_out)

# # Replace each MLP in the model
# for i, block in enumerate(model.blocks):
#     block.mlp = SimpleReplacementMLP(block.mlp)

# # 3. Set up the model attributes that ReplacementModel would have
# setattr(model, 'd_transcoder', transcoders[0].d_transcoder)  # Assuming all transcoders have same d_transcoder
# setattr(model, 'feature_input_hook', feature_input_hook)
# setattr(model, 'feature_output_hook', feature_output_hook)
# setattr(model, 'skip_transcoder', transcoders[0].W_skip is not None)

# print("Model configured with transcoders!")
# print(f"d_transcoder: {model.d_transcoder}")
# print(f"feature_input_hook: {model.feature_input_hook}")
# print(f"feature_output_hook: {model.feature_output_hook}")
# print(f"skip_transcoder: {model.skip_transcoder}")
    







