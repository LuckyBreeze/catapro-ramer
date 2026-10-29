"""Short CPU-only environment/tokenizer/LoRA compatibility probe; no model weights."""
import importlib
import json
import os
import sys
import traceback

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
result = dict(python=sys.version, executable=sys.executable, imports={}, loaded_large_weights=False,
              original_environment_modified=False, probe='CPU only, full GPU model not yet validated')
for name in ['torch', 'transformers', 'peft', 'accelerate', 'sentencepiece',
             'google.protobuf', 'numpy', 'h5py', 'Bio']:
    try:
        module = importlib.import_module(name)
        result['imports'][name] = dict(version=getattr(module, '__version__', 'unknown'))
    except Exception as exc:
        result['imports'][name] = dict(error=type(exc).__name__ + ': ' + str(exc))
try:
    import torch
    torch.set_num_threads(1)
    from transformers import T5Tokenizer, T5EncoderModel, T5Config
    from peft import LoraConfig, get_peft_model, TaskType
    tokenizer = T5Tokenizer.from_pretrained('/share/home/u17015/data/RAMER/model/ProtT5',
                                            legacy=False, local_files_only=True)
    seq = 'ACDEFGHIKLMNPQRSTVWYX'
    encoded = tokenizer(' '.join(seq), add_special_tokens=False)
    assert len(encoded['input_ids']) == len(seq) and encoded['attention_mask'] == [1] * len(seq)
    result['tokenizer'] = dict(aa_count=len(seq), token_count=len(encoded['input_ids']),
                                ids=encoded['input_ids'], passed=True)
    toy = T5EncoderModel(T5Config(d_model=16, d_kv=8, d_ff=32, num_layers=1,
                                 num_heads=2, vocab_size=128))
    toy = get_peft_model(toy, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=8,
                         lora_alpha=32, lora_dropout=0.1, target_modules=['q','k','v','o','wi','wo']))
    toy.eval()
    with torch.no_grad():
        output = toy(input_ids=torch.tensor([[2,3,4]]),
                     attention_mask=torch.ones(1,3,dtype=torch.long)).last_hidden_state
    assert list(output.shape) == [1,3,16] and bool(torch.isfinite(output).all())
    result['tiny_cpu_lora'] = dict(shape=list(output.shape), passed=True,
                                  lora_keys=[k for k in toy.state_dict() if 'lora_' in k])
    result['status'] = 'cpu_interface_passed_gpu_pending'
    result['h5py_note'] = 'Numpy token-cache writer does not import h5py; official encoder modules do not import h5py'
except Exception:
    result['status'] = 'failed'
    result['error'] = traceback.format_exc()
print(json.dumps(result))
if result['status'] == 'failed':
    sys.exit(1)
