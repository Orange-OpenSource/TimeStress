import math
import re


models_constraints = [
    ('meta-llama/Llama-3.1-70B', False, ['queue:name=gpu_mem_80', 'gpu:2']),
    ('meta-llama/Llama-3.1-70B-Instruct', True, ['queue:name=gpu_mem_80', 'gpu:2']),
    # ('mistralai/Mixtral-8x7B-v0.1', False, ['queue:name=gpu_mem_80', 'gpu:2']),
    # ('mistralai/Mixtral-8x7B-Instruct-v0.1', True, ['queue:name=gpu_mem_80', 'gpu:2']),
    ('mistralai/Mistral-Nemo-Base-2407', False, ['queue:name=gpu_mem_40', 'gpu:1']),
    ('mistralai/Mistral-Nemo-Instruct-2407', True, ['queue:name=gpu_mem_40', 'gpu:1']),
    ('google/gemma-2-9b-it', True, ['queue:name=gpu_mem_40', 'gpu:1']),
    ('google/gemma-2-9b', False, ['queue:name=gpu_mem_40', 'gpu:1']),
    ('google/gemma-2-27b-it', True, ['queue:name=gpu_mem_80', 'gpu:1']),
    ('google/gemma-2-27b', False, ['queue:name=gpu_mem_80', 'gpu:1']),
]
additional_models_constraints = [
    ('meta-llama/Llama-3.1-8B', False),
    ('meta-llama/Llama-3.1-8B-Instruct', True),
    ('mistralai/Mistral-7B-v0.3', False),
    ('mistralai/Mistral-7B-Instruct-v0.3', True),

    # ("apple/OpenELM-270M", False),
    ("apple/OpenELM-450M", False),
    # ("apple/OpenELM-1_1B", False),
    ("apple/OpenELM-3B", False),
    # ("apple/OpenELM-270M-Instruct", True),
    ("apple/OpenELM-450M-Instruct", True),
    # ("apple/OpenELM-1_1B-Instruct", True),
    ("apple/OpenELM-3B-Instruct", True),
    ("google/gemma-2-2b-it", True),
    ("google/gemma-2-2b", False),
]

_all_models = [(x,y) for x,y,_ in models_constraints] + additional_models_constraints

def is_instruct(model : str) -> bool:
    for model_, instruct in _all_models:
        if model_.replace('/', '_') == model.replace('/', '_'):
            return instruct
        
classic2instruct = {
    "mistralai_Mistral-Nemo-Base-2407" : "mistralai_Mistral-Nemo-Instruct-2407",
    "mistralai_Mixtral-8x7B-v0.1" : "mistralai_Mixtral-8x7B-Instruct-v0.1",
    "mistralai_Mistral-7B-v0.3" : "mistralai_Mistral-7B-Instruct-v0.3"
}
def get_instruct_version_of_model(model : str) -> str:
    model = model.replace('/', '_')
    if model.startswith('meta-llama_') or model.startswith('apple_'):
        return model + "-Instruct"
    if model.startswith('google_gemma'):
        return model + '-it'
    return classic2instruct.get(model)

def get_classical_version_of_model(model : str) -> str:
    model = model.replace('/', '_')
    if model.startswith('meta-llama_') or model.startswith('apple_'):
        return model[:-len("-Instruct")]
    if model.startswith('google_gemma'):
        return model[:-len('-it')]
    inst2instruct = {v:k for k,v in classic2instruct.items()}
    return inst2instruct.get(model)

def get_model_family(model : str):
    model = model.replace('/', '_')
    if model.startswith('apple_'):
        return 'OpenELM'
    if model.startswith('meta-llama_'):
        return 'LLaMa-3.1'
    if model.startswith('google_gemma'):
        return "Gemma-2"
    if model.startswith('mistralai'):
        return "Mistral"
    return model.split('_', 1)[-1]

def extract_metadata_from_filename(filename : str) -> str:
    m = re.findall(r"(.+?)=(.+?)(\_\_|.pkl)", filename)
    metadata = {x:y for x,y,_ in m}
    return metadata

def lm2req(model : str, precision=16) -> int:
    """Model name to memory requirements (in GB)
    """
    requirement = get_model_num_params(model) * precision / 8 / 10**9
    requirement += math.ceil(requirement*0.1)
    return int(requirement)

def get_model_num_params(model : str) -> int:
    model = model.replace('/', '_')
    if model in ('mistralai_Mistral-Nemo-Instruct-2407', 'mistralai_Mistral-Nemo-Base-2407'):
        return 12.2 * 10**9
    if model.startswith('apple_OpenELM-1_1B'):
        return 1.1 * 10 ** 9
    if '8x7B' in model:
        return 8*7*10**9
    char2size = {
        'm' : 10**6,
        'b' : 10**9
    }
    l = re.findall(r'([0-9]+)(M|B|m|b)', model)
    assert len(l) == 1, "Many model sizes found : %s" % model
    n, g = l[0]
    n = int(n)
    g = char2size[g.lower()]
    n_params = n*g
    return n_params
