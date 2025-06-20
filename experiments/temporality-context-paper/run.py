import os
import pandas as pd
import argparse
from lm_tools.core import LanguageModel, LogProbability
import torch
import time

from tqdm import tqdm
import signal
import sys
from datasets import load_dataset

def sigterm_handler(_signo, _stack_frame):
    save_progress()
    sys.exit(42)

signal.signal(signal.SIGTERM, sigterm_handler)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", 
    type=str,
    required=True, 
    help="The huggingface language model name used for inference"
)
parser.add_argument(
    "--instruct",
    action="store_true",
    help="The language model is instruct-tuned"
)
parser.add_argument(
    "--test",
    action="store_true",
    help="Run only 100 inferences for testing purposes."
)
parser.add_argument(
    "--batch_size",
    type=int,
    help="Inference batch size",
    default=32
)
parser.add_argument(
    "--experiment",
    type=str,
    help="What experiment to perform",
    default="classic",
    choices=['classic', 'explain_granularity_generalization', 'explain_date',
            #  'fewshot', 'reason_explain', 'fewshot_reason_explain'
            ]
)

parser.add_argument(
    "--torch_dtype",
    type=str,
    help="Precision of the model",
    default="bfloat16",
    choices=['bfloat16', "float32"]
)

parser.add_argument(
    "--timestress_path",
    type=str,
    help="Path to the TimeStress dataset file. If not provided, the TimeStress dataset on Huggingface is used.",
    default=None
)

parser.add_argument(
    "--output_folder",
    type=str,
    help="Folder where to save results. Defaults to current folder.",
    default='./'
)

def load_from_huggingface() -> pd.DataFrame:
    from wikidata_tools.core import Entity, Relation, TimedTriple, Date, Interval
    df = load_dataset('Orange/TimeStress')["train"].to_pandas()
    df['Subject'] = df[['SubjectID', 'SubjectLabel']].apply(lambda x: Entity(x['SubjectID'], x['SubjectLabel']), axis=1)
    df['Relation'] = df[['RelationID', 'RelationLabel']].apply(lambda x: Relation(x['RelationID'], x['RelationLabel']), axis=1)
    df['Object'] = df[['ObjectID', 'ObjectLabel']].apply(lambda x: Entity(x['ObjectID'], x['ObjectLabel']), axis=1)
    df['StartDate'] = df['StartDate'].apply(Date.from_string)
    df['EndDate'] = df['EndDate'].apply(Date.from_string)
    df['Fact'] = df[['Subject', 'Relation', 'Object', 'StartDate', 'EndDate']].apply(lambda x : TimedTriple(x['Subject'], x['Relation'], x['Object'], 
                                                                                                            Interval(x['StartDate'], x['EndDate'])), axis=1)
    df.drop(columns=[
        'Subject', 'SubjectID', 'SubjectLabel',
        'Relation', 'RelationID', 'RelationLabel',
        'Object', 'ObjectID', 'ObjectLabel',
        'StartDate', 'EndDate'
    ], inplace=True)
    return df
    



def treat_special_cases(model_name : str) -> str:
    if model_name.startswith('apple/OpenELM'):
        return "NousResearch/Llama-2-7b-hf"
    return model_name

def edge_cases(tokenizer):
    if tokenizer.name_or_path == "NousResearch/Llama-2-7b-hf":
        tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    # elif tokenizer.name_or_path.startswith('meta-llama/Llama-3.1'):
    #     tokenizer.chat_template = tokenizer.chat_template.lstrip("{{- bos_token }}")

def get_queries_instruct():
    return get_preprompt() + df['Statement']

def get_queries_pretrained():
    return get_preprompt() + df['Statement']

def get_preprompt():
#     REASON_PROMPT = """Here is an example of a reasoning chain to answer temporal questions such as "In February 2012, what was the position of Barack Obama?":
# 1. Retrieve the history of the position held by Obama over time and the associated periods when they were valid
# 2. Iterate over them and output the position whose validity period includes the date "February 2012"
# Note that if Obama was the president of the US in February 2012, then necessarily he was the president on every day of this month (February 1, 2012, February 23, 2012, etc.)
# """
#     FEWSHOT_PROMPT = """Answer the question at the end like the following examples:
# In December 2010, who was the head of state of Algeria? Abdelaziz Bouteflika
# On September 30, 2008, which team did Cristiano Ronaldo play for? Manchester United F.C.
# """
    
    EXPLAIN_GRANULARITY_GENERALIZATION_PROMPT = """A date is a specific point in time. If a fact is valid for a specific year, it holds true for all dates within that year. If a fact is valid for a specific month of a specific year, it holds true for all dates within that month. Answer the following question.
"""
    EXPLAIN_DATE_PROMPT = """A date is a specific point in time, expressed through a year, a month, and a day. A year is divided into months, and a month is divided into days. Answer the following question.
"""
    if args.experiment == 'classic':
        return ""
    # if args.experiment == 'fewshot':
    #     return FEWSHOT_PROMPT
    # if args.experiment == 'reason_explain':
    #     return REASON_PROMPT
    # if args.experiment == 'fewshot_reason_explain':
    #     return REASON_PROMPT + FEWSHOT_PROMPT
    if args.experiment == 'explain_date':
        return EXPLAIN_DATE_PROMPT
    if args.experiment == 'explain_granularity_generalization':
        return EXPLAIN_GRANULARITY_GENERALIZATION_PROMPT
    

def save_progress():
    if len(inferred_batches) == 0:
        print('Warning: no progress made')
        inferred = pd.DataFrame(columns=df.columns)
    else:
        inferred = pd.concat(inferred_batches)
    
    os.makedirs(output_folder, exist_ok=True)
    if progress is not None:
        print('Concat previous progress with new results')
        progress_and_inferred = pd.concat([progress, inferred])
    else:
        print('Concat previous progress with new results (progress is empty)')
        progress_and_inferred = inferred
    print('Saving file : %s' % (filepath + "_"))
    progress_and_inferred.to_pickle(filepath + "_")
    print('file saved')
    if len(inferred) == len(df):
        # Inference finished
        print('Whole job finished.')
        os.rename(filepath + "_", filepath)
    
args = parser.parse_args()

output_folder = args.output_path


end_time = time.time() + float('inf')
print('Job ends at timestamp %s' % end_time)

# path = "./"
filepath = os.path.join(output_folder,'model=%s__experiment=%s.pkl' % (args.model.replace('/', '_'), args.experiment))

if os.path.exists(filepath):
    print('Computation already done.')
    exit(0)

if os.path.exists(filepath + "_"):
    progress = pd.read_pickle(filepath + "_")
    start = len(progress)
else:
    progress = None
    start = 0

if args.timestress_path is not None:
    print('Loading TimeStress Locally from %s' % args.timestress_path)
    df : pd.DataFrame = pd.read_pickle(args.timestress_path)
else:
    print('Loading TimeStress from Huggingface')
    df = load_from_huggingface()
# df : pd.DataFrame = pd.read_pickle(f'data/facts.pkl')
if args.test:
    df = df.iloc[:100]

df = df.iloc[start:]

lm_dict = {
            "pretrained_model_name_or_path": args.model,
            "device_map": "auto",
            "torch_dtype": torch.bfloat16 if args.torch_dtype == 'bfloat16' else torch.float32,
            "trust_remote_code": True,
        }
tok_dict = {"pretrained_model_name_or_path": treat_special_cases(args.model), "trust_remote_code": True}

lm = LanguageModel.from_pretrained(lm_dict, tok_dict)
edge_cases(lm.hf_tokenizer)
cred = LogProbability()

print('Infering Normal questions + Instructions questions...')
def f(x : dict):
    return lm.hf_tokenizer.apply_chat_template(
        conversation=[  
            dict(role='user', content=x['Statement']),
            dict(role='assistant', content=x['Fact'].object.label)
        ],
        tokenize=False,
        continue_final_message=True,
    )
inferred_batches = []
with torch.no_grad():
    queries = get_queries_instruct()
    queries.name = 'Statement'
    data = pd.concat([queries, df['Fact']], axis=1).apply(f, axis=1).tolist() if args.instruct else None
    queries = get_queries_pretrained().tolist()
    compute_on = df['Fact'].apply(lambda x : x.object.label).tolist()
    for i in tqdm(range(0, len(df), args.batch_size)):
        if time.time() > end_time:
            save_progress()
            exit(42)
        inferred_batch = df.iloc[i:i+args.batch_size].copy()
        co = compute_on[i:i+args.batch_size]
        if args.instruct:
            inp = data[i:i+args.batch_size]
            inferred_batch["CondLogProbInstruct"] = lm.credibility_text(inp, cred, co, ignore_case=True).float().cpu().numpy()
        else:
            inferred_batch["CondLogProbInstruct"] = float('nan')
        inp = queries[i:i+args.batch_size]
        inferred_batch['CondLogProb'] = lm.credibility_text(inp, cred, co, ignore_case=True).float().cpu().numpy()
        inferred_batches.append(inferred_batch)

save_progress()
