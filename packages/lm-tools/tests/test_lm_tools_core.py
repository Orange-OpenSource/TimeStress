# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

from typing import Iterable
from lm_tools.utils import _find_connected_intervals
import numpy as np
import pytest
import torch

from ke_utils.glob_core import Precision
from lm_tools.core import (
    LanguageModel,
    LogProbability,
    NegPerplexity,
    PromptedLanguageModel,
    RandomLM,
    _decode_and_add_prompt,
)
from transformers import GPT2LMHeadModel, GPT2Config, PreTrainedTokenizerFast

from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import AutoTokenizer, AutoModelForCausalLM


class TestPerp(NegPerplexity):
    __test__ = False

    def __init__(self, set_zero_compute=True) -> None:
        super().__init__()
        self.n = 0
        self.set_zero_compute = set_zero_compute

    def compute(
        self,
        logits: Iterable[torch.Tensor] | torch.Tensor,
        input_ids: Iterable[torch.Tensor] | torch.Tensor,
        attention_mask: Iterable[torch.Tensor] | torch.Tensor = None,
    ) -> torch.Tensor:
        if self.set_zero_compute:
            self.n = 0
        return super().compute(logits, input_ids, attention_mask)

    def _compute_tensor(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        self.n += 1
        return super()._compute_tensor(logits, input_ids, attention_mask)

    
def build_dummy_tokenizer(vocab_size : int):    
    tokenizer = Tokenizer(BPE())
    tokenizer.add_special_tokens(["t%s " % i for i in range(1,vocab_size)])
    awesome_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    awesome_tokenizer.add_special_tokens({"pad_token" : "t0 "})
    return awesome_tokenizer


def test_NegPerplexity():
    x = torch.zeros((10, 20, 100))
    ids = torch.randint(0, 100, (10, 20))
    f = NegPerplexity()
    perp = f._compute_tensor(logits=x, input_ids=ids)
    assert torch.isclose(perp, torch.full_like(perp, -100)).all()

    x = torch.tensor([0, 0, 1, 1]).repeat(4).view(4, 4).float().unsqueeze(0)
    ids = torch.arange(4).unsqueeze(0)
    v1 = np.log(1 / (np.exp(1) * 2 + 2))
    v2 = np.log(np.exp(1) / (np.exp(1) * 2 + 2))
    perp = f._compute_tensor(logits=x, input_ids=ids)
    true = -torch.tensor([np.exp(-(v1 + v2 + v2) / 3)]).float()
    assert torch.isclose(perp, true).all()

    attention_mask = torch.tensor([1, 1, 1, 0]).unsqueeze(0)
    true = -torch.tensor([np.exp(-(v1 + v2) / 2)]).float()
    perp = f._compute_tensor(logits=x, input_ids=ids, attention_mask=attention_mask)
    assert torch.isclose(perp, true).all()


def test_logprob():
    x = torch.zeros((10, 20, 100))
    ids = torch.randint(0, 100, (10, 20))
    f = LogProbability()
    prob = f._compute_tensor(logits=x, input_ids=ids)
    assert torch.isclose(prob, torch.full_like(prob, -19 * np.log(100))).all()

    x = torch.tensor([0, 0, 1, 1]).repeat(4).view(4, 4).float().unsqueeze(0)
    ids = torch.arange(4).unsqueeze(0)
    v1 = np.log(1 / (np.exp(1) * 2 + 2))
    v2 = np.log(np.exp(1) / (np.exp(1) * 2 + 2))
    prob = f._compute_tensor(logits=x, input_ids=ids)
    true = torch.tensor([v1 + v2 + v2]).float()
    assert torch.isclose(prob, true).all()

    attention_mask = torch.tensor([1, 1, 1, 0]).unsqueeze(0)
    true = torch.tensor([v1 + v2]).float()
    prob = f._compute_tensor(logits=x, input_ids=ids, attention_mask=attention_mask)
    assert torch.isclose(prob, true).all()


def test_cred_func_core():
    f = TestPerp()
    N, S, V = 200, 20, 100
    x = torch.zeros((N, S, V))
    ids = torch.randint(0, V, (N, S))

    def input_ids():
        for i in range(0, N, 10):
            yield ids[i : i + 10]

    def logits():
        for i in range(0, N, 10):
            yield x[i : i + 10]

    perp = f.compute(logits(), input_ids())
    assert perp.shape == (N,)
    assert f.n == 20

    perp = f.compute(x, ids)
    assert perp.shape == (N,)
    assert f.n == 1

@pytest.fixture
def lm_dummytok():
    # Define a small GPT-2 configuration
    config = GPT2Config(
        vocab_size=100,  # Use the default GPT-2 tokenizer vocab size
        n_positions=64,  # Maximum sequence length
        n_ctx=64,  # Context window size
        n_embd=8,  # Size of the embeddings
        n_layer=1,  # Number of layers
        n_head=2,  # Number of attention heads
    )

    # Instantiate a model with the custom configuration
    model = GPT2LMHeadModel(config)
    model.eval()
    tokenizer = build_dummy_tokenizer(100)
    return LanguageModel(model, tokenizer, precision=Precision.FLOAT32)


def create_lm_realtok():
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # Define a small GPT-2 configuration
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,  # Use the default GPT-2 tokenizer vocab size
        n_positions=64,  # Maximum sequence length
        n_ctx=64,  # Context window size
        n_embd=8,  # Size of the embeddings
        n_layer=1,  # Number of layers
        n_head=2,  # Number of attention heads
    )

    # Instantiate a model with the custom configuration
    model = GPT2LMHeadModel(config)
    model.eval()

    return LanguageModel(model, tokenizer, precision=Precision.FLOAT32)

@pytest.fixture
def lm_realtok():
    return create_lm_realtok()


def test_lm(lm_dummytok: LanguageModel):
    cred = TestPerp(set_zero_compute=False)
    N, S, V = 200, 20, 100
    ids = torch.randint(0, V, (N, S))
    perp = lm_dummytok.credibility(ids, cred)
    assert perp.shape == (N,)
    assert cred.n == 1

    cred.n = 0
    perp = lm_dummytok.credibility(ids, cred, batch_size=10)
    assert perp.shape == (N,)
    assert cred.n == 20

    out = lm_dummytok.batch_forward_text(["t1 t2 t3 ", "t3 t4 t6 t8 t10 ", "t2 t5 t90 "])
    assert out.logits.shape == (3, 5, V)

    out = lm_dummytok.credibility_text(
        ["t1 t2 t3 ", "t3 t4 t6 t8 t10 ", "t2 t5 t90 "], cred, batch_size=1
    )
    assert out.shape == (3,)


def test_rand_lm():
    lm = RandomLM(build_dummy_tokenizer(100), exact=True)
    cred = NegPerplexity()
    perp = lm.credibility_text(["t1 t2 t3 ", "t3 t4 t6 t8 t10 ", "t2 t5 t90 "], cred)
    assert torch.isclose(perp, -torch.full_like(perp, 100)).all()

    cred = LogProbability()
    prob = lm.credibility_text(["t1 t2 t3 ", "t3 t4 t6 t8 t10 ", "t2 t5 t90 "], cred)
    assert torch.isclose(prob, torch.tensor([-2, -4, -2]) * np.log(100)).all()

    prob = lm.credibility_text(
        ["t1 t2 t3 ", "t3 t4 t6 t8 t10 ", "t2 t5 t90 ", "t1 t23 t12 t23 t86 " ],
        cred,
        ["t1 t2 ", "t6 t8 ", "t2 t5 t90 ", "t23 t12 t23 t86 "],
    )
    assert torch.isclose(prob, -torch.tensor([1, 2, 2, 4]) * np.log(100)).all()


tok_kwargs = {"pretrained_model_name_or_path": "gpt2"}
lm_rand = RandomLM.from_pretrained(tok_kwargs)

@pytest.mark.parametrize('lm', [lm_rand, create_lm_realtok()])
@pytest.mark.parametrize('retokenize', [True, False])
@pytest.mark.parametrize('truncate_prompt', [True, False])
def test_prompted_lm(lm, retokenize, truncate_prompt):
    if truncate_prompt and retokenize:
        pytest.skip("Not supported")
    prompt = "Joe Biden is the president of USA. "
    prompted_lm = PromptedLanguageModel(lm, prompt, retokenize=retokenize, truncate_prompt=truncate_prompt)
    sent = ["Hello my name is", "Hi", "Hihihihiii Sco", "Matrix : Revolution"]
    l = prompted_lm.tokenize(sent)

    l2 = _decode_and_add_prompt(
        **l, tokenizer=prompted_lm.hf_tokenizer, prompt=prompted_lm.prompt
    )
    assert [x == prompt + s for x, s in zip(l2, sent)]

    output = prompted_lm.forward(**l)
    if truncate_prompt:
        assert output.logits.shape[1] == l["input_ids"].shape[1]
    else:
        assert output.logits.shape[1] > l["input_ids"].shape[1]


@pytest.mark.parametrize("credibility_func", ['tokens', 'text'])
def test_credibility_text(lm_realtok : LanguageModel, credibility_func: str):
    texts = ["In 2020, D’Amour", " Celâl Şengör", " 香港女版孟慶樹"]

    # Read this carefully to unbderstand how to use this test
    # pos2logprobs = [
    #     { # texts[0]
    #         (from this text position, to this text position) : compute conditional logprobs of text[pos:] given text[pos:] using logits[:-1] strating from this logit index,
    #     },
    #     { # texts[1]
    #         ...
    #     },
    #     ...
    # ]

    pos2logprobs = [
        {
            (2,6) : 0,
            (7,7) : 1,
            (8,9) : 2,
            (10,10) : 3,
            (11,12) : 5,
            (13,15) : 6
        },
        {
            (4,4) : 0,
            (5,5) : 1,
            (6,7) : 2,
            (8,10) : 4,
            (11,12) : 5,
        },
        {
            (2,2) : 2,
            (3,3) : 5,
            (4,4) : 6,
            (5,5) : 7,
            (6,7) : 9
        }
    ]

    cred = LogProbability()

    for text_idx, text, pos2lp in zip(range(len(texts)), texts, pos2logprobs):
        input_ids = lm_realtok.hf_tokenizer(text, return_tensors='pt').input_ids[0]
        with torch.no_grad():
            logprobs = lm_realtok.hf_model(input_ids.unsqueeze(0)).logits[0, :-1].log_softmax(-1)
        sumlp = lambda x : torch.tensor([logprobs[i, input_ids[i+1]] for i in range(x, logprobs.shape[0])]).sum()
        pos2lp = {k:sumlp(v) for k,v in pos2lp.items()}

        for (i,j), true_cred in pos2lp.items():
            for pos in range(i,j+1):
                with torch.no_grad():
                    if credibility_func == 'text':
                        pred_cred = lm_realtok.credibility_text([text], cred, compute_on=[text[pos:]])[0]
                    elif credibility_func == 'tokens':
                        input_ids = lm_realtok.hf_tokenizer([text], return_tensors='pt').input_ids
                        compute_on = torch.zeros_like(input_ids, dtype=torch.bool)
                        start_idx = pos2logprobs[text_idx][(i,j)]
                        compute_on[:, start_idx+1:] = True
                        pred_cred = lm_realtok.credibility(input_ids, cred, compute_on=compute_on)[0]
                assert torch.isclose(pred_cred, true_cred)


def test_find_connected_intervals():
    spans = [(0, 3), (3, 6), (6, 10), (10, 15), (15, 18), (18, 29), (29, 32), (32, 40), (40, 41), (41, 43), (42, 43), (42, 43), (43, 44), (43, 44), (43, 44), (44, 45), (45, 46), (46, 47), (46, 47), (47, 48), (47, 48), (47, 49), (48, 49), (48, 49)]
    exp_conns = [(0, 3), (3, 6), (6, 10), (10, 15), (15, 18), (18, 29), (29, 32), (32, 40), (40, 41), (41, 43), (43,44), (44,45), (45, 46), (46, 47), (47,49)]
    exp_counts = [1,1,1,1,1,1,1,1,1,3,3,1,1,2,5]

    conns, counts = _find_connected_intervals(spans)
    assert conns == exp_conns and counts == exp_counts

    conns, counts = _find_connected_intervals([(0,0)] + spans)
    assert conns == [(0,0)] + exp_conns and counts == [1] + exp_counts

    conns, counts = _find_connected_intervals([(0,0)] + spans + [(50,60)])
    assert conns == [(0,0)] + exp_conns + [(50,60)] and counts == [1] + exp_counts + [1]

    conns, counts = _find_connected_intervals([(0,0)] + spans + [(50,60), (59,65)])
    assert conns == [(0,0)] + exp_conns + [(50,65)] and counts == [1] + exp_counts + [2]