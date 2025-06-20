# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

from abc import ABC, abstractmethod
from copy import copy
import math
from typing import Iterable
import multi_choices_parser
import torch

from multi_choices_parser import MultiChoicesParser, DEFAULT_END_SYMB
from functools import lru_cache


class TokenLimiter(ABC):
    @abstractmethod
    def step(self, token: int) -> None:
        """Inject one token into the limiter which influences the state of the limiter

        Args:
            token (int): Token ID from the vocabulary of the tokenizer
        """
        pass

    @abstractmethod
    def authorized_tokens(self) -> list[int]:
        """Get the list authorized of authorized tokens for the next step"""
        pass

    @abstractmethod
    def copy(self) -> "TokenLimiter":
        """Return a stateful copy of this limiter."""
        pass

    @abstractmethod
    def is_at_initial_state(self) -> bool:
        """Is the limiter at its initial state?"""
        pass


class TokenLimitersCombinator(TokenLimiter):
    def __init__(self, token_limiters: list[TokenLimiter]) -> None:
        super().__init__()
        self.token_limiters = tuple(token_limiters)
    
    @property
    def finished(self):
        return all(x.finished for x in self.token_limiters)
    
    def step(self, token: int) -> None:
        if self.finished:
            raise multi_choices_parser.parser.ParserError("The parser is in 'finished' mode")
        for tl in self.token_limiters:
            if not tl.finished:
                tl.step(token)

    @lru_cache(maxsize=4096)
    def authorized_tokens(self) -> list[int]:
        if self.finished:
            return []
        return list(
            set(
                tok
                for token_limiter in self.token_limiters
                for tok in token_limiter.authorized_tokens()
            )
        )

    def copy(self) -> TokenLimiter:
        return TokenLimitersCombinator(tuple(tl.copy() for tl in self.token_limiters))

    def is_at_initial_state(self) -> bool:
        return all(tl.is_at_initial_state() for tl in self.token_limiters)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, TokenLimitersCombinator):
            return False
        return set(self.token_limiters) == set(value.token_limiters)

    def __hash__(self) -> int:
        return hash(self.token_limiters)


class MultiChoicesLimiter(TokenLimiter):
    def __init__(self, tokens: list[list[int]], eos_token_id: int) -> None:
        super().__init__()
        self.parser = MultiChoicesParser([tokens])
        self.eos_token_id = eos_token_id

    def step(self, token: int) -> None:
        if token == self.eos_token_id:
            token = DEFAULT_END_SYMB
        self.parser.step(token)

    @lru_cache(maxsize=4096)
    def authorized_tokens(self) -> list[int]:
        return [
            x if x is not DEFAULT_END_SYMB else self.eos_token_id for x in self.parser.next()
        ]

    def copy(self) -> TokenLimiter:
        cp = copy(self)
        cp.parser = cp.parser.copy()
        return cp

    def is_at_initial_state(self) -> bool:
        return self.parser.is_at_initial_state
    
    @property
    def finished(self):
        return self.parser.finished

    # Defining __eq__ and __hash__ for LRU cache for authorized_tokens method
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, MultiChoicesLimiter):
            return False
        return self.eos_token_id == value.eos_token_id and self.parser == value.parser

    def __hash__(self) -> int:
        return sum(hash(x) for x in (self.eos_token_id, self.parser))


class DoesNothingLimiter(TokenLimiter):
    def step(self, token: int) -> None:
        return

    def authorized_tokens(self) -> list[int]:
        return slice(None, None, None)

    def copy(self) -> TokenLimiter:
        return self

    def is_at_initial_state(self) -> bool:
        return True
    
    @property
    def finished(self):
        return False


def select_mask_list(l: list, mask: Iterable[bool]) -> list:
    """Select a portion of a list using a boolean mask"""
    return [x for x, b in zip(l, mask) if b]


def select_index_list(l: list, index: Iterable[int]) -> list:
    """Select a portion of a list using a boolean mask"""
    return [l[idx] for idx in index]


class ListTokensLimiter(TokenLimiter):
    def __init__(self, list_tokens: list[int]) -> None:
        super().__init__()
        self.list_tokens = list_tokens

    def step(self, token: int) -> None:
        return

    def authorized_tokens(self) -> list[int]:
        return self.list_tokens

    def copy(self) -> TokenLimiter:
        return self

    def is_at_initial_state(self) -> bool:
        return True


def compute_scores(
    previous_scores: torch.Tensor,
    new_log_probs: torch.Tensor,
    score_fn: str,
    num_tokens_produced: int,
):
    if score_fn == "logprob":
        new_scores = previous_scores + new_log_probs
    elif score_fn == "perplexity":
        if num_tokens_produced == 0:
            logprobs = new_log_probs
        else:
            logprobs = torch.log(previous_scores) * -num_tokens_produced + new_log_probs
        new_scores = torch.exp(-1 / (num_tokens_produced + 1) * logprobs)
    return new_scores


def enforce_token_limiter(token_limiters: list[TokenLimiter], log_probs: torch.Tensor):
    mask = torch.ones_like(log_probs, dtype=torch.bool)
    for i, token_limiter in enumerate(token_limiters):
        auth_toks = token_limiter.authorized_tokens()
        mask[i, auth_toks] = False
    log_probs[mask] = -torch.inf


def batched_inference_for_next_token_probs(
    hf_model, input_ids: torch.LongTensor, batch_size: int
) -> torch.Tensor:
    if batch_size is None:
        batch_size = input_ids.shape[0]
    log_probs = []
    for i in range(0, input_ids.shape[0], batch_size):
        inp = input_ids[i : i + batch_size]
        try:
            out = torch.log_softmax(hf_model(inp).logits[:, -1, :], dim=-1)
        except NotImplementedError:
            # Special case for random LM
            vocab_size = hf_model.state_dict()["vocab_size"].item()
            dtype = hf_model.state_dict()["dummy"].dtype
            out = torch.randn(
                (inp.shape[0], vocab_size), device=input_ids.device, dtype=dtype
            )
        log_probs.append(out)
    return torch.cat(log_probs, dim=0)


@torch.no_grad()
def beam_search(
    hf_model,
    input_ids: torch.LongTensor,
    beam_width: int,
    max_new_tokens: int,
    eos_token_id: int,
    score_fn="logprob",
    token_limiter: TokenLimiter = None,
    batch_size=32,
):
    assert (
        len(input_ids.shape) == 2 and input_ids.shape[0] == 1
    ), "Requirement: input_ids.shape == (1,S) for some integer S"
    device = input_ids.device
    input_ids = input_ids.cpu()
    finished = torch.tensor([False])
    unfinished = ~finished
    initial_input_length = input_ids.shape[1]
    scores = torch.tensor([0.0])
    token_limiter = DoesNothingLimiter() if token_limiter is None else token_limiter
    assert (
        token_limiter.is_at_initial_state()
    ), "Requirement: token limiter must be at its initial state"
    token_limiters = [token_limiter]

    # past_key_values = None
    num_new_tok = 0

    while unfinished.any() and num_new_tok < max_new_tokens:
        # NOTE: Finished hypotheses are guaranteed to be at the start of input_ids (if they exist) ... (1)
        # Take finished inputs only
        inputs_unfinished = input_ids[unfinished]
        num_finished = finished.sum().item()

        # Inference (TODO: Accelerate inference using past_key_values)
        log_probs_next_tok_unfinished = batched_inference_for_next_token_probs(
            hf_model, inputs_unfinished.to(device), batch_size
        ).cpu()

        token_limiters_unfinished = select_mask_list(
            token_limiters, unfinished.tolist()
        )
        enforce_token_limiter(token_limiters_unfinished, log_probs_next_tok_unfinished)

        # top-tokens computation (this step needs (1) to work properly)
        probs_unfinished, top_tokens_unfinished = log_probs_next_tok_unfinished.topk(
            beam_width, dim=-1
        )
        probs_unfinished, top_tokens_unfinished = probs_unfinished.view(
            -1
        ), top_tokens_unfinished.view(-1)
        idx_in = torch.arange(
            num_finished,
            num_finished + inputs_unfinished.shape[0],
            device=inputs_unfinished.device,
        ).repeat_interleave(beam_width)

        # Compute new scores of each hypotheses
        pre_scores_unfinished = compute_scores(
            scores[idx_in], probs_unfinished, score_fn, num_new_tok
        )
        pre_scores = torch.cat([scores[finished], pre_scores_unfinished])

        # Keep the best hypotheses (input_ids, scores, token_limiters)
        _, top_idx = pre_scores.topk(beam_width)
        top_idx = top_idx[~torch.isinf(pre_scores[top_idx])]  # Remove -inf values
        scores = pre_scores[top_idx]
        input_ids = torch.cat([input_ids[finished], input_ids[idx_in]])[top_idx]

        token_limiters = select_index_list(
            token_limiters[:num_finished]
            + [token_limiters[idx].copy() for idx in idx_in.tolist()],
            top_idx,
        )

        # Append best tokens and EOS for finished hypotheses + update token limiters
        eos = torch.full((num_finished,), eos_token_id)
        best_tokens = torch.cat([eos, top_tokens_unfinished])[top_idx]

        input_ids = torch.cat([input_ids, best_tokens.unsqueeze(-1)], dim=1)
        for tok, token_limiter in zip(best_tokens, token_limiters):
            if not token_limiter.finished:
                token_limiter.step(tok.item())

        # Put finished hypotheses at the begining
        finished = input_ids[:, -1] == eos_token_id
        unfinished = ~finished
        input_ids = torch.cat([input_ids[finished], input_ids[unfinished]])
        scores = torch.cat([scores[finished], scores[unfinished]])
        token_limiters_finished = select_mask_list(token_limiters, finished.tolist())
        token_limiters = token_limiters_finished + select_mask_list(token_limiters, unfinished.tolist())
        finished = input_ids[:, -1] == eos_token_id
        unfinished = ~finished

        num_new_tok += 1

    # Order best sequences by score
    best_sequences = input_ids[:, initial_input_length:]
    sort_idx = torch.argsort(scores, descending=True)
    scores = scores[sort_idx]
    best_sequences = best_sequences[sort_idx]
    return best_sequences, scores

def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    """
    mask = -math.log(2) < x  # x < 0
    return torch.where(
        mask,
        (-x.expm1()).log(),
        (-x.exp()).log1p(),
    )

# def _set_column_to_value(tensor, column_index, value):
#     tensor[:, column_index] = value
#     return tensor

# @torch.no_grad()
# def beam_search_star(
#     hf_model,
#     input_ids: torch.LongTensor,
#     beam_width: int,
#     max_new_tokens: int,
#     eos_token_id: int,
#     score_fn="logprob",
#     token_limiter: TokenLimiter = None,
#     batch_size=32,
# ):
#     assert (
#         len(input_ids.shape) == 2 and input_ids.shape[0] == 1
#     ), "Requirement: input_ids.shape == (1,S) for some integer S"
#     device = input_ids.device
#     input_ids = input_ids.cpu()
#     finished = torch.tensor([False])
#     can_finish_finished_before = torch.tensor([False])
#     unfinished = ~finished
#     initial_input_length = input_ids.shape[1]
#     scores = torch.tensor([0.0])
#     log_probs_next_tok_before = None
#     token_limiter = DoesNothingLimiter() if token_limiter is None else token_limiter
#     assert (
#         token_limiter.is_at_initial_state()
#     ), "Requirement: token limiter must be at its initial state"
#     token_limiters = [token_limiter]

#     # past_key_values = None
#     num_new_tok = 0

#     # Previous logprobs

#     while unfinished.any() and num_new_tok < max_new_tokens:
#         # NOTE: Finished hypotheses are guaranteed to be at the start of input_ids (if they exist) ... (1)
#         # Take finished inputs only
#         inputs_unfinished = input_ids[unfinished]
#         num_finished = finished.sum().item()

#         # Inference (TODO: Accelerate inference using past_key_values)
#         log_probs_next_tok_unfinished = batched_inference_for_next_token_probs(
#             hf_model, inputs_unfinished.to(device), batch_size
#         ).cpu()

#         token_limiters_unfinished = select_mask_list(
#             token_limiters, unfinished.tolist()
#         )
#         enforce_token_limiter(token_limiters_unfinished, log_probs_next_tok_unfinished)

#         # top-tokens computation (this step needs (1) to work properly)
#         probs_unfinished, top_tokens_unfinished = log_probs_next_tok_unfinished.topk(
#             beam_width, dim=-1
#         )
#         probs_unfinished, top_tokens_unfinished = probs_unfinished.view(
#             -1
#         ), top_tokens_unfinished.view(-1)
#         idx_in = torch.arange(
#             num_finished,
#             num_finished + inputs_unfinished.shape[0],
#             device=inputs_unfinished.device,
#         ).repeat_interleave(beam_width)

#         # Compute new scores of each hypotheses
#         pre_scores_unfinished = compute_scores(
#             scores[idx_in], probs_unfinished, score_fn, num_new_tok
#         )
#         pre_scores = torch.cat([scores[finished], pre_scores_unfinished])

#         # Keep the best hypotheses (input_ids, scores, token_limiters)
#         _, top_idx = pre_scores.topk(beam_width)
#         top_idx = top_idx[~torch.isinf(pre_scores[top_idx])]  # Remove -inf values
#         scores = pre_scores[top_idx]
#         input_ids = torch.cat([input_ids[finished], input_ids[idx_in]])[top_idx]

#         token_limiters = select_index_list(
#             token_limiters[:num_finished]
#             + [token_limiters[idx].copy() for idx in idx_in.tolist()],
#             top_idx,
#         )

#         # Append best tokens and EOS for finished hypotheses + update token limiters
#         eos = torch.full((num_finished,), eos_token_id)
#         best_tokens = torch.cat([eos, top_tokens_unfinished])[top_idx]

#         input_ids = torch.cat([input_ids, best_tokens.unsqueeze(-1)], dim=1)

#         for tok, token_limiter in zip(best_tokens, token_limiters):
#             token_limiter.step(tok.item())
#         token_limiters_finished = select_mask_list(token_limiters, finished.tolist())
#         token_limiters = token_limiters_finished + select_mask_list(token_limiters, unfinished.tolist())

#         can_finish_now = []
#         for token_limiter in token_limiters[num_finished:]:
#             toks = token_limiter.authorized_tokens()
#             can_finish_now.append(eos_token_id in toks)
#         can_finish_now = torch.tensor(can_finish_now, dtype=torch.bool)

#         # Put finished hypotheses at the begining
#         unfinished = ~finished
#         input_ids = torch.cat([input_ids[finished], input_ids[can_finish_now], input_ids])

#         scores = torch.cat([scores[finished], scores[can_finish_now], scores[unfinished]])
#         if log_probs_next_tok_before is not None:
#             scores[finished][can_finish_finished_before] += log1mexp(_set_column_to_value(log_probs_next_tok_before[can_finish_finished_before].clone(), eos_token_id, -torch.inf).logsumexp(-1))

#         num_finished = finished.sum()
#         token_limiters = token_limiters[:num_finished] + select_mask_list(token_limiters, can_finish_now.tolist()) + token_limiters[num_finished:]
#         finished = torch.cat([torch.ones(num_finished, dtype=torch.bool), can_finish_now, torch.zeros(len(input_ids) - can_finish_now.sum(), dtype=torch.bool)])
#         unfinished = ~finished
        

#         num_new_tok += 1

#     # Order best sequences by score
#     best_sequences = input_ids[:, initial_input_length:]
#     sort_idx = torch.argsort(scores, descending=True)
#     scores = scores[sort_idx]
#     best_sequences = best_sequences[sort_idx]
#     return best_sequences, scores
