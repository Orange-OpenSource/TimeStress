# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models
import numpy as np
import torch
from ke_utils.beamsearch import beam_search
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_beam_search():
    # Results of beam search were compared with huggingface implementation (https://huggingface.co/spaces/m-ric/beam_search_visualizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained("gpt2", device_map=device)
    model.eval()
    tok = AutoTokenizer.from_pretrained("gpt2")
    prompt = "The capital of France is"
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    sequences, scores = beam_search(
        model, input_ids, beam_width=3, max_new_tokens=1, eos_token_id=tok.eos_token_id
    )
    sequences = [tok.decode(s) for s in sequences]
    assert sequences == [" the", " now", " a"]
    assert np.isclose(
        scores.cpu().numpy(), np.array([-2.4699, -3.0377, -3.0756]), atol=0.0001
    ).all()

    sequences, scores = beam_search(
        model, input_ids, beam_width=3, max_new_tokens=2, eos_token_id=tok.eos_token_id
    )
    sequences = [tok.decode(s) for s in sequences]
    assert sequences == [" the capital", " now home", " now the"]
    assert np.isclose(
        scores.cpu().numpy(), np.array([-4.2437, -5.3013, -5.3408]), atol=0.0001
    ).all()

    sequences, scores = beam_search(
        model, input_ids, beam_width=3, max_new_tokens=3, eos_token_id=tok.eos_token_id
    )
    sequences = [tok.decode(s) for s in sequences]
    assert sequences == [" the capital of", " now home to", " now the capital"]
    assert np.isclose(
        scores.cpu().numpy(), np.array([-4.3194, -5.3057, -7.7173]), atol=0.0001
    ).all()

    sequences, scores = beam_search(
        model, input_ids, beam_width=3, max_new_tokens=4, eos_token_id=tok.eos_token_id
    )
    sequences = [tok.decode(s) for s in sequences]
    assert sequences == [
        " the capital of the",
        " the capital of France",
        " the capital of a",
    ]
    assert np.isclose(
        scores.cpu().numpy(), np.array([-5.5825, -5.9150, -7.1716]), atol=0.0001
    ).all()

    sequences, scores = beam_search(
        model, input_ids, beam_width=3, max_new_tokens=5, eos_token_id=tok.eos_token_id
    )
    sequences = [tok.decode(s) for s in sequences]
    assert sequences == [
        " the capital of France,",
        " the capital of France.",
        " the capital of the French",
    ]
    assert np.isclose(
        scores.cpu().numpy(), np.array([-6.9453, -7.1549, -7.5727]), atol=0.0001
    ).all()


# def test_beam_search_star():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = AutoModelForCausalLM.from_pretrained("gpt2", device_map=device)
#     model.eval()
#     tok = AutoTokenizer.from_pretrained("gpt2")
#     prompt = "The capital of France is"

#     toklim = MultiChoicesLimiter([
#         [6342, 31248],
#         [6342]
#     ],
#     eos_token_id=tok.eos_token_id)
#     input_ids : torch.Tensor = tok(prompt, return_tensors='pt').input_ids.to(device)


#     # Compute real solutions and their scores
#     logits : torch.Tensor = model(torch.cat([input_ids, torch.tensor([6342,31248], device=device).unsqueeze(0)], dim=-1)).logits[0]
#     logprobs = logits.log_softmax(-1)
#     probtok_1, probtok_2 = logprobs[-3, 6342], logprobs[-2, 31248]
#     prob_sol_1 = probtok_1 + probtok_2
#     prob_sol_2 = probtok_1 + log1mexp(probtok_2)
#     true_solutions = torch.tensor([[6342, 31248], [6342, tok.eos_token_id]])
#     true_scores = torch.stack([prob_sol_1, prob_sol_2])
#     if prob_sol_2 > prob_sol_1:
#         true_solutions = true_solutions.flip(0)
#         true_scores = true_scores.flip(0)

#     solutions, scores = beam_search_star(model, input_ids, beam_width=5, 
#                      eos_token_id=tok.eos_token_id, max_new_tokens=10, 
#                      token_limiter=toklim)
        
#     assert (solutions == true_solutions).all()
#     assert (scores == true_scores).all()