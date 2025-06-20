# Software Name : TimeStress
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT

# This software is distributed under the MIT License,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html

# Authors: see CONTRIBUTORS.md
# Software description: Evaluating the Consistency of the Temporal Representation of Facts in Large Language Models

from ke_utils.general import search_str_in_tokens


def remove_successive_duplicates(iterator, return_duplicate_count=False, 
                                 comp_func=lambda x,y : x.__eq__(y), 
                                 select_func=lambda l : l[0]):
    def output():
        if return_duplicate_count:
            return select_func(equals), count
        else:
            return select_func(equals)
    iterator = iter(iterator)
    
    try:
        last_item = next(iterator)
        count = 1
        equals = [last_item]
        for current_item in iterator:
            if not comp_func(current_item, last_item):
                yield output()
                last_item = current_item
                count = 1
                equals = [last_item]
            else:
                count += 1
                equals.append(current_item)
        yield output()
        
    except StopIteration:
        yield output()


def intervals_overlap(interval1 : tuple[int, int], interval2: tuple[int, int]) -> bool:
    """
    Checks if two left-closed and right-open intervals overlap.

    Args:
        interval1 (Tuple[int, int]): The first interval represented as (a, b),
                                      where 'a' is the start (inclusive) and 'b' is the end (exclusive).
        interval2 (Tuple[int, int]): The second interval represented as (a', b'),
                                      where 'a'' is the start (inclusive) and 'b'' is the end (exclusive).

    Returns:
        bool: True if the intervals overlap, False otherwise.
    """
    a, b = interval1  # interval1 is [a, b)
    a_prime, b_prime = interval2  # interval2 is [a', b')
    return not (b <= a_prime or b_prime <= a)

def union_intervals(intervals: list[tuple[int, int]]) -> tuple[int, int]:
    """
    Computes the union of a list of left-closed and right-open intervals.

    Args:
        intervals (List[Tuple[int, int]]): A list of intervals represented as tuples (a, b),
                                             where 'a' is the start (inclusive) and 'b' is the end (exclusive).

    Returns:
        Tuple[int, int]: A tuple representing the union of the input intervals as (min, max),
                         or an empty tuple if there are no intervals.
    """
    if not intervals:
        raise ValueError('Input interval list is empty')

    # Initialize the union bounds
    min_bound = float('inf')
    max_bound = float('-inf')

    for interval in intervals:
        min_bound = min(min_bound, interval[0])
        max_bound = max(max_bound, interval[1])

    return (min_bound, max_bound)

def _find_connected_intervals(intervals: list[tuple[float, float]], sort=False) -> tuple[list[tuple[float, float]], list[int]]:
    # This function assumnes intervals is already sorted (by first then second element)
    # If not sorted, set sort=True
    if len(intervals) == 0:
        return [], []
    if sort:
        intervals = sorted(intervals)
    connected_intervals = []
    counts = []
    scope = intervals[0]
    count = 0
    for a, b in intervals:
        if a < scope[1] or a == b:
            scope = scope[0], max(scope[1], b)
            count += 1
        else:
            counts.append(count)
            connected_intervals.append(scope)
            count = 1
            scope = a, b
    counts.append(count)
    connected_intervals.append(scope)

    return connected_intervals, counts

def get_tokens(tokenizer, text, return_input_ids=False) -> tuple[list[str], list[int]]:
    """Partition text just how the tokenizer would have done it. 
    Returns the list of tokens (list[str]) and the sub-character token count (list[int]).
    
    The sub-character token count contains a list of integers such that its sum is equal to the length of the list of tokens. If an integer from 
    this list, at some position i, is superior to 1, for example 2, you will notice that returned tokens from i to i+2 are the same one character and it means
    that the tokenizer used 2 tokens to represent a single character. 
    
    Most of the time all the sub-character token count is equal to 1."""
    enc = tokenizer.encode_plus(text, return_offsets_mapping=True)
    spans = enc.offset_mapping
    n = len(enc.input_ids)

    # spans should be already sorted by the first then second element of intervals
    spans, subchar_token_count = _find_connected_intervals(spans)
    tokens = [text[a:b] for a,b in spans]
    assert ''.join(tokens) == text, """When encoding the text using the tokenizer, the concatenation of tokens obtained through offset mapping did not match the original text
Original text: "%s"
tokens obtained with offset mapping: %s""" % (text, tokens)
    assert sum(subchar_token_count) == n
    res = tokens, subchar_token_count
    if return_input_ids:
        res = res + (enc.input_ids,)
    return res


def get_span_tokens(text : str, sub_text : str, tokenizer, return_input_ids_text=False, ignore_case=False) -> list[tuple[int,int]]:
    """Get the spans of tokens of text that contain sub_text. These spans are a set of intervals [a,b).

    Args:
        text (str): Text
        sub_text (str): Sub-text to look in text
        tokenizer: Huggingface tokenizer
        return_input_ids_text (bool, optional): Returns the input_ids of text in addition to the spans. Defaults to False.

    Returns:
        list[tuple[int,int]]: Spans of tokens
    """
    res = get_tokens(tokenizer, text, return_input_ids=return_input_ids_text)
    if not return_input_ids_text:
        tokens, subchar_tokens_count = res
    else:
        tokens, subchar_tokens_count, input_ids = res
    spans = search_str_in_tokens(sub_text, tokens, subchar_tokens_count, ignore_case)

    if return_input_ids_text:
        return spans, input_ids
    return spans