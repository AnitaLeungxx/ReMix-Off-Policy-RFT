# for evaluation only

from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math, multiply, countdown


from deepscaler.rewards.math_reward import deepscaler_reward_fn_for_evaluator
def _select_rm_score_fn(data_source):
    assert data_source != 'openai/gsm8k'
    assert data_source != 'lighteval/MATH'
    assert "multiply" not in data_source or "arithmetic" in data_source
    if "countdown" in data_source:
        return countdown.compute_score_for_eval
    else:        
        return deepscaler_reward_fn_for_evaluator


class RewardManagerForEval():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""
        print("RewardManagerForEval")
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        rewardstrict_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        is_correct_format_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        is_correct_finalanswer_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        problem_ids = []

        already_print_data_sources = {}

        from concurrent.futures import ThreadPoolExecutor
        from typing import Dict, Any
        #import threading
        # Thread-safe dict for tracking printed data sources
        # print_lock = threading.Lock()
        
        def process_item(args):
            i, data_item, already_print_data_sources = args
            
            try:
                problem_id = data_item.non_tensor_batch['extra_info']['index']
            except KeyError:
                problem_id = None  
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)
            score, reward, is_correct_format, is_correct_finalanswer = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
            
            # with print_lock:
            #     if data_source not in already_print_data_sources:
            #         already_print_data_sources[data_source] = 0

            #     if already_print_data_sources[data_source] < self.num_examine:
            #         already_print_data_sources[data_source] += 1
            #         print(sequences_str)      
            return i, score, valid_response_length, reward, is_correct_format, is_correct_finalanswer, problem_id

        # Process items in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=96) as executor:
            args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
            results = list(executor.map(process_item, args))

        # Fill reward tensor with results
        for i, score, valid_response_length, reward, is_correct_format, is_correct_finalanswer, problem_id in results:
            reward_tensor[i, valid_response_length - 1] = score
            rewardstrict_tensor[i, valid_response_length - 1] = reward
            is_correct_finalanswer_tensor[i, valid_response_length - 1] = is_correct_finalanswer
            is_correct_format_tensor[i, valid_response_length - 1] = is_correct_format
            problem_ids.append(problem_id)

        return reward_tensor, rewardstrict_tensor, is_correct_finalanswer_tensor, is_correct_format_tensor, problem_ids