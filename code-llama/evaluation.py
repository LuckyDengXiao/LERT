import json
import os
from typing import List
import numpy as np
import argparse
import torch
from tqdm import tqdm
from transformers.trainer_utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

INPUT_PROMPT = '''Determine if the provided function contains any defects. A defect is any kind of issue that would prevent the function from compiling successfully, running correctly, executing efficiently, or adhering to good coding practices.
**Defective Example:**
- **Input:**
```
int divide(int numerator, int denominator) {{
    if (denominator == 0) {{
        throw "Division by zero condition!";
    }}
    return numerator / denominator;
}}
```
- **Output:**
The function is defective.

**Non-Defective Example:**
- **Input:**
```
int add(int a, int b) {{
    return a + b;
}}
```
- **Output:**
The function is non-defective.

**Your Function for Analysis:**

- **Input:**
```
{function}
```
- **Output:**
'''


def load_models_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        padding_side='left',
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        pad_token_id=tokenizer.pad_token_id,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path,
        pad_token_id=tokenizer.pad_token_id,
        trust_remote_code=True
    )
    if torch.cuda.is_bf16_supported():
        model.to(dtype=torch.bfloat16)
    return model, tokenizer


def format_example(sample):
    _input = INPUT_PROMPT.format(function=sample['func'])
    # _output = 'The function is defective.' if sample['target']==1 else 'The function is non-defective.'
    prompt = "<|user|>\n" + _input.strip() + "\n" + "<|assistant|>\n" + "The function is "
    return prompt


def get_logits(tokenizer, model, inputs: List[str]):
    input_ids = tokenizer(inputs, padding='longest')["input_ids"]
    input_ids = torch.tensor(input_ids, device=model.device)
    if torch.cuda.is_bf16_supported():
        input_ids.to(dtype=torch.bfloat16)

    if input_ids.shape[1] > args.max_seq_len:
        input_ids = input_ids[:, input_ids.shape[1] - args.max_seq_len + 1 :]
    tokens = {"input_ids": input_ids}
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    # print("input_ids")
    outputs = model(input_ids, attention_mask=attention_mask)["logits"]
    logits = outputs[:, -1, :]
    log_probs = torch.nn.functional.softmax(logits, dim=-1)
    return log_probs, {"tokens": tokens}


@torch.no_grad()
def eval_dataset(
    model,
    tokenizer,
    eval_data,
    save_result_dir=None,
    batch_size=1,
    **kwargs,
):
    result = []
    score = []

    choices = ["non", "defect"]
    all_probs = {"prob_defect": [], "prob_non": []}
    choices_ids = torch.tensor(
        tokenizer("non", add_special_tokens=False)["input_ids"] + 
        tokenizer("defect", add_special_tokens=False)["input_ids"]
    ).unsqueeze(0).to(model.device)

    idx_list = list(range(0, len(eval_data), batch_size))
    for idx in tqdm(idx_list):
        full_prompt_list = []
        answer_list = []
        sample_list = []
        for sample in eval_data[idx:idx+batch_size]:
            full_prompt = format_example(sample['sample'])
            if args.debug and idx<3:
                print(f"full_prompt: {full_prompt}")
            full_prompt_list.append(full_prompt)
            answer_list.append(sample['sample']['target'])
            sample_list.append(sample['sample'])

        logits, input_info = get_logits(tokenizer, model, full_prompt_list)
        # print(logits.shape)
        # print(choices_ids)
        # print(choices_ids.expand(logits.size(0), -1))
        # print(logits[0][1661], logits[0][23503])
        # print(logits.gather(1, choices_ids.expand(logits.size(0), -1)))
        softval = logits.gather(1, choices_ids.expand(logits.size(0), -1)).softmax(1)
        # exit()
        
        if softval.dtype in {torch.bfloat16, torch.float16}:
            softval = softval.to(dtype=torch.float32)
        probs = softval.detach().cpu().numpy()

        for i in range(len(probs)):
            for j, choice in enumerate(choices):
                all_probs[f"prob_{choice}"].append(probs[i][j])
            pred = {0: 0, 1: 1}[np.argmax(probs[i])]

            if answer_list != []:
                correct = 1 if pred == answer_list[i] else 0
                # print(pred, answer_list[i])
                score.append(correct)
                if args.debug:
                    print(f'idx:{i} pred: {pred} ref: {answer_list[i]}')
            result.append({'sample':sample_list[i], 'probs':probs[i].tolist(), 'pred':pred})
        # print current correct rate
        if idx % 100 == 0:
            print(f"Current correct rate: {np.mean(score)}")
            
    if save_result_dir:
        os.makedirs(save_result_dir, exist_ok=True)
        json.dump(result, open(os.path.join(save_result_dir, "result.json"), "w"), indent=4)
        with open(os.path.join(save_result_dir, "acc.txt"), "w") as f:
            f.write(f"{np.mean(score)}")

    print(f"Final correct rate: {np.mean(score)}")
    return result

def main(args):
    model, tokenizer = load_models_tokenizer(args)
    eval_data = json.load(open(args.eval_data_path, "r"))
    result = eval_dataset(
        model,
        tokenizer,
        eval_data,
        save_result_dir=args.output_data_path,
        batch_size=args.batch_size
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Checkpoint path",
        default="Qwen/Qwen-7B",
    )
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")

    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title="Evaluation options")
    group.add_argument("-d", "--eval-data-path", type=str, help="Path to eval data")
    group.add_argument("-o", "--output-data-path", type=str, help="Path to output data")
    group.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Size of the output generated text.",
    )
    group.add_argument(
        "--debug", action="store_true", default=False, help="Print infos."
    )
    group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="batch size",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    main(args)