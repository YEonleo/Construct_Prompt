import sys
import random
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria, 
    StoppingCriteriaList
)
from tqdm import tqdm
import wandb
import re

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        # stop_token_ids: 중단 기준으로 삼을 토큰 ID 리스트들의 리스트
        # 예: tokenizer("\n\n###", add_special_tokens=False).input_ids
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # input_ids: 현재까지 생성된 전체 토큰 ID 시퀀스
        for stop_ids in self.stop_token_ids:
            # 생성된 시퀀스의 끝 부분이 stop_ids와 일치하는지 확인
            if len(input_ids[0]) >= len(stop_ids) and torch.all(input_ids[0][-len(stop_ids):] == torch.tensor(stop_ids, device=input_ids.device)):
                return True # 중단 조건 충족
        return False # 계속 생성

#################################
# 전처리 함수들 수정
#################################
def preprocess_function_wic(sample):
    instruction = ( # 태스크 지시문
        "다음 질문에 예, 아니오 중에서 답변하세요. "
        "그 외에는 아무것도 포함하지 않는 것을 엄수하십시오."
    )
    gold = "예" if sample["label"] == 1 else "아니오"
    # 입력 부분만 포맷팅
    input_part = (
        f"문장1: {sample['context_1']}\n"
        f"문장2: {sample['context_2']}\n"
        f"질문: 문장1과 문장2에서 쓰인 단어 [{sample['word']}]가 같은 뜻으로 쓰였나?"
    )
    return {"instruction": instruction, "input_part": input_part, "gold": gold}

def preprocess_function_sentineg(sample):
    instruction = "주어진 문장에 대한 감정을 positive, negative 중에서 선택해 주세요." # 태스크 지시문
    gold = "positive" if sample["label"] == 1 else "negative"
    # 입력 부분만 포맷팅
    input_part = f"문장: {sample['sentence']}"
    return {"instruction": instruction, "input_part": input_part, "gold": gold}

def preprocess_function_boolq(sample):
    instruction = ( # 태스크 지시문
        "다음 질문에 예, 아니오 중에서 답하세요. "
        "그 외에는 아무것도 넣지 말아주십시오."
    )
    gold = "예" if sample["label"] == 1 else "아니오"
    # 입력 부분만 포맷팅
    input_part = (
        f"문서: {sample['paragraph']}\n"
        f"질문: {sample['question']}"
    )
    return {"instruction": instruction, "input_part": input_part, "gold": gold}

def preprocess_function_hellaswag(sample):
    instruction = ( # 태스크 지시문
        "전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. "
        "답변에는 0, 1, 2, 3 외에는 아무것도 포함하지 않는 것을 엄수하십시오."
    )
    gold = str(sample["label"])
    # 입력 부분만 포맷팅
    input_part = (
        f"전제: {sample['context']}\n"
        f"0: {sample['ending_1']}\n"
        f"1: {sample['ending_2']}\n"
        f"2: {sample['ending_3']}\n"
        f"3: {sample['ending_4']}"
    )
    return {"instruction": instruction, "input_part": input_part, "gold": gold}

def preprocess_function_copa(sample):
    instruction = ( # 태스크 지시문
        "전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. "
        "답변에는 1, 2 외에는 아무것도 포함하지 않는 것을 엄수하십시오."
    )
    CONNECTOR_MAP = {"원인": "왜냐하면", "결과": "그래서"}
    connector = CONNECTOR_MAP.get(sample["question"], "")
    gold = str(sample["label"] + 1)  # 0->"1", 1->"2"
    # 입력 부분만 포맷팅
    input_part = (
        f"전제: {sample['premise']} {connector}\n"
        f"1: {sample['alternative_1']}\n"
        f"2: {sample['alternative_2']}"
    )
    return {"instruction": instruction, "input_part": input_part, "gold": gold}

#################################
# 데이터셋 불러오기 - get_dataset 함수 내부의 .map 호출은 그대로 작동합니다.
#################################
def get_dataset(dataset_name, split="validation"):
    # Function remains the same as provided before
    if dataset_name == "wic":
        ds = load_dataset("skt/kobest_v1", "wic", split=split)
        ds = ds.map(preprocess_function_wic, remove_columns=ds.column_names) # 컬럼 정리 추가
    elif dataset_name == "copa":
        ds = load_dataset("skt/kobest_v1", "copa", split=split)
        ds = ds.map(preprocess_function_copa, remove_columns=ds.column_names) # 컬럼 정리 추가
    elif dataset_name == "hellaswag":
        ds = load_dataset("skt/kobest_v1", "hellaswag", split=split)
        ds = ds.map(preprocess_function_hellaswag, remove_columns=ds.column_names) # 컬럼 정리 추가
    elif dataset_name == "sentineg":
        ds = load_dataset("skt/kobest_v1", "sentineg", split=split)
        ds = ds.map(preprocess_function_sentineg, remove_columns=ds.column_names) # 컬럼 정리 추가
    elif dataset_name == "boolq":
        ds = load_dataset("skt/kobest_v1", "boolq", split=split)
        ds = ds.map(preprocess_function_boolq, remove_columns=ds.column_names) # 컬럼 정리 추가
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    # Ensure necessary columns exist
    if not all(col in ds.column_names for col in ["instruction", "input_part", "gold"]):
         raise ValueError(f"Dataset {dataset_name} missing required columns after preprocessing: instruction, input_part, gold")
    return ds

# get_combination 함수는 반환되는 데이터셋이 "instruction", "input_part", "gold" 컬럼을 갖도록 주의 필요
def get_combination(n_samples):
    """
    Combines samples from wic, copa, hellaswag. (Requires careful column handling)
    NOTE: This simple combination might lose task-specific instructions if not handled properly.
          A more robust approach would store task type and apply instructions later.
          For now, it assumes the base datasets have the required columns after mapping.
    """
    splitsize = [n_samples // 3]*2 + [n_samples // 3 + n_samples % 3]
    all_sets = []
    available_datasets = ["wic", "copa", "hellaswag"]
    num_datasets = len(available_datasets)
    splitsize = [n_samples // num_datasets] * (num_datasets -1) + [n_samples // num_datasets + n_samples % num_datasets]

    for idx, name in enumerate(available_datasets):
        try:
            dset = get_dataset(name, split="train") # Use train split for combination source? Or validation?
            if len(dset) > 0:
                current_splitsize = min(splitsize[idx], len(dset))
                if current_splitsize > 0:
                    indices = random.sample(range(len(dset)), current_splitsize)
                    dset = dset.select(indices)
                    # Add a 'task' column to identify origin if needed later
                    dset = dset.map(lambda example: {'task': name})
                    all_sets.append(dset)
                else:
                    print(f"[WARN] Not enough samples in {name} dataset for combination.")
            else:
                 print(f"[WARN] Dataset {name} is empty.")
        except Exception as e:
            print(f"[WARN] Could not load or process dataset {name} for combination: {e}")

    if not all_sets:
         raise ValueError("No datasets could be selected for combination.")

    # Ensure all datasets have the same columns before concatenating
    # This part needs careful implementation if columns differ
    common_columns = ["instruction", "input_part", "gold", "task"] # Example common columns
    processed_sets = []
    for dset in all_sets:
        # Select/rename columns to match the common structure
        # This is simplified; real implementation might need more logic
        if all(col in dset.column_names for col in common_columns):
             processed_sets.append(dset.select_columns(common_columns))
        else:
            print(f"[WARN] Dataset for task {dset[0]['task']} missing common columns, skipping.")


    if not processed_sets:
        raise ValueError("No datasets with common columns found for combination.")

    combined = concatenate_datasets(processed_sets)
    print(f"[INFO] Combined dataset created with {len(combined)} samples.")
    return combined


def parse_hellaswag_label(gen_text: str) -> str:
    # Function remains the same as provided before
    idx = gen_text.rfind("### 응답:") # 응답 마커 기준으로 찾기
    if idx == -1:
        idx = gen_text.rfind("답변:") # 이전 방식 호환
        if idx == -1:
            substring = gen_text
        else:
            substring = gen_text[idx + len("답변:"):]
    else:
        substring = gen_text[idx + len("### 응답:"):]
    match = re.search(r"([0-3])", substring)
    return match.group(1) if match else ""

#################################
# 메인 실행
#################################
def main():
    parser = argparse.ArgumentParser("KoBEST ICL Evaluation (llm-kr-eval style prompts)")
    # --- Arguments remain the same ---
    parser.add_argument("--dataset", type=str, required=True, help="wic, copa, hellaswag, sentineg, boolq, combination")
    parser.add_argument("--split", type=str, default="validation", help="train, validation, test")
    parser.add_argument("--model", type=str, default="beomi/Llama-3-Open-Ko-8B")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation (usually 1)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="Max tokens to generate for answer")
    parser.add_argument("--project_name", type=str, default="KoBEST_ICL_Eval", help="wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Custom wandb run name")
    parser.add_argument("--wandb_group", type=str, default=None, help="Wandb group name for organizing runs")
    parser.add_argument("--mode", type=str, default="base", choices=["base", "store"], help="'base' for EM eval, 'store' to save correct/mistakes")
    parser.add_argument("--output_json_mistakes", type=str, default="mistakes.json", help="File for incorrect samples (if mode=store)")
    parser.add_argument("--output_json_correct", type=str, default="correct.json", help="File for correct samples (if mode=store)")
    parser.add_argument("--scenario_file", type=str, default="icl_scenarios.json", help="Path to the JSON file with ICL scenarios.")
    parser.add_argument("--icl_copa", action="store_true", help="Apply ICL for CoPA")
    parser.add_argument("--icl_boolq", action="store_true", help="Apply ICL for BoolQ")
    parser.add_argument("--icl_hellaswag", action="store_true", help="Apply ICL for HellaSwag")
    parser.add_argument("--icl_sentineg", action="store_true", help="Apply ICL for SentiNeg")
    parser.add_argument("--icl_wic", action="store_true", help="Apply ICL for WiC")
    parser.add_argument("--copa_scenario", type=str, default=None, choices=["scenario1", "scenario2"], help="CoPA scenario name (requires --icl_copa)")
    parser.add_argument("--boolq_scenario", type=str, default=None, choices=["scenario1", "scenario2"], help="BoolQ scenario name (requires --icl_boolq)")
    parser.add_argument("--hellaswag_scenario", type=str, default=None, choices=["scenario1", "scenario2"], help="HellaSwag scenario name (requires --icl_hellaswag)")
    parser.add_argument("--sentineg_scenario", type=str, default=None, choices=["scenario1", "scenario2"], help="SentiNeg scenario name (requires --icl_sentineg)")
    parser.add_argument("--wic_scenario", type=str, default=None, choices=["scenario1", "scenario2"], help="WiC scenario name (requires --icl_wic)")

    args = parser.parse_args()

    # --- Load Scenarios ---
    all_scenarios = {}
    try:
        # Scenario file should now contain prompts formatted with ### markers
        with open(args.scenario_file, 'r', encoding='utf-8') as f:
            all_scenarios = json.load(f)
        print(f"[INFO] Loaded ICL scenarios from {args.scenario_file}")
    except FileNotFoundError:
        print(f"[WARN] Scenario file not found: {args.scenario_file}. Few-shot runs will effectively be zero-shot.")
    except json.JSONDecodeError:
        print(f"[ERROR] Error decoding JSON from {args.scenario_file}. Check file format.")
        return

    # --- wandb setup (remains the same) ---
    selected_scenario_name = None
    # (Logic to determine selected_scenario_name remains the same)
    if args.dataset == "copa" and args.icl_copa and args.copa_scenario: selected_scenario_name = f"copa_{args.copa_scenario}"
    elif args.dataset == "boolq" and args.icl_boolq and args.boolq_scenario: selected_scenario_name = f"boolq_{args.boolq_scenario}"
    elif args.dataset == "hellaswag" and args.icl_hellaswag and args.hellaswag_scenario: selected_scenario_name = f"hellaswag_{args.hellaswag_scenario}"
    elif args.dataset == "sentineg" and args.icl_sentineg and args.sentineg_scenario: selected_scenario_name = f"sentineg_{args.sentineg_scenario}"
    elif args.dataset == "wic" and args.icl_wic and args.wic_scenario: selected_scenario_name = f"wic_{args.wic_scenario}"

    if args.wandb_run_name: run_name = args.wandb_run_name
    else:
        run_name = f"{args.dataset}_{args.split}_{args.mode}"
        if selected_scenario_name:
            run_name += f"_{selected_scenario_name}"
        elif args.icl_copa or args.icl_boolq or args.icl_hellaswag or args.icl_sentineg or args.icl_wic:
            run_name += "_ICL_flagged"

    wandb.init(
        project=args.project_name,
        name=run_name,
        group=args.wandb_group,
        config=vars(args)
    )

    # --- 1) Load Model & Tokenizer (remains the same) ---
    print(f"[INFO] Loading model: {args.model}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", trust_remote_code=True)
        base_model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"[ERROR] Failed to load model or tokenizer: {e}"); wandb.finish(); return

    # --- 2) Prepare Dataset (Uses updated get_dataset) ---
    print(f"[INFO] Loading dataset: {args.dataset} (split={args.split})")
    try:
        if args.dataset == "combination":
             dataset = get_combination(args.max_samples if args.max_samples else 1000)
        else:
            dataset = get_dataset(args.dataset, split=args.split)
        if args.max_samples is not None and args.dataset != "combination":
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        print(f"[INFO] Using {len(dataset)} samples for evaluation.")
        # Check for new columns
        if not all(col in dataset.column_names for col in ["instruction", "input_part", "gold"]):
             raise ValueError("Dataset must contain 'instruction', 'input_part', 'gold' columns after preprocessing.")
    except Exception as e:
        print(f"[ERROR] Failed to load or prepare dataset: {e}"); wandb.finish(); return

    # --- 3) Evaluation Loop ---
    n_correct = 0
    n_total = len(dataset)
    mistakes_samples = []
    correct_samples = []

    # --- Determine ICL settings (remains the same) ---
    use_icl = False
    scenario_key = None
    # (Logic to determine use_icl and scenario_key remains the same)
    if args.dataset == "copa" and args.icl_copa and args.copa_scenario: use_icl = True; scenario_key = args.copa_scenario
    elif args.dataset == "boolq" and args.icl_boolq and args.boolq_scenario: use_icl = True; scenario_key = args.boolq_scenario
    elif args.dataset == "hellaswag" and args.icl_hellaswag and args.hellaswag_scenario: use_icl = True; scenario_key = args.hellaswag_scenario
    elif args.dataset == "sentineg" and args.icl_sentineg and args.sentineg_scenario: use_icl = True; scenario_key = args.sentineg_scenario
    elif args.dataset == "wic" and args.icl_wic and args.wic_scenario: use_icl = True; scenario_key = args.wic_scenario

    scenario_text = "" # This will hold the pre-formatted scenario block from JSON
    if use_icl and args.dataset != "combination" and args.dataset in all_scenarios and scenario_key in all_scenarios[args.dataset]:
        scenario_text = all_scenarios[args.dataset][scenario_key]
        print(f"[INFO] Applying {args.dataset} scenario '{scenario_key}' from file (formatted with ###).")
    elif use_icl:
         print(f"[WARN] ICL flag for {args.dataset} is set, but scenario '{scenario_key}' not found or specified correctly. Running Zero-Shot style.")
         use_icl = False # Fallback to zero-shot formatting if scenario text is missing
         
    # --- stop_token_ids for stopping criteria ---
    stop_sequence = "\n\n###"
    stop_ids = tokenizer(stop_sequence, add_special_tokens=False).input_ids
    stopper = StopOnTokens(stop_token_ids=[stop_ids])
    stopping_criteria_list = StoppingCriteriaList([stopper])

    # --- Define meta-instruction ---
    meta_instruction = "다음은 작업을 설명하는 지침과 컨텍스트 입력의 조합입니다. 요구를 적절하게 만족시키는 응답을 적으십시오.\n\n"

    if args.batch_size != 1:
        print("[WARN] Batch size > 1 not fully implemented, using batch_size=1.")
        args.batch_size = 1

    for i in tqdm(range(n_total), desc=f"Evaluating {args.dataset} ({run_name})"):
        example = dataset[i]
        # Retrieve new fields from the processed example
        task_instruction = example["instruction"]
        input_part = example["input_part"]
        gold = example["gold"]
        current_task = example.get("task", args.dataset) # For combination dataset

        # --- Construct Final Prompt (Approach 2) ---
        final_prompt = ""
        if use_icl and scenario_text: # Few-Shot case
            # scenario_text is loaded from JSON, already contains ### markers and instruction
            final_prompt = meta_instruction + scenario_text + f"\n\n### 입력:\n{input_part}\n\n### 응답:\n"
        else: # Zero-Shot case (or fallback)
            final_prompt = meta_instruction + f"### 지시:\n{task_instruction}\n\n### 입력:\n{input_part}\n\n### 응답:\n"
        # --- Tokenize and Generate (remains similar) ---
        try:
            inputs = tokenizer(final_prompt, return_tensors="pt", truncation=True, max_length=4096 - args.max_new_tokens) # Increased context length
            inputs = {k: v.to(base_model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = base_model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    stopping_criteria=stopping_criteria_list,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id # Explicitly set EOS token ID
                )

            # Decode full output and slice based on the final prompt length
            full_gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"[INFO] Full generated text for sample {i}:\n{full_gen_text}") # Debugging output

            # Improved slicing to find response after "### 응답:"
            response_marker = "### 응답:"
            marker_index = full_gen_text.rfind(response_marker)
            if marker_index != -1:
                 # Extract text after the last "### 응답:" marker in the *entire* generated text
                 # This assumes the model repeats the prompt structure
                 gen_answer = full_gen_text[marker_index + len(response_marker):].strip()
            else:
                 # Fallback: slice based on input prompt length (less reliable if model doesn't repeat prompt)
                 gen_answer = full_gen_text[len(final_prompt):].strip() # Might still contain parts of the prompt
        except Exception as e:
            print(f"\n[ERROR] Generation failed for sample {i}: {e}")
            print(f"Prompt length: {len(tokenizer.encode(final_prompt))}") # Log token length
            gen_answer = ""

        # --- Parse Prediction (remains similar, but uses current_task) ---
        predicted_label = None
        first_token = ""
        if current_task == "hellaswag": # Check task type first
            # Pass the generated answer part for parsing HellaSwag
            first_token = parse_hellaswag_label(gen_answer) # Pass only the answer part
            if first_token in ["0", "1", "2", "3"]:
                 predicted_label = first_token
        elif gen_answer:
            answer_parts = gen_answer.split()
            if answer_parts:
                first_token_raw = answer_parts[0]
                first_token = first_token_raw.strip().rstrip('.:!?,')

                # Task-specific Label Mapping (using current_task)
                if current_task == "sentineg": # Use 'positive'/'negative'
                    if first_token.lower().startswith("positive"): predicted_label = "positive"
                    elif first_token.lower().startswith("negative"): predicted_label = "negative"
                elif current_task == "wic": # Use '예'/'아니오'
                    if first_token.startswith("예"): predicted_label = "예"
                    elif first_token.startswith("아니오"): predicted_label = "아니오"
                elif current_task == "boolq": # Use '예'/'아니오'
                    if first_token.startswith("예"): predicted_label = "예"
                    elif first_token.startswith("아니오"): predicted_label = "아니오"
                elif current_task == "copa": # Use '1'/'2'
                    if first_token.startswith("1"): predicted_label = "1"
                    elif first_token.startswith("2"): predicted_label = "2"
        # Ensure first_token has a value even if parsing fails
        if not first_token and gen_answer:
             first_token = gen_answer.split()[0] if gen_answer.split() else ""


        # --- Compare and Store (Store final_prompt) ---
        is_correct = (predicted_label == gold)
        if is_correct:
            n_correct += 1
        print(f"[INFO] Sample {i}: Gold: {gold}, Prediction: {predicted_label}, Correct: {is_correct}")
        if args.mode == "store":
            sample_info = {
                "index": i,
                "task": current_task,
                "instruction": task_instruction,
                "input_part": input_part,
                "prompt_used": final_prompt, # Store the actual prompt sent to model
                "gold": gold,
                "prediction": predicted_label if predicted_label is not None else "None",
                "raw_prediction_token": first_token,
                "model_output": gen_answer
            }
            if is_correct: correct_samples.append(sample_info)
            else: mistakes_samples.append(sample_info)

    # --- Final Results & Logging (remains the same) ---
    em_score = (n_correct / n_total) * 100 if n_total > 0 else 0.0
    print(f"\n[RESULT] Dataset: {args.dataset}, Split: {args.split}, Mode: {args.mode}")
    if use_icl and scenario_text: print(f"[RESULT] ICL Scenario: {selected_scenario_name}")
    print(f"[RESULT] Total: {n_total}, Correct: {n_correct}, EM Score: {em_score:.2f}%")
    wandb.log({"em_score": em_score, "total_samples": n_total, "correct_samples": n_correct})

    if args.mode == "store":
        try:
            with open(args.output_json_mistakes, "w", encoding="utf-8") as f_m: json.dump(mistakes_samples, f_m, ensure_ascii=False, indent=2)
            print(f"[INFO] Mistakes ({len(mistakes_samples)}) saved to {args.output_json_mistakes}")
            with open(args.output_json_correct, "w", encoding="utf-8") as f_c: json.dump(correct_samples, f_c, ensure_ascii=False, indent=2)
            print(f"[INFO] Correct ({len(correct_samples)}) saved to {args.output_json_correct}")
        except Exception as e: print(f"[ERROR] Failed to save output JSON files: {e}")

    wandb.finish()
    print("[INFO] Evaluation finished.")

if __name__ == "__main__":
    main()