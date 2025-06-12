from datasets import load_dataset, concatenate_datasets
from torch.utils.data.dataset import Dataset
import random
import torch

############################
# 1. 전처리 함수 (정답을 text에 넣지 않고 label만 따로 저장)
############################

def preprocess_function_wic_no_answer(sample):
    """
    KoBEST WiC:
    - label: 0(아니오) or 1(예)
    - context_1, context_2, word
    """
    INSTRUCTION = (
        "다음 질문에 예, 아니오 중에서 답변하세요. "
        "그 외에는 아무것도 포함하지 않는 것을 엄수하십시오."
    )
    # text 끝에 '정답:'까지만 포함
    text = (
        f"문장1: {sample['context_1']}\n"
        f"문장2: {sample['context_2']}\n"
        f"질문: 문장1과 문장2에서 쓰인 단어 [{sample['word']}]가 같은 뜻으로 쓰였나?\n"
        f"{INSTRUCTION}"
        "\n정답:"  # 실제 답변(예/아니오)은 넣지 않음
    )
    
    # label은 별도로 저장 (0 또는 1)
    return {
        "text": text,
        "label": sample["label"]
    }

def preprocess_function_hellaswag_no_answer(sample):
    """
    KoBEST HellaSwag:
    - ctx: 전제(문맥)
    - endings: 4개의 선택지
    - label: 0,1,2,3
    """
    INSTRUCTION = (
        "전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. "
        "답변에는 0, 1, 2, 3 외에는 아무것도 포함하지 않는 것을 엄수하십시오."
    )
    # text 끝에 '정답:'까지만 포함
    text = (
        f"전제: {sample['context']}\n"
        f"0: {sample['endings'][0]}\n"
        f"1: {sample['endings'][1]}\n"
        f"2: {sample['endings'][2]}\n"
        f"3: {sample['endings'][3]}\n"
        f"{INSTRUCTION}"
        "\n정답:"
    )
    
    # label은 별도로 저장 (0~3)
    return {
        "text": text,
        "label": sample["label"]
    }

def preprocess_function_copa_no_answer(sample):
    """
    KoBEST COPA:
    - premise: 전제
    - question: "원인" or "결과"
    - alternative_1, alternative_2: 선택지
    - label: 0이면 1번 정답, 1이면 2번 정답
    """
    INSTRUCTION = (
        "전제 뒤에 올 문장으로 적절한 문장의 번호를 선택하세요. "
        "답변에는 1, 2 외에는 아무것도 포함하지 않는 것을 엄수하십시오."
    )
    CONNECTOR_MAP = {"원인": "왜냐하면", "결과": "그래서"}
    connector = CONNECTOR_MAP.get(sample["question"], "")

    # text 끝에 '정답:'까지만 포함
    text = (
        f"전제: {sample['premise']} {connector}\n"
        f"1: {sample['alternative_1']}\n"
        f"2: {sample['alternative_2']}\n"
        f"{INSTRUCTION}"
        "\n정답:"
    )

    # label은 별도로 저장 (0->1번, 1->2번)
    return {
        "text": text,
        "label": sample["label"]
    }

############################
# 2. get_dataset 함수 예시
############################

def get_dataset(dataset_name):
    """
    dataset_name: "wic", "hellaswag", "copa" 중 선택
    (train/validation 구분은 예시로 validation만 표시)
    """
    if dataset_name == "wic":
        dataset = load_dataset("skt/kobest_v1", "wic", split="validation")
        preprocess_function = preprocess_function_wic_no_answer

    elif dataset_name == "hellaswag":
        dataset = load_dataset("skt/kobest_v1", "hellaswag", split="validation")
        preprocess_function = preprocess_function_hellaswag_no_answer

    elif dataset_name == "copa":
        dataset = load_dataset("skt/kobest_v1", "copa", split="validation")
        preprocess_function = preprocess_function_copa_no_answer

    # map으로 text, label을 만들고 반환
    dataset = dataset.map(preprocess_function)
    return dataset

############################
# 3. BookCorpus 예시 (본문 그대로)
############################

def get_bookcorpus(tokenizer, n_samples, seq_len):
    traindata = load_dataset('bookcorpus', split='train')
    
    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
    return torch.cat(tokenized_samples, dim=0 )

############################
# 4. 여러 데이터 합치는 함수 예시
############################

def get_combination(n_samples):
    # wic, copa, hellaswag 각 n_samples/6 씩 뽑아서 합치는 예시
    datasets = []
    splitsize = [n_samples // 6] * 5 + [n_samples // 6 + n_samples % 6]
    dataset_names = ['wic', 'copa', 'hellaswag']
    
    for idx, dataset_name in enumerate(dataset_names):
        dataset = get_dataset(dataset_name)
        # 무작위 인덱스 뽑아서 부분 선택
        indices = [random.randint(0, len(dataset) - 1) for _ in range(splitsize[idx])]
        dataset = dataset.select(indices)
        # 이번 예시에서는 text와 label 컬럼만 사용
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ['text','label']])
        datasets.append(dataset)
    
    combined_dataset = concatenate_datasets(datasets)
    return combined_dataset, None, None
