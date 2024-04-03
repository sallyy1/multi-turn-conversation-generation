import torch
import os
import pandas as pd
from sacrebleu import corpus_bleu
# from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM, FlaxAutoModelForCausalLM, PreTrainedTokenizerFast
import re
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from nltk.translate.bleu_score import sentence_bleu
# from process import preprocess_function_generate, CustomDataset, replace_dialog
from process import MultiturnDataset, pad_collate_right, pad_collate_left, replace_dialog
from pprint import pprint
from args import parse_args
from transformers import pipeline
from torch.nn.functional import pad

CUDA_LAUNCH_BLOCKING=1

def main(args):        
    # model_path = "heegyu/kodialogpt-v1" # BASELINE 성능 측정
    # model_path = "heegyu/kogpt-j-base"
    # model_path = "/data/hyunkyung_lee/multi-turn/baseline/0206_leftpadding_AutoModelForCausalLM_warmup_heegyu_kogpt-j-base_AutoModelForCausalLM_v5_10%_14530_50/1"
    # model_path = "/data/hyunkyung_lee/multi-turn/baseline/0214_heegyu_kogpt-j-base_100%_72647_7/1"
    # model_path = "/data/hyunkyung_lee/multi-turn/baseline/0216_heegyu_kogpt-j-base_0.1%_73_2/36"
    # model_path = "/data/hyunkyung_lee/multi-turn/baseline/0216_heegyu_kogpt-j-base_10%_7265_2/57"
    # model_path = "/data/hyunkyung_lee/multi-turn/baseline/0216_heegyu_kogpt-j-base_100%_72647_2/76"
    # model_path = "/data/hyunkyung_lee/multi-turn/baseline/0216_heegyu_kogpt-j-base_100%_72647_2/76"
    # model_path = "/data/hyunkyung_lee/multi-turn/baseline/0229_heegyu_kogpt-j-base_100%_valid_random1%만사용_3e-5_72647_2/114"
    # model_path = "/data/hyunkyung_lee/multi-turn/baseline/0304_trainshuffle_False_heegyu_kogpt-j-base_100%_valid_random1%만사용_5e-5_72647_2/106"
    # 0308
    # model_path = "/data/hyunkyung_lee/multi-turn/baseline/0308_warmup5%_acc4_trainshuffle_False_heegyu_kogpt-j-base_10%_valid_random1%만사용_5e-5_7265_4/2epoch"
    # 0318
    # model_path = "/data/hyunkyung_lee/multi-turn/baseline/0312_warmup5%_a4_bs32_trainshuffle_False_heegyu_kogpt-j-base_100%_valid_random1%만사용_5e-5_36324_4/8625"
    # model_path = "/data/hyunkyung_lee/multi-turn/baseline/0315_이어서_False_acc4_bs32_trainshuffle_False_heegyu_kogpt-j-base_100%_valid_random1%만사용_1e-5_36324_4/1epoch"
    # model_path = "/data/hyunkyung_lee/multi-turn/baseline/0315_이어서_False_acc4_bs32_trainshuffle_False_heegyu_kogpt-j-base_100%_valid_random1%만사용_1e-5_36324_4/18555"
    # 0321
    # model_path = "/data/hyunkyung_lee/multi-turn/baseline/0315_이어서_False_acc4_bs32_trainshuffle_False_heegyu_kogpt-j-base_100%_valid_random1%만사용_1e-5_36324_4/2epoch"
    # 0322
    # model_path = "/data/hyunkyung_lee/multi-turn/baseline/0315_이어서_False_acc4_bs32_trainshuffle_False_heegyu_kogpt-j-base_100%_valid_random1%만사용_1e-5_36324_4/28680"
    model_path = "/data/hyunkyung_lee/multi-turn/baseline/0315_이어서_False_acc4_bs32_trainshuffle_False_heegyu_kogpt-j-base_100%_valid_random1%만사용_1e-5_36324_4/3epoch"

    
    # device = 'cuda'
    device = 'cuda:0'
    # device = 'cpu'

    model = AutoModelForCausalLM.from_pretrained(model_path) ###
    model.to(device)
    model.eval()
    batch_size = args.batch_size    
    
    # TEST 데이터셋 로드
    # Load entire validation dataset as test
    # test_dataset = load_dataset("csv", data_files="(new)Validation_history_and_gtresponse.tsv", split="train[:1%]", sep="\t") 
    # test_dataset = load_dataset("csv", data_files="0205_test_history_and_gtresponse.tsv", split="train[15%:]", sep="\t") ### 디버깅 해야 함
    # test_dataset = load_dataset("csv", data_files="0205_test_history_and_gtresponse.tsv", split="train[:10%]", sep="\t") ###
    ###test_dataset = load_dataset("csv", data_files="0207_test_history_and_gtresponse.tsv", split="train[:100%]", sep="\t") ###
    test_dataset = load_dataset("csv", data_files="0326_test_history_and_gtresponse_100.tsv", split="train[:100%]", sep="\t")
    print(len(test_dataset))
    

    # save to file
    # directory_path = "생성 결과 tsv/"
    # output_path = directory_path + "pipeline_" + "".join(model_path.split('/')[5:]) + ".tsv"
    # directory_path = "qualitative_evaluation/"
    directory_path = "inference_result/"
    output_path = directory_path + "pipeline_100_" + "".join(model_path.split('/')[5:]) + ".tsv"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    save_df = pd.DataFrame()

    # generation_args = dict(
    #     # repetition_penalty=1.3,
    #     no_repeat_ngram_size=4,
    #     # eos_token_id=375, # \n
    #     eos_token_id=2, # </s>
    #     pad_token_id=0,
    #     max_new_tokens=16,
    #     min_new_tokens=8,
    #     do_sample=False,
    #     top_p=0.7,
    #     # num_beams=4,
    #     early_stopping=True
    # )    

    # seed 고정
    seed_value = 42
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    
    if model_path == "heegyu/kodialogpt-v1":
        tokenizer_path = "heegyu/kodialogpt-v1"
        max_seq_len = 512
        generator = pipeline("text-generation", model=model_path, device=device)
        generation_args = dict(
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
            eos_token_id=375, # \n
            max_new_tokens=32,
            do_sample=True,
            top_p=0.7,
            early_stopping=True
        )
        
    else:
        tokenizer_path = "heegyu/kogpt-j-base"
        # max_seq_len = 256
        max_seq_len = 128
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) # MIT License
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
        '''기본'''
        # generation_args = dict(
        #     repetition_penalty=1.3,
        #     no_repeat_ngram_size=4,
        #     pad_token_id=tokenizer.pad_token_id,
        #     eos_token_id=tokenizer.eos_token_id, # </s>
        #     # max_length=max_seq_len,
        #     max_new_tokens=30,
        #     do_sample=True, # True or False
        #     # seed=42, # do_sample=True 시 seed 설정 (사용되는 두 개의 정수를 포함하는 샘플링을 제아하는 랜덤 시드)
            
        #     top_p=0.7, # 값이 낮을수록 견고하게 다음 토큰 선택하도록 함, 높을수록 다양한 선택을 하도록 허용함 # 0.7 or 0.3
            
        #     # num_beams=2,
        #     # early_stopping=True
        # )
        
        '''커스텀'''
        generation_args = dict(
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, # </s>
            # max_length=max_seq_len,
            max_new_tokens=30,
            do_sample=True, # True or False
            
            top_k = 50, # 확률 순위가 50위 밖인 토큰은 샘플링에서 제외
            top_p=0.1, # 값이 낮을수록 견고하게 다음 토큰 선택하도록 함, 높을수록 다양한 선택을 하도록 허용함 # 0.7 or 0.3
            num_beams=1,
            early_stopping=True
        )
        
        # generation_args = dict(
        #     max_length=256,
        #     num_beams=1,  # Beam size를 1로 설정하여 greedy search와 동일하게 만듭니다.
        #     no_repeat_ngram_size=4,  # no_repeat_ngram_size를 1로 설정하여 반복되는 토큰을 방지합니다.
        #     top_k=1,  # top_k를 1로 설정하여 각 단계에서 확률이 가장 높은 토큰만 선택합니다.
        #     top_p=1.0,  # top_p를 1.0으로 설정하여 상위 확률을 고려하지 않습니다.
        #     temperature=1.0,  # temperature를 1.0으로 설정하여 softmax 출력을 조절하지 않습니다.
        #     pad_token_id=tokenizer.pad_token_id,
        #     eos_token_id=tokenizer.eos_token_id,
        #     max_new_tokens=32,
        #     do_sample=True,  # do_sample을 False로 설정하여 샘플링을 사용하지 않습니다.
        #     early_stopping=False  # early_stopping을 False로 설정하여 조건에 따른 생성 조기 종료를 비활성화합니다.
        # )
    ### 생성 예시
    generator(
        ["0 : **는 게임 좋아하니\n1 :",
        "0 : 어제 강남에서 살인사건 났대 ㅜㅜ 너무 무서워\n1 : 헐 왜? 무슨 일 있었어?\n0 : 사진보니까 막 피흘리는 사람있고 경찰들이 떠서 제압하고 난리도 아니었다던데??\n1 :",
        "0 : 자기야 어제는 나한테 왜 그랬어?\n1 : 뭔 일 있었어?\n0 : 어떻게 나한테 말도 없이 그럴 수 있어? 나 진짜 실망했어\n1 : "],
        **generation_args
    )

    
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # 'Validation_history_and_gtresponse.tsv' 파일 읽기
    # df = pd.read_csv('Validation_instances.tsv', sep='\t')

    # vocab 사전에서 -1 인덱스 확인
    if -100 in tokenizer.get_vocab().values():
        print("Tokenizer vocab에 -1이 포함되어 있습니다.")
        IGNORE_INDEX = -100
    else:
        print("Tokenizer vocab에 -1이 포함되어 있지 않습니다.")
        IGNORE_INDEX = tokenizer.pad_token_id    
    
    
    
    
    
    
    
    print(" * =" * 50)
    
    prompt_list = []
    gt_response_list = []
    pipeline_generated_list = []
    
    
    total_bleu_scores = []
    generated_nothing_cnt = 0 # 빈 문자열 생성한 케이스
    
    example_idx = 0
    
    pipeline_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    cuda_error_count = 0 # 에러 케이스 카운트
    
    for example in test_dataset:
        try:
            ###print(example) # {'history': '<U> 안녕하세요 저는 20대 남성입니다. 반가워요! ', 'response': '저도 20대 여성입니다! 반가워요 ', 'labels': '<U> 안녕하세요 저는 20대 남성입니다. 반가워요! <S> 저도 20대 여성입니다! 반가워요 '}
            ###print(type(example))
            
            
            example_idx += 1
            INDEX = 900
            
            if example_idx > INDEX: print(f"# # # {example_idx} 번째 행 데이터 # # #")
            prompt = example["history"]
            
            if model_path == "heegyu/kodialogpt-v1":
                # prompt 전처리
                # 이렇게 맞춰줘야 함
                # "0 : 어제 강남에서 살인사건 났대 ㅜㅜ 너무 무서워\n1 : 헐 왜? 무슨 일 있었어?\n0 : 사진보니까 막 피흘리는 사람있고 경찰들이 떠서 제압하고 난리도 아니었다던데??\n1 :"
                prompt = prompt.lstrip()
                prompt = prompt.replace("<U>", "\n0 : ").replace("<S>", "\n1 : ") + "\n1 : " # <\s>0 : ... <\s>1 : ... 구조
                prompt = prompt.lstrip()
                if example_idx > INDEX: print("# # # # # # # # # # ")
                if example_idx > INDEX: print("prompt: ", prompt, end='\n\n')
                ### 기존: device-side assertions 에러
                inputs = pipeline_tokenizer(prompt)
                
                ### 수정: max_seq_len 만큼 앞에 잘라줌
                ###inputs = pipeline_tokenizer(prompt, padding=False, truncation='only_first', return_tensors="pt", max_length=(max_seq_len-32))
            
            # 너무 긴 대화는 앞에 truncate
            else:
                prompt, _ = replace_dialog(prompt, tokenizer.eos_token_id)
                ### BLEU: 0.23 빈 문자열: 8개/98개
                # inputs = pipeline_tokenizer(prompt, padding=False, truncation='only_first', return_tensors="pt", max_length=(max_seq_len-50))
                
                inputs = pipeline_tokenizer(prompt, padding=False, truncation='only_first', return_tensors="pt", max_length=(max_seq_len-30))
                # left 패딩 추가
                padding_length = (max_seq_len-30) - inputs['input_ids'].size(1)
                inputs['input_ids'] = pad(inputs['input_ids'], (padding_length, 0), value=tokenizer.pad_token_id)
                inputs['attention_mask'] = pad(inputs['attention_mask'], (padding_length, 0), value=0)  # 0으로 패딩

            # 결과 확인
            ###print(inputs['input_ids'])
            
            # inputs = torch.tensor(inputs, dtype=torch.long)
            # inputs.to(device) ###

            # if "min_new_tokens" in kwargs:
            #     min_new_tokens = kwargs["min_new_tokens"]
            #     del kwargs["min_new_tokens"]
            #     kwargs["min_length"] = inputs["input_ids"].shape[-1] + min_new_tokens

            # outs = model.generate(**inputs.to(device), **kwargs)
            # # outs = tokenizer.batch_decode(outs.sequences, skip_special_tokens=True)
            # outs = tokenizer.batch_decode(outs, skip_special_tokens=True)
            
            
            ### 디버깅 해야함        
            # if len(inputs['input_ids']) > max_seq_len:
            #     decoded = pipeline_tokenizer.decode(inputs, skip_special_tokens=True)
            #     print(inputs)
            #     print(type(inputs))
            #     print(len(prompt), len(decoded))
                
            #     prompt = prompt[-(len(decoded)-10):]
            
            gt_response_str = str(example["response"])
            
            result = generator(prompt, **generation_args)
            ###print(result)
            ###print(type(result))
            # prompt를 list로 인풋 시
            generated_text = result[0]["generated_text"]
            if example_idx > INDEX: print("generated_text: ", generated_text)
            new_generated_text = generated_text[len(prompt):]
            
            if example_idx > INDEX: print("new_generated_text: ", new_generated_text)
            if example_idx > INDEX: print("gt_response_str: ", gt_response_str)
            ###print("prompt 토크나이징 길이: ", len(inputs['input_ids'][0]))

            # save to file
            prompt_list.append(prompt)
            gt_response_list.append(gt_response_str)
            pipeline_generated_list.append(new_generated_text)
            

            if new_generated_text.strip() == "":
                print("# # # # # # # # # # ")
                print("prompt: ", prompt, end='\n\n')
                print(result)
                print("new_generated_text: ", new_generated_text)
                print("gt_response_str: ", gt_response_str)
                generated_nothing_cnt += 1
                    
            else:
                print("# # # # # # # # # # ")
                print("prompt: ", prompt, end='\n\n')
                print(result)
                print("new_generated_text: ", new_generated_text)
                print("gt_response_str: ", gt_response_str)        
                
            # BLEU calculate
            '''
            # reference = [
            #     'this is a dog'.split(),
            #     'it is dog'.split(),
            #     'dog it is'.split(),
            #     'a dog, it is'.split() 
            # ]
            # candidate = 'it is a dog'.split()
            '''
            
            reference = [list(gt_response_str)] # 각 음절 단위
            candidate = list(new_generated_text) # 각 음절 단위
            # reference = [gt_response_str] # 1개
            # candidate = [new_generated_text]
            
            
            # reference = [list(gt_response_str)]
            # candidate = list(model_generate_outputs_decoded_beam_i)

            
            print(reference)
            print(candidate)
            print('Individual 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
            print('Individual 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 1, 0, 0)))
            print('Individual 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 1, 0)))
            print('Individual 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 0, 1)))
            print('cumulative 4-gram BLEU score (BLEU-4 score): %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))
            print()
            
            
            this_bleu = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
            ###print("bleu score: ", this_bleu)
            total_bleu_scores.append(this_bleu)
        
        except RuntimeError as e:
            if "CUDA error: device-side assert triggered" in str(e):
                print("CUDA error: device-side assert triggered. Skipping the operation.")
                cuda_error_count += 1
            else:
                # 다른 CUDA 에러인 경우 예외를 다시 발생시킴
                raise e
        
    
# print('mean_ppl_score: ', perplexity_scores_epoch / len(valid_dataloader), end='\n')
# print('-> len(valid_dataloader): ', len(valid_dataloader))

    
    mean_bleu_score = sum(total_bleu_scores) / (len(total_bleu_scores)-cuda_error_count)
    print('TOTAL mean_bleu_score: ', mean_bleu_score, end='\n')
    print('-> len(total_bleu_scores): ', len(total_bleu_scores))
    print(f"생성을 하지 않은 케이스: {generated_nothing_cnt} 개")
    print(f'CUDA error(device-side assert triggered) 케이스: {cuda_error_count} 개')

    # save to file
    save_df["이전 문맥 promt"] = prompt_list
    save_df["정답 response"] = gt_response_list
    save_df["pipeline 생성 결과"] = pipeline_generated_list
    save_df.to_csv(output_path, sep='\t')

if __name__ == "__main__":
    args = parse_args()
    main(args)

