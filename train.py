# -*- coding: utf-8 -*-
"""240111_멀티턴_gpt2_훈련 코드"""

"""## tokenizer"""
from transformers import PreTrainedTokenizerFast, AutoTokenizer, GPT2LMHeadModel, LlamaForCausalLM, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
# from process import preprocess_function, preprocess_function_generate, CustomDataset, CustomDataset_1turn
from process_2 import MultiturnDataset, pad_collate_right, pad_collate_left
import random
# from transformers import DataCollatorForLanguageModeling
# from transformers import TrainingArguments, Trainer
# from transformers import EarlyStoppingCallback
import torch
import os
import wandb
import math
# import torch.nn.functional as F
import numpy as np
from scipy.special import softmax
from sklearn.metrics import log_loss
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

# from eval_utils import cal_BLEU_1
from nltk.translate.bleu_score import sentence_bleu
from args import parse_args

import time

random.seed(42)
args = parse_args()
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


"""## load_model"""
if args.tokenizer_path == "skt/kogpt2-base-v2":
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token = '</s>', eos_token = '</s>', unk_token='<unk>',
                                                    pad_token = '<pad>', mask_token = '<mask>')
else:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path) # MIT License

# if args.model_path == "heegyu/kogpt-j-base": 
#     tokenizer.pad_token = tokenizer.eos_token
#     special_tokens_dict = {'additional_special_tokens': [f'<unused{i}>' for i in range(1, 10)]}
#     tokenizer.add_special_tokens(special_tokens_dict)
#     # 모델 사이즈도 키워야?
#     model.resize_token_embeddings(len(tokenizer))


'''변경 전 pad token
>>> print(tokenizer.pad_token)
<pad>
>>> print(tokenizer.pad_token_id)
0
>>> print(tokenizer.eos_token)
</s>
>>> print(tokenizer.eos_token_id)
2

>>> print(tokenizer.special_tokens_map)
{'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '</s>'}

'''
'''변경 후 pad token
>>> tokenizer.pad_token = tokenizer.eos_token
>>> print(tokenizer.pad_token)
</s>
>>> print(tokenizer.pad_token_id)
2

>>> print(tokenizer.special_tokens_map)
{'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '</s>', 'additional_special_tokens': ['<unused1>', '<unused2>', '<unused3>', '<unused4>', '<unused5>', '<unused6>', '<unused7>', '<unused8>', '<unused9>']}
'''



# args.ignore_index_INDEX = -100
# vocab_size_with_args.ignore_index = tokenizer.vocab_size + 1  # args.ignore_index_index를 추가했을 때의 vocab size
    
if args.model == "GPT2LMHeadModel":
    model = GPT2LMHeadModel.from_pretrained(args.model_path)#, vocab_size=vocab_size_with_args.ignore_index)
elif args.model == "AutoModelForCausalLM":
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    

print("pre-trained model 구조")
print(model, end='\n\n')


### 0202 추가 ###
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id # 기존에는 model.config.pad_token_id이 "AutoModelForCausalLM"와 "GPT2LMHeadModel" 모두 None 이었음
    
    
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = args.device
model.to(device)
n_gpu = 1
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=[0,1])
    n_gpu = torch.cuda.device_count()


# Load training dataset and split into train/valid
training_dataset = load_dataset("csv", data_files="(new)Training_history_and_gtresponse.tsv", split="train", sep="\t") # train: 총 116만
validation_dataset = load_dataset("csv", data_files="0206_(new)Validation_history_and_gtresponse.tsv", split="train", sep="\t") # valid: 총 13만
# valid: 10%만 (1.3만)
# valid: 1%만 (1.3천)
num_samples = int(len(validation_dataset) * 0.01)
random_subset_indices = random.sample(range(len(validation_dataset)), num_samples)
validation_dataset = validation_dataset.select(random_subset_indices)

###test_dataset = load_dataset("csv", data_files="0207_test_history_and_gtresponse.tsv", split="train[:100%]", sep="\t") # test: 총 9779개

training_dataset = training_dataset.remove_columns(["labels"])
validation_dataset = validation_dataset.remove_columns(["labels"]) 

# Create DatasetDict
if args.subset_percentage == 1:
    # total_dataset_dict = {"train": training_dataset, "valid": validation_dataset, "test": test_dataset}
    sub_training_dataset = training_dataset
    sub_validation_dataset = validation_dataset
else:
    # 데이터셋의 일부만 랜덤으로 선택 (예: 10%)
    num_samples = int(len(training_dataset) * args.subset_percentage)
    random_subset_indices = random.sample(range(len(training_dataset)), num_samples)
    sub_training_dataset = training_dataset.select(random_subset_indices)

    num_samples = int(len(validation_dataset) * args.subset_percentage)
    random_subset_indices = random.sample(range(len(validation_dataset)), num_samples)
    sub_validation_dataset = validation_dataset.select(random_subset_indices)


sub_training_dataset = MultiturnDataset(sub_training_dataset, tokenizer, args.max_seq_len, ignore_index=args.ignore_index, ignore_loss_fn=args.ignore_loss_fn)
sub_validation_dataset = MultiturnDataset(sub_validation_dataset, tokenizer, args.max_seq_len, ignore_index=args.ignore_index, ignore_loss_fn=args.ignore_loss_fn)
if args.pad_side == "right":
    train_dataloader = DataLoader(sub_training_dataset, batch_size=args.batch_size, collate_fn=pad_collate_right)
    valid_dataloader = DataLoader(sub_validation_dataset, batch_size=args.batch_size, collate_fn=pad_collate_right)
else:
    train_dataloader = DataLoader(sub_training_dataset, batch_size=args.batch_size, collate_fn=pad_collate_left)
    valid_dataloader = DataLoader(sub_validation_dataset, batch_size=args.batch_size, collate_fn=pad_collate_left)



# Create DatasetDict
tokenized_total_dataset_dict = {"train": sub_training_dataset, "valid": sub_validation_dataset}


print(" % * " * 50)
print(tokenized_total_dataset_dict)



print(tokenized_total_dataset_dict["train"][0])
print(tokenized_total_dataset_dict["valid"][0])
###print(tokenized_total_dataset_dict["test"][0])

print("Train Dataset Size:", len(tokenized_total_dataset_dict["train"]))
print("Validation Dataset Size:", len(tokenized_total_dataset_dict["valid"]))




# """## train model"""
# # 모델 및 옵티마이저 초기화
optimizer = AdamW(model.parameters(), lr=args.learning_rate)

if args.scheduler_setting:
    # Noam Learning Rate Scheduler 설정
    num_training_steps = (len(training_dataset) // args.batch_size) * args.num_epochs  # 1 epoch 기준 7만 2천
    num_warmup_steps = (len(training_dataset) // args.batch_size) * 0.05  # warm-up 스텝 수 (16000)

    if args.scheduler_setting=="warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps # 7만 2천 / (16 b.s * 3 epoch) = 
        )
    elif args.scheduler_setting=="steplr":
        scheduler = torch.optim.lr_scheduler.StepLR( # step_size 마다 gamma 비율로 lr 감소
            optimizer,
            step_size=num_training_steps, gamma=0.01)


# 에폭 수
patience_setting = args.patience_setting
best_valid_loss = float('inf')
current_patience = 0
best_bleu_score = 0.0


if args.wandb_setting:
    # WandB 초기화
    ###wandb.init(project="multi-turn-baseline-gpt2", name="0112_baseline_experient_heegyu_kogpt_데이터셋수정후", config={"args.batch_size": args.batch_size, "num_train_epochs": args.num_epochs, "train_data_cnt": len(lm_train_dataset)})
    # wandb.init(project="multi-turn-baseline-gpt2_v2", name="0123_baseline_experient_heegyu_kogpt_warmup_v2_origintokenizer_500epoch_trainloss수정_GPTHeadModel_100%_no_scheduler", config={"args.batch_size": args.batch_size, "num_train_epochs": args.num_epochs, "train_data_cnt": len(training_dataset)})
    # wandb.init(project="multi-turn-baseline-gpt2_v2", name=f"{args.output_dir}", config={"args.batch_size": args.batch_size, "num_train_epochs": args.num_epochs, "train_data_cnt": len(training_dataset), "subset_percentage": '100%'})
    wandb.init(project="multi-turn-baseline-gpt2_v2", 
               name=f"{args.output_dir}",
               config = args 
            #    config={"args.batch_size": args.batch_size, "num_train_epochs": args.num_epochs, "train_data_cnt": len(training_dataset), "subset_percentage": subset_percentage, 
            #             }
                    #    "StepLR": "TRUE", "lr": 3e-3, "gamma": 0.1}, 
               )
    # wandb.config.update({"log_freq": "epoch"})
    wandb.config.update({"log_freq": 100}) # 100번째 미니배치마다




criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_index) # Pytorch CrossEntropyLoss에 args.ignore_index argument 줌


# 훈련 루프
print("Training 시작")
torch.cuda.empty_cache()

# global_step = 0
# step = 0
# 이어서 학습 시
global_step = 8625
step = 17283

epoch = 0
best_train_loss = float('inf')
best_train_loss_by_accumulation_steps = float('inf')
this_train_loss_by_accumulation_steps = []
t_loss_desc_flag = False

if args.continue_from:
    global_step = args.continue_from
    step = global_step*args.gradient_accumulation_steps - 1

# while True:
for epoch in range(1, args.num_epochs+1):
    print(f"# # # # # ", epoch)
    train_start_time_epoch = time.time()
    print(train_start_time_epoch)
    
    model.train()
    total_loss = 0.0
    
    # if epoch == 1: batch_i = (17283-1)
    # else: batch_i = -1
    # train_dataloader_iter = iter(train_dataloader)
    
    # 훈련 데이터셋 반복
    ###for batch in train_dataloader:        
    for (batch_i, batch) in enumerate(train_dataloader):
        if epoch == 1 and batch_i < 17283:
            continue
        
        # activate new training mode
        if batch is None:
            print("batch가 None 입니다.")
            continue

        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device) # (args.batch_size, sequence_length)
        attention_mask = batch["attention_mask"].to(device)
        gt_response_str = batch["gt_response_str"]
    
        ### Forward pass
        if args.ignore_loss_fn:
            outputs = model(inputs, attention_mask=attention_mask) # 이전 코드: outputs = model(inputs, labels=labels, attention_mask=attention_mask)
        else:
            outputs = model(inputs, labels=labels, attention_mask=attention_mask)
        logits = outputs.logits # (args.batch_size, sequence_length, vocab_size)
        ###logits_with_softmax = torch.nn.functional.log_softmax(logits, dim=-1) # (args.batch_size, sequence_length, vocab_size)
        

        # CrossEntropyLoss 계산
        # input: (args.batch_size * sequence_length, vocab_size)
        # target: # (args.batch_size, sequence_length)
        if args.ignore_loss_fn:
            ###loss = criterion(logits_with_softmax.view(-1, tokenizer.vocab_size), labels.view(-1))
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            print("loss: ", loss)
        else:
            loss = outputs.loss

        if n_gpu > 1:
            loss = torch.mean(loss)
            
        # 0306
        this_train_loss_by_accumulation_steps.append(loss.item())
        if loss < best_train_loss: # best_train_loss: best train loss by step
            best_train_loss = loss
            #####t_loss_desc_flag = True
            
        
        ### Backward pass
        # gradient update
        loss.backward() # Gradient Accumulation
        step += 1 # default 2 mini-batch 마다 저장
        if step % args.gradient_accumulation_steps == 0: # backward steps (gradient_accumulation_steps 만큼 수행)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        
            if args.scheduler_setting:
                scheduler.step()
                
            if global_step >= args.num_optim_steps:
                break

            print("Loss at step {}: {}".format(global_step, loss.item()))
            if args.wandb_setting:
                wandb.log({"global_step": global_step, "train_loss_step": loss.item()})

            total_loss += loss.item()
            
            # 0308
            mean_train_loss = np.mean(this_train_loss_by_accumulation_steps)
            if mean_train_loss < best_train_loss_by_accumulation_steps:
                best_train_loss_by_accumulation_steps = mean_train_loss
                t_loss_desc_flag = True
            else:
                t_loss_desc_flag = False
            this_train_loss_by_accumulation_steps = []
            '''
            # print("* " * 50)
            # print("1st input_ids: ", inputs[0])
            # print("1st labels: ", labels)
            '''

            # 검증 데이터셋 평가
            model.eval()
            total_eval_loss = 0.0
            
            bleu_scores_epoch = []
            perplexity_scores_epoch = 0.0
            

            with torch.no_grad():
                ###excluded_indices = set(range(step - args.gradient_accumulation_steps, step))
                ###sampled_indices = random.sample(set(range(len(valid_dataloader))) - excluded_indices, args.gradient_accumulation_steps)

                for (batch_idx, batch) in enumerate(valid_dataloader):
                    ###if batch_idx not in sampled_indices:
                        ###continue
                    
                    # KeyError가 발생했을 때, 다음 batch로 넘어가도록 처리
                    if batch is None:
                        continue


                    inputs = batch["input_ids"].to(device) # history + response
                    labels = batch["labels"].to(device) # (args.batch_size, sequence_length)
                    attention_mask = batch["attention_mask"].to(device)
                    gt_response_str = batch["gt_response_str"]
                

                    if args.ignore_loss_fn:
                        outputs = model(inputs, attention_mask=attention_mask)
                    else:
                        outputs = model(inputs, labels=labels, attention_mask=attention_mask) #output_hidden_states=True)
                    logits = outputs.logits # (args.batch_size, sequence_length, vocab_size)
                    ###logits_with_softmax = torch.nn.functional.log_softmax(logits, dim=-1) # (args.batch_size, sequence_length, vocab_size)
                    

                    # CrossEntropyLoss 계산
                    # input: (args.batch_size * sequence_length, vocab_size)
                    # target: # (args.batch_size, sequence_length)
                    if args.ignore_loss_fn: 
                        # loss = criterion(logits_with_softmax.view(-1, tokenizer.vocab_size), labels.view(-1))
                        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    else: 
                        loss = outputs.loss

                    if n_gpu > 1:
                        loss = torch.mean(loss)  # 병렬이라서 torch.Size([2]) -> 스칼라 값으로 변환해줌
                    total_eval_loss += loss.item()
                    

            
                    perplexity = torch.exp(loss)
                    perplexity_scores_epoch += perplexity.item()       
                    
                    
                # 중간 생성 결과 체크 (마지막 배치만)
                '''
                # warning message
                #   warnings.warn(
                # We strongly recommend passing in an `attention_mask` since your input_ids may be padded. See https://huggingface.co/docs/transformers/troubleshooting#incorrect-output-when-padding-tokens-arent-masked.
                model_generate_outputs = model.module.greedy_search(inputs.to(device), max_length=args.max_seq_len, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                model_generate_outputs_beam = model.module.generate(inputs.to(device), max_new_tokens=30, num_beams=3, no_repeat_ngram_size=2)
                model_generate_outputs_decoded = tokenizer.batch_decode(model_generate_outputs, skip_special_tokens=True)
                model_generate_outputs_decoded_beam = tokenizer.batch_decode(model_generate_outputs_beam, skip_special_tokens=True)
                
                gt_response_str_i = gt_response_str[0]
                model_generate_outputs_decoded_i = model_generate_outputs_decoded[0]
                model_generate_outputs_decoded_i = model_generate_outputs_decoded_i.split("\n1 : ")[-1].strip()
                prompt = model_generate_outputs_decoded[0][:-(len(model_generate_outputs_decoded_i)+1)]
                
                model_generate_outputs_decoded_beam_i = model_generate_outputs_decoded_beam[0]
                model_generate_outputs_decoded_beam_i = model_generate_outputs_decoded_beam_i.split("\n1 : ")[-1].strip()

                # print("prompt: ", prompt)
                # print("% % % % % % % % % % % % % % % % % % 정답 데이터: ", gt_response_str_i)
                # print("# # # 모델이 생성한 응답 output (generate) GREEDY: ", model_generate_outputs_decoded_i)
                # print("# # # 모델이 생성한 응답 output (generate) BEAM: ", model_generate_outputs_decoded_beam_i)  
                
                # BLEU calculate
                reference = [list(gt_response_str_i)]
                candidate = list(model_generate_outputs_decoded_i)
                bleu_scores_greedy = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
                
                reference = [list(gt_response_str_i)]
                candidate = list(model_generate_outputs_decoded_beam_i)
                bleu_scores_beam = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))

                # print("= = = BLEU score GREEDY: ", bleu_scores_greedy * 100)
                # print("= = = BLEU score BEAM: ", bleu_scores_beam * 100, end='\n\n')
                bleu_scores_epoch.append(bleu_scores_beam)
                        
                
                # 마지막 배치 결과만 출력
                print("prompt: ", prompt)
                print("% % % % % % % % % % % % % % % % % % 정답 데이터: ", gt_response_str_i)
                print("# # # 모델이 생성한 응답 output (generate) GREEDY: ", model_generate_outputs_decoded_i)
                print("# # # 모델이 생성한 응답 output (generate) BEAM: ", model_generate_outputs_decoded_beam_i)  
                
                print("= = = BLEU score GREEDY: ", bleu_scores_greedy * 100)
                print("= = = BLEU score BEAM: ", bleu_scores_beam * 100, end='\n\n')
                '''
            
            model.train()
    
                    

            avg_eval_loss = total_eval_loss / (len(valid_dataloader))
            ###avg_eval_loss = total_eval_loss / (args.gradient_accumulation_steps)
            # avg_bleu_score = sum(bleu_scores_epoch) / len(bleu_scores_epoch) # BLEU 기준으로 훈련
            # print(f"Epoch {epoch + 1}, Validation Loss: {avg_eval_loss}")
            print(f"Step {global_step}, Validation Loss: {avg_eval_loss}")
            if args.wandb_setting:
                wandb.log({"validation_loss_step": avg_eval_loss})

            # 이전 에폭보다 검증 손실이 낮아졌을 때만 모델 저장
            # if (global_step > 58) and t_loss_desc_flag:
            print("t_loss_desc_flag: ", t_loss_desc_flag)
            print("mean this accumul steps loss: ", mean_train_loss)
            if t_loss_desc_flag: ### 추가 (train_loss가 감소했을 때만 저장)
                t_loss_desc_flag = False
                if avg_eval_loss < best_valid_loss:
                    best_valid_loss = avg_eval_loss
                    current_patience = 0
                    model_save_path = f"/data/hyunkyung_lee/multi-turn/baseline/{args.output_dir}_{len(train_dataloader)}_{args.gradient_accumulation_steps}/{global_step}/"
                    model.module.save_pretrained(model_save_path)

                    # patience 값을 wandb에 로깅
                    if args.wandb_setting:
                        wandb.log({"patience": current_patience})
                        wandb.save(model_save_path)

                    print(f"Model saved at step {global_step}")
                else:
                    current_patience += 1
                    model_save_path = f"/data/hyunkyung_lee/multi-turn/baseline/{args.output_dir}_{len(train_dataloader)}_{args.gradient_accumulation_steps}/{global_step}/"
                    model.module.save_pretrained(model_save_path)

                    if args.wandb_setting:
                        # patience 값을 wandb에 로깅
                        wandb.log({"patience": current_patience})

                    print(f"Patience: {current_patience}/{patience_setting}")

                    # patience가 초과되면 훈련 중단
                    if current_patience >= patience_setting:
                        print("Training early stopped due to patience.")
                        break

            # 설정한 save steps 수마다 모델 저장
            if (step % args.save_by_steps) == 0: ### 추가 (train_loss가 감소했을 때만 저장)
                model_save_path = f"/data/hyunkyung_lee/multi-turn/baseline/{args.output_dir}_{len(train_dataloader)}_{args.gradient_accumulation_steps}/{global_step}/"
                model.module.save_pretrained(model_save_path)    
                
    if global_step >= args.num_optim_steps:
        break
    
    avg_train_loss = total_loss / (len(train_dataloader))
    print("custom calculated loss value: ", loss) ###
    print(f"Epoch {epoch}, Training Loss: {avg_train_loss}")
    if args.wandb_setting:
        wandb.log({"training_loss": avg_train_loss})
    
    print('mean_ppl_score: ', perplexity_scores_epoch / len(valid_dataloader), end='\n')
    print('-> len(valid_dataloader): ', len(valid_dataloader))
    if args.wandb_setting:
        wandb.log({'mean_ppl_score': perplexity_scores_epoch / len(valid_dataloader)})
    
    if len(bleu_scores_epoch) > 0:
        mean_bleu_score = sum(bleu_scores_epoch) / len(bleu_scores_epoch)
        print('mean_bleu_score: ', mean_bleu_score, end='\n')
        print('-> len(bleu_scores_epoch): ', len(bleu_scores_epoch))
        if args.wandb_setting:
            wandb.log({'mean_bleu_score': mean_bleu_score})
    else:
        print('No BLEU scores calculated.')
    
    
    # epoch 끝날 때마다 모델 저장
    model_save_path_e = f"/data/hyunkyung_lee/multi-turn/baseline/{args.output_dir}_{len(train_dataloader)}_{args.gradient_accumulation_steps}/{epoch}epoch/"
    model.module.save_pretrained(model_save_path_e) 
    #####epoch += 1
 
# (train_dataloader * 3 epoch 종료 시)   
model_save_path_e = f"/data/hyunkyung_lee/multi-turn/baseline/{args.output_dir}_{len(train_dataloader)}_{args.gradient_accumulation_steps}/{epoch}epoch/"
model.module.save_pretrained(model_save_path_e)
