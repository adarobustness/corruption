#!/bin/bash
python corruption.py --task gqa --split testdev --corruption_category change_char --severity_begin 1 --severity_end 5
python corruption.py --task vqa --split karpathy_test --corruption_category change_char --severity_begin 1 --severity_end 5
python corruption.py --task nlvr --split test --corruption_category change_char --severity_begin 1 --severity_end 5

python corruption.py --task gqa --split testdev --corruption_category none --corruption_method nonsense
python corruption.py --task vqa --split karpathy_test --corruption_category none --corruption_method nonsense
python corruption.py --task nlvr --split test --corruption_category none --corruption_method nonsense


python corruption.py --task gqa --split testdev --corruption_category none --corruption_method to_passive
python corruption.py --task gqa --split testdev --corruption_category none --corruption_method to_active 
python corruption.py --task gqa --split testdev --corruption_category none --corruption_method to_casual 
python corruption.py --task gqa --split testdev --corruption_category none --corruption_method to_formal
python corruption.py --task gqa --split testdev --corruption_category none --corruption_method tense
python corruption.py --task gqa --split testdev --corruption_category none --corruption_method double_denial 

python corruption.py --task vqa --split karpathy_test --corruption_category none --corruption_method to_passive
python corruption.py --task vqa --split karpathy_test --corruption_category none --corruption_method to_active 
python corruption.py --task vqa --split karpathy_test --corruption_category none --corruption_method to_casual 
python corruption.py --task vqa --split karpathy_test --corruption_category none --corruption_method to_formal
python corruption.py --task vqa --split karpathy_test --corruption_category none --corruption_method tense
python corruption.py --task vqa --split karpathy_test --corruption_category none --corruption_method double_denial 
 
python corruption.py --task nlvr --split test --corruption_category none --corruption_method to_passive
python corruption.py --task nlvr --split test --corruption_category none --corruption_method to_active 
python corruption.py --task nlvr --split test --corruption_category none --corruption_method to_casual 
python corruption.py --task nlvr --split test --corruption_category none --corruption_method to_formal
python corruption.py --task nlvr --split test --corruption_category none --corruption_method tense
python corruption.py --task nlvr --split test --corruption_category none --corruption_method double_denial

python corruption.py --task gqa --split testdev --corruption_category add_text
python corruption.py --task vqa --split karpathy_test --corruption_category add_text
python corruption.py --task nlvr --split test --corruption_category add_text

python corruption.py --task gqa --split testdev --corruption_category drop_text_on_pos
python corruption.py --task vqa --split karpathy_test --corruption_category drop_text_on_pos
python corruption.py --task nlvr --split test --corruption_category drop_text_on_pos

python corruption.py --task gqa --split testdev --corruption_category swap_text 
python corruption.py --task gqa --split testdev --corruption_category none --corruption_method back_trans 
python corruption.py --task gqa --split testdev --corruption_category none --corruption_method random_word_swap 
python corruption.py --task vqa --split karpathy_test --corruption_category none --corruption_method back_trans 
python corruption.py --task vqa --split karpathy_test --corruption_category none --corruption_method random_word_swap
python corruption.py --task vqa --split karpathy_test --corruption_category swap_text 
python corruption.py --task nlvr --split test --corruption_category none --corruption_method back_trans 
python corruption.py --task nlvr --split test --corruption_category none --corruption_method random_word_swap 
python corruption.py --task nlvr --split test --corruption_category swap_text


python corruption.py --task gqa --split testdev --corruption_category positional 
python corruption.py --task vqa --split karpathy_test --corruption_category positional 
python corruption.py --task nlvr --split test --corruption_category positional 

python corruption.py --task gqa --split testdev --corruption_category change_char --severity_begin 1 --severity_end 4
python corruption.py --task vqa --split karpathy_test --corruption_category change_char --severity_begin 1 --severity_end 4
python corruption.py --task nlvr --split test --corruption_category change_char --severity_begin 1 --severity_end 4

python corruption.py --task gqa --split testdev --corruption_category none --corruption_method random_word_insert --severity_begin 1 --severity_end 4
python corruption.py --task vqa --split karpathy_test --corruption_category none --corruption_method random_word_insert --severity_begin 1 --severity_end 4
python corruption.py --task nlvr --split test --corruption_category none --corruption_method random_word_insert --severity_begin 1 --severity_end 4
python corruption.py --task gqa --split testdev --corruption_category none --corruption_method random_word_delete --severity_begin 1 --severity_end 4
python corruption.py --task vqa --split karpathy_test --corruption_category none --corruption_method random_word_delete --severity_begin 1 --severity_end 4
python corruption.py --task nlvr --split test --corruption_category none --corruption_method random_word_delete --severity_begin 1 --severity_end 4
python corruption.py --task gqa --split testdev --corruption_category none --corruption_method random_word_swap --severity_begin 1 --severity_end 4
python corruption.py --task vqa --split karpathy_test --corruption_category none --corruption_method random_word_swap --severity_begin 1 --severity_end 4
python corruption.py --task nlvr --split test --corruption_category none --corruption_method random_word_swap --severity_begin 1 --severity_end 4

python corruption.py --task vqa --split karpathy_test --corruption_category none --corruption_method swap_syn_word_emb --severity_begin 1 --severity_end 4
python corruption.py --task nlvr --split test --corruption_category none --corruption_method swap_syn_word_emb --severity_begin 1 --severity_end 4

python corruption.py --task gqa --split testdev --corruption_category none --corruption_method swap_syn_word_net --severity_begin 1 --severity_end 4
python corruption.py --task vqa --split karpathy_test --corruption_category none --corruption_method swap_syn_word_net --severity_begin 1 --severity_end 4
python corruption.py --task nlvr --split test --corruption_category none --corruption_method swap_syn_word_net --severity_begin 1 --severity_end 4

