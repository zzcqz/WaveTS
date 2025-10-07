export CUDA_VISIBLE_DEVICES=0

model_name=WaveTSB

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --model_id ETTh1_720_96 \
#     --model $model_name \
#     --data ETTh1 \
#     --features M \
#     --seq_len 720 \
#     --pred_len 96 \
#     --e_layers 2 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --itr 1 \
#     --lradj type3 \
#     --dropout 0.05 \
#     --batch_size 16 \
#     --train_epochs 50 \
#     --learning_rate 0.00005 \
#     --hidden 64 \

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --model_id ETTh1_720_192 \
#     --model $model_name \
#     --data ETTh1 \
#     --features M \
#     --seq_len 720 \
#     --pred_len 192 \
#     --e_layers 2 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --itr 1 \
#     --lradj type3 \
#     --dropout 0.05 \
#     --batch_size 8 \
#     --train_epochs 50 \
#     --learning_rate 0.00009 \
#     --hidden 7 \
    
# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --model_id ETTh1_720_336 \
#     --model $model_name \
#     --data ETTh1 \
#     --features M \
#     --seq_len 720 \
#     --pred_len 336 \
#     --e_layers 2 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --itr 1 \
#     --lradj type3 \
#     --dropout 0.05 \
#     --batch_size 16 \
#     --train_epochs 50 \
#     --learning_rate 0.00005 \
#     --hidden 5 \

# python -u run.py \
#     --is_training 1 \
#     --root_path ./dataset/ETT-small/ \
#     --data_path ETTh1.csv \
#     --model_id ETTh1_720_720 \
#     --model $model_name \
#     --data ETTh1 \
#     --features M \
#     --seq_len 720 \
#     --pred_len 720 \
#     --e_layers 2 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --itr 1 \
#     --lradj type3 \
#     --dropout 0.05 \
#     --batch_size 64 \
#     --train_epochs 50 \
#     --learning_rate 0.00005 \
#     --hidden 2 \