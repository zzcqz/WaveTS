export CUDA_VISIBLE_DEVICES=0

model_name=WaveTSB

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange_720_96 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 720 \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --itr 1 \
    --lradj type1 \
    --dropout 0.05 \
    --batch_size 32 \
    --train_epochs 50 \
    --learning_rate 0.0005 \

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange_720_192 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 720 \
    --pred_len 192 \
    --e_layers 2 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --itr 1 \
    --lradj type3 \
    --dropout 0.05 \
    --batch_size 32 \
    --train_epochs 50 \
    --learning_rate 0.0001 \

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange_720_336 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 720 \
    --pred_len 336 \
    --e_layers 2 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --itr 1 \
    --lradj type3 \
    --dropout 0.05 \
    --batch_size 32 \
    --train_epochs 50 \
    --learning_rate 0.0001 \

python -u run.py \
    --is_training 1 \
    --root_path ./dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange_720_720 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 720 \
    --pred_len 720 \
    --e_layers 2 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --itr 1 \
    --lradj type3 \
    --dropout 0.05 \
    --batch_size 32 \
    --train_epochs 50 \
    --learning_rate 0.0001 \