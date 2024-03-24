for model_name in PatchTST
do
for pred_len in 96 192 336 720
do
seq_len=720
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id electricity_720_$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --H_order 6 \
  --lradj type3 \
  --dropout 0.05 \
  --batch_size 8 \
  --train_epochs 50 \
  --learning_rate 0.0005 \

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_720_$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --H_order 6 \
  --lradj type3 \
  --dropout 0.05 \
  --batch_size 8 \
  --train_epochs 50 \
  --learning_rate 0.0005 \


done
done