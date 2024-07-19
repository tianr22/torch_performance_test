timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
model_size=7
batch_size=1
seq_len=4096
log_file="logs/${timestamp}_ms_${model_size}_bs_${batch_size}_sl_${seq_len}.txt"
echo "Logging to ${log_file}"
python test_mlp.py --model_size $model_size --batch_size $batch_size --seq_len $seq_len | tee -a ${log_file}
python test_rotary.py --model_size $model_size --batch_size $batch_size --seq_len $seq_len | tee -a ${log_file}
python test_attn.py --model_size $model_size --batch_size $batch_size --seq_len $seq_len | tee -a ${log_file}
python test_rmsnorm.py --model_size $model_size --batch_size $batch_size --seq_len $seq_len | tee -a ${log_file}