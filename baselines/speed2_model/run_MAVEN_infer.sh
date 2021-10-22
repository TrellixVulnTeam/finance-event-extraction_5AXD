python3 run_maven.py \
    --data_dir ../../maven/ \
    --model_type bertcrf \
    --model_name_or_path bert-base-uncased \
    --output_dir ./output/MAVEN-$1/checkpoint-$2 \
    --max_seq_length 128 \
    --do_lower_case \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 5 \
    --save_steps 100 \
    --logging_steps 100 \
    --seed 0 \
    --do_infer
bash ../../ar.scripts/make_zip.sh ./output/MAVEN-$1/checkpoint-$2/results.jsonl $1-$2
