# Start training on the ARITHMETIC environment:
python -m grpo_slm.train \
  --env arithmetic \
  --model_name gpt2 \
  --batch_size 8 \
  --epochs 3 \
  --lr 1e-5
