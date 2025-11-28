cvm 기능 추가
실행방법(예시)
python -m scripts.localize_image `
  --image "node_images\node_images\401(1).jpg" `
  --csv "ai_4f_node_map_fixed_embeded.csv" `
  --model "ViT-L-14" --pretrained "laion2b_s32b_b82k" `
  --device cpu --topk 5 `
  --use_geo --node_images_dir "node_images\node_images" `
  --prev_node 427 --w_geo 0.7 --w_prior 0.3
