cvm 기능 추가
실행방법(예시)
python -m scripts.localize_image `
  --image "node_images\node_images\401(1).jpg" `
  --csv "ai_4f_node_map_fixed_embeded.csv" `
  --model "ViT-L-14" --pretrained "laion2b_s32b_b82k" `
  --device cpu --topk 5 `
  --use_geo --node_images_dir "node_images\node_images" `
  --prev_node 427 --w_geo 0.7 --w_prior 0.3

## Run CVM + value map (example)

레포 루트(`caainp-cvm`)에서 아래 명령으로
현재 이미지 기준 위치 추정 + value map을 확인할 수 있습니다.

`python -m scripts.run_cvm_step`

## 벤치마크 측정 코드 예시
--out_dir 다음으로 오는 저장 파일명 뒤에 넘버링 체크 부탁드립니다.
### room benchmark
python scripts/eval_localization.py --test_dir "benchmarks/room_test_60deg" --csv "ai_4f_node_map_fixed_embeded.csv" --out_dir "benchmark_results/room_patch1" --device cuda --use_ocr --use_geo --node_images_dir "node_images/node_images" --w_clip 1.0 --w_ocr 0.8 --w_geo 0.4 --w_prior 0.2 --ocr_langs "ko,en" --ocr_use_roi --ocr_grayscale --ocr_upscale 2.0 --ocr_contrast --ocr_sharpen --ocr_adaptive
### sequence benchmark
python scripts/eval_sequence.py --test_dir "benchmarks/route_test" --set "4150_414" --csv "ai_4f_node_map_fixed_embeded.csv" --out_dir "benchmark_results/sequence_patch1" --device cuda --ocr_use_roi --ocr_grayscale --ocr_upscale 2.0 --ocr_contrast --ocr_sharpen --ocr_adaptive
