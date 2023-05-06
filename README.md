# Colmap Neus

This repository provides a pipeline for fitting [NeuS](https://github.com/Totoro97/NeuS) surface reconstruction from a Video input


Preprocess the captured video with [COLMAP](https://colmap.github.io/) 
    
```sh
    cd preprocess_data
    python preprocess_data.py --data $xxx.mp4 --output_dir $preprocess_dir
```

Rescale the scene

```sh
    python rescale_and_mask.py --work_dir $preprocess_dir
```
To generate object mask for the scene, refer to [$U^2$-Net](https://github.com/xuebinqin/U-2-Net)


Run the resonctruciton
```sh
    CUDA_VISIBLE_DEVICES=1 python scripts/train_colmap.py \
        --data_dir $preprocess_dir \
        --config configs/default_colmap.yaml \
        --train_dir ckpt/$xxx 
        --print_every 10 \
        --export_mesh resultvis/$xxx/sdf_mc.ply 
        --ek_lambda 0.05 
        --n_epochs 3 
        --n_iters 5000
```


### Reference
* [NeuS](https://github.com/Totoro97/NeuS) 
