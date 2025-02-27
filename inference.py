from ddpm_cfg.train import train, eval

def main(model_config=None):
    modelConfig = {
        "state": "test",
        "epoch": 200,
        "batch_size": 80,
        "T": 1000,
        "dim": 128,
        "dim_scale": [1, 2, 3, 4],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "clip_grad": 1.0,
        "device": "cuda:0",
        "load_weight": None,
        "save_dir": "/root/autodl-tmp/simple-ddpm-cfg/exp/",
        "test_load_weight": "ckpt_100.pt",
        "sampled_dir": "/root/autodl-tmp/simple-ddpm-cfg/sampledimgs",
        "sampled_noise_name": "noise_image.png",
        "sampled_image_name": "sampled_image.png",
        "nrow": 8
    }
    
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)
        
if __name__=="__main__":
    main()