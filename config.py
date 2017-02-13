class Config:
    
    epochs = 3

    # Original image width / height
    w = 300
    h = 160

    row  = 66
    col = 200
    ch = 3
    roi = ( (60, 140), (0, 320))

    data_folder = "./data"
    model_dir = "/data/CarND-Behavioral-Cloning/"

    model = "nvidia_original"

    samples_per_epoch = 20224
    batch_size = 256

    bias = 1.0
    
conf = Config()

