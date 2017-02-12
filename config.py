class Config:
    
    epochs = 3

    # Original image width / height
    w = 300
    h = 160

    row  = 64
    col = 64
    ch = 3
    roi = ( (60, 135), (0, 320))

    data_folder = "./data"
    model_dir = "/data/CarND-Behavioral-Cloning/"

    model = "nvidia_original"
    
conf = Config()

