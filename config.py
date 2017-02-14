class Config:
    
    epochs = 3

    # Original image width / height
    w = 320
    h = 160

    row  = 66
    col = 200
    ch = 3
    roi = ( (60, 140), (0, 320))

    data_folder = "./data"
    model_dir = "./"

    model = "nvidia_relu_dropout"

    samples_per_epoch = 20224
    batch_size = 512

    bias = 1.0

    # Camera angle offset
    left_offset = 0.2
    right_offset = -0.2
    
conf = Config()

