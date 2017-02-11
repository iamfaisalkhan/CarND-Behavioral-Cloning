import numpy as np

def training_generator(data, angle_bias=1.0, batch_size=128):
    new_shape = (64, 64)
    batch_X = np.zeros((batch_size, new_shape[0], new_shape[1], 3))
    batch_y = np.zeros(batch_size)
    cnt = 0
    while 1:
        while (cnt < batch_size):
            # Randomly select an image. 
            index = np.random.randint(len(data))
            camera = ['center', 'left', 'right']
            ci = np.random.randint(3)

            file = os.path.join(data_folder, data[camera[ci]].iloc[index].strip())
            shift = [0, 0.25, -0.25] #shift angle for center, left, and right camera
            
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            y = data['steering'].iloc[index] + shift[ci]
            
            X, y = prepare(image, y)

            X = cv2.resize(X, (64, 64))
            X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
            
            # Higher bias value will pick smaller angles. 
            threshold = np.random.uniform()
            if abs(y) + angle_bias < threshold:
                continue

            batch_X[cnt] = X
            batch_y[cnt] = y

            cnt += 1
        angle_bias = angle_bias / 1.1
        yield (batch_X, batch_y)


def validation_generator(data, steering, batch_size=128):
    new_shape = (64, 64)
    batch_X = np.zeros((batch_size, new_shape[0], new_shape[1], 3))
    batch_y = np.zeros(batch_size)
    cnt = 0
    while 1:
        while (cnt < batch_size):
            # Randomly select an image. 
            index = np.random.randint(len(data))

            file = data[index][0]
            image = cv2.imread(file)

            y = steering[index]
            
            X = image[60:135, :, :]
            X = cv2.resize(X, new_shape, interpolation=cv2.INTER_AREA)

            batch_X[cnt] = X
            batch_y[cnt] = y

            cnt += 1
            
        yield (batch_X, batch_y)