
def model_load():
    load_model('model.h5', compile=False)
    
def compute_distance(encoded_image, known_encoding):
    return np.sqrt(np.sum(np.square(abs(encoded_image - known_encoding))))
    #return np.max(abs(encoded_image - known_encoding),axis=1)
    #return cosine_similarity(encoded_image, known_encoding)
    
def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y
        
def encode_image(image):
    img = prewhiten(np.reshape(cv2.resize(image, (160,160)), (-1,160,160,3)))
    return prediction(img)

def compute_distance():
    