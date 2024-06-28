import cv2, numpy as np, os, logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

def pre_process_image(img):
    # Preprocess the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
    proc = cv2.bitwise_not(proc, proc)

    # Dilation
    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
    proc = cv2.dilate(proc, kernel, iterations = 2)

    # Find the biggest contour
    contours, h = cv2.findContours(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]

    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=lambda x: x[1])
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=lambda x: x[1])
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=lambda x: x[1])
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=lambda x: x[1])
    corners = [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]

    # Crop and warp the image
    pts1 = np.float32(corners)
    pts2 = np.float32([[0, 0], [28*9, 0], [28*9, 28*9], [0, 28*9]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    processed_sudoku = cv2.warpPerspective(proc, M, (28*9, 28*9))

    return processed_sudoku

def predict_sudoku(img_path, model_path):
    img = cv2.imread(img_path)
    processed_sudoku = pre_process_image(img)
    model = tf.keras.models.load_model(model_path)

    # Predict the sudoku
    mat = np.zeros((9, 9), dtype=int)
    for i in range(9):
        for j in range(9):
            cropped = processed_sudoku[28*i:28*(i+1), 28*j:28*(j+1)]
            cropped_normalized = (cropped / 255.0).reshape(1, 28, 28, 1)  # Reshape and normalize
            prediction = model.predict(cropped_normalized, verbose = 0)
            predicted_number = np.argmax(prediction)
            if cropped[7:21, 7:21].mean() >= 20:
                mat[i][j] = predicted_number
                
    return mat

