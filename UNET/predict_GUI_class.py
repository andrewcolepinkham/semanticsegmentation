import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from smooth_tiled_predictions import predict_img_with_smooth_windowing
import sys
matplotlib.use('TkAgg') 

class Predictor():
    def __init__(self):
        print("initing")
        self.model = load_model("models/model100ep.hdf5", compile=False)
        self.model.summary()
        quit()
        self.scaler = MinMaxScaler()
        self.patch_size = 256
        self.n_classes = 6

    def predict(self, img):
        img = cv2.imread(img, 1)
        input_img = self.scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
        predictions_smooth = predict_img_with_smooth_windowing(
            input_img,
            window_size=self.patch_size,
            subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
            nb_classes=self.n_classes,
            pred_func=(
                lambda img_batch_subdiv: self.model.predict((img_batch_subdiv))
            )
        )
        final_prediction = np.argmax(predictions_smooth, axis=2)
        prediction_with_smooth_blending=self.label_to_rgb(final_prediction)
        plt.figure(figsize=(16, 8))
        plt.subplot(121)
        plt.title('Testing Image')
        plt.imshow(img)
        plt.subplot(122)
        plt.title('Prediction with smooth blending')
        plt.imshow(prediction_with_smooth_blending)
        plt.show()

    def label_to_rgb(self, predicted_image):
        Building = '#3C1098'.lstrip('#')
        Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4))) # 60, 16, 152
        Land = '#8429F6'.lstrip('#')
        Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246
        Road = '#6EC1E4'.lstrip('#') 
        Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4))) #110, 193, 228
        Vegetation =  'FEDD3A'.lstrip('#') 
        Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4))) #254, 221, 58
        Water = 'E2A929'.lstrip('#') 
        Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4))) #226, 169, 41
        Unlabeled = '#9B9B9B'.lstrip('#') 
        Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4))) #155, 155, 155
        segmented_img = np.empty((predicted_image.shape[0], predicted_image.shape[1], 3))    
        segmented_img[(predicted_image == 0)] = Building
        segmented_img[(predicted_image == 1)] = Land
        segmented_img[(predicted_image == 2)] = Road
        segmented_img[(predicted_image == 3)] = Vegetation
        segmented_img[(predicted_image == 4)] = Water
        segmented_img[(predicted_image == 5)] = Unlabeled
        segmented_img = segmented_img.astype(np.uint8)
        return(segmented_img)

if __name__ == '__main__':
    predictor = Predictor()
    # predictor.predict("test_data/ktr4-18.jpg")

