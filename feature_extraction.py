# import os
# import pickle
#
# base_path = 'data/Bollywood_celeb_face_localized'
# categories = os.listdir(base_path)  # e.g. bollywood_celeb_faces2, faces_0, etc.
#
# filenames = []
# for category in categories:
#     category_path = os.path.join(base_path, category)
#     actors = os.listdir(category_path)
#     for actor in actors:
#         actor_path = os.path.join(category_path, actor)
#         for file in os.listdir(actor_path):
#             filenames.append(os.path.join(actor_path, file))
#
# pickle.dump(filenames, open('filenames.pkl', 'wb'))
# print(len(filenames))




from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm

filenames = pickle.load(open('filenames.pkl','rb'))

model = VGGFace(model ='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

def feature_extractor(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()

    return result

features  = []
for file in tqdm(filenames):
    features.append(feature_extractor(file,model))

pickle.dump(features,open('embedding.pkl','wb'))


