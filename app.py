import streamlit as st
import os
from PIL import Image
from mtcnn import MTCNN
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2

detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
feature_list = pickle.load(open('embedding.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))


def save_uploaded_image(uploaded_image):
    try:
        os.makedirs('uploads', exist_ok=True)  # auto-create folder if missing
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except Exception as e:
        st.error(f"Save error: {e}")
        return False


def extract_feature(img_path, model, detector):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img)

    if len(results) == 0:
        return None  # no face found

    x, y, width, height = results[0]['box']
    x, y = max(x, 0), max(y, 0)  # avoid negative crop values

    face = img[y:y + height, x:x + width]

    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)
    face_array = face_array.astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()
    return result


def recommend(feature_list, feature):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(feature.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos


def get_actor_name(filepath):
    # Grabs the parent folder name (actor name) regardless of OS path separator ('\' or '/')
    folder_name = os.path.basename(os.path.dirname(filepath))
    return folder_name.replace('_', ' ')


st.title('Which Bollywood Celebrity are you?')

uploaded_image = st.file_uploader('Choose an Image')

if uploaded_image is not None:
    if save_uploaded_image(uploaded_image):
        display_image = Image.open(uploaded_image)

        with st.spinner('Detecting face and finding your celebrity match... please wait'):
            feature = extract_feature(os.path.join('uploads', uploaded_image.name), model, detector)

        if feature is None:
            st.error("No face detected in the image. Please upload a clearer photo with a visible face.")
        else:
            index_pos = recommend(feature_list, feature)
            predicted_actor = get_actor_name(filenames[index_pos])

            col1, col2 = st.columns(2)

            with col1:
                st.header("Your uploaded image")
                st.image(display_image)

            with col2:
                st.header("Seems like " + predicted_actor)
                st.image(filenames[index_pos], width=300)
    else:
        st.error("Error saving uploaded image.")