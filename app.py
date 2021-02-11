import os
import time
import pathlib
# model
import numpy as np
import tensorflow as tf
# visualize
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
# helper
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# set up default variables
# path for saved model and label index
PATH_TO_SAVED_MODEL = "model/saved_model"
PATH_TO_LABEL = "model/label_map.pbtxt"

DEFAULT_THRESHOLD = .35


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_detector(path_to_saved_model, path_to_label):
    # load saved model and build the detection function
    detect_fn = tf.saved_model.load(path_to_saved_model)
    # load label index
    label_index = label_map_util.create_category_index_from_labelmap(path_to_label,
                                                            use_display_name=True)
    return detect_fn, label_index

# function image to tensor
@st.cache
def load_image_to_tensor(image):
    img_np = np.array(image)
    img_tensor = tf.convert_to_tensor(img_np)
    return img_np, img_tensor[tf.newaxis, ...]

def inference(image, threshold, n_boxes, path_to_saved_model, path_to_label):

    # load image, model, and index
    detect_fn, label_index = load_detector(path_to_saved_model, path_to_label)
    image_np, image_tensor = load_image_to_tensor(image)

    # make inference
    detections = detect_fn(image_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # annotate image
    # image_np_with_detections = image_np.copy()
    image_np_with_detections = viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np.copy(),
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            label_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=n_boxes,
            min_score_thresh=threshold,
            agnostic_mode=False)
    
    return image_np_with_detections

def main():
    st.title("Furniture Detection 🪑👀")

    st.sidebar.header("Upload your image")
    uploaded_image = st.sidebar.file_uploader("Choose a png or jpg image", 
                                      type=["jpg", "png", "jpeg"])

    threshold = st.sidebar.slider("Confidence threshold",
                                    0.0, 1.0, DEFAULT_THRESHOLD, 0.01)
    n_boxes = st.sidebar.slider(label="Number of boxes to draw",
                                min_value=1, 
                                max_value=10, 
                                value=10)
    
    resize = st.sidebar.checkbox("Resize images")

    if uploaded_image is not None:
        image = Image.open(uploaded_image)

        if resize:
            w, h = image.size
            height = 512
            width = int((h/height) * w)
            image = image.resize((width, height))

        # make sure image is RGB
        image = image.convert("RGB")

        if st.sidebar.button("Make a prediction"):
            "Making a prediction and drawing", n_boxes, "boxes on your image..."
            with st.spinner("Predicting..."):
                image_np_with_detections = inference(image,
                                                    threshold,
                                                    n_boxes,
                                                    PATH_TO_SAVED_MODEL,
                                                    PATH_TO_LABEL)
                
                st.image(image_np_with_detections, use_column_width=True)
                if resize:
                    st.text("Image resized to {}x{}".format(width, height))
        else:
            st.subheader("Your image will displayed here...")

if __name__ == "__main__":
    main()