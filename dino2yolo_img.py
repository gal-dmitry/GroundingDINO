# python annotate_image.py

from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

model = load_model(
    "groundingdino/config/GroundingDINO_SwinT_OGC.py", 
    "weights/groundingdino_swint_ogc.pth"
)
IMAGE_PATH = ".asset/cat_dog.jpeg"
TEXT_PROMPT = "chair . person . dog ."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("annotations/annotated_image.jpg", annotated_frame)


# https://www.kdnuggets.com/2023/05/automatic-image-labeling-grounding-dino.html
# # load image
# image = cv2.imread(SOURCE_IMAGE_PATH)

# # detect objects
# detections = grounding_dino_model.predict_with_classes(
#    image=image,
#    classes=enhance_class_name(class_names=CLASSES),
#    box_threshold=BOX_TRESHOLD,
#    text_threshold=TEXT_TRESHOLD
# )

# # print(detections)
# # annotate image with detections
# box_annotator = svn.BoxAnnotator()
# labels = [
#    f"{CLASSES[class_id]} {confidence:0.2f}"
#    for _, _, confidence, class_id, _
#    in detections]
# annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

# # %matplotlib inline
# # svn.plot_image(annotated_frame, (16, 16))