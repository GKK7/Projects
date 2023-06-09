import pickle
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import face_recognition
from PIL import Image, ImageDraw
from sklearn.metrics import classification_report, accuracy_score

DEFAULT_ENCODINGS_PATH = Path("/home/gkirilov/PycharmProjects/face_recognizer/output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

# Create directories if they don't already exist
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)


def encode_known_faces(
    model: str = "cnn", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    names = []
    encodings = []

    filepaths = list(Path("training").glob("*/*"))
    for filepath in tqdm(filepaths, desc="Encoding faces"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)


def recognize_faces(
    image_location: str,
    model: str = "cnn",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> tuple[str, str]:
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    names = []
    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        names.append(name)
        _display_face(draw, bounding_box, name)

    del draw
    # pillow_image.show()  # remove this line if you don't want to display images

    # Returning the most common recognized face in the image
    common_name = Counter(names).most_common(1)[0][0]

    return image_location, common_name


def _recognize_face(unknown_encoding, loaded_encodings, tolerance=0.5):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding, tolerance=tolerance
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]



def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill=BOUNDING_BOX_COLOR,
        outline=BOUNDING_BOX_COLOR,
    )
    draw.text(
        (text_left, text_top),
        name,
        fill=TEXT_COLOR,
    )


def validate(test_folder: Path, model: str = "cnn"):
    y_true = []
    y_pred = []

    for person_folder in test_folder.iterdir():
        if person_folder.is_dir():
            # Loop over desired file types
            for file_type in ["*.jpg", "*.png"]:
                for image_path in person_folder.glob(file_type):
                    true_name = person_folder.name
                    y_true.append(true_name)

                    filename, pred_name = recognize_faces(image_location=str(image_path.absolute()), model=model)
                    print(f"File Name: {filename}, Recognized Person: {pred_name}")
                    y_pred.append(pred_name)

    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Accuracy: ", accuracy_score(y_true, y_pred))


# Example usage
if __name__ == "__main__":
    # Train the model on known faces
    # encode_known_faces()

    # Validate and score the model
    # validation_folder = Path("/home/gkirilov/PycharmProjects/face_recognizer/validation")
    # validate(test_folder=validation_folder, model="cnn")

    # Test the model
    test_folder = Path("/home/gkirilov/PycharmProjects/face_recognizer/testing")
    test_true_labels = []
    test_predictions = []
    for person_folder in test_folder.iterdir():
        if person_folder.is_dir():
            for extension in ["*.jpg", "*.png"]:
                for image_path in person_folder.glob(extension):   # Use `extension` instead of `f"*{extension}"`
                    _, pred_name = recognize_faces(image_location=str(image_path.absolute()), model="cnn")
                    true_name = image_path.parent.name
                    print(f"Image Path: {image_path}, Recognized Person: {pred_name}")
                    test_true_labels.append(true_name)
                    test_predictions.append(pred_name)


    # Now we have the true labels and predictions, so we can compute metrics
    print(classification_report(test_true_labels, test_predictions))
    print("Accuracy: ", accuracy_score(test_true_labels, test_predictions))

