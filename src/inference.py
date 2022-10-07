import cv2
import numpy as np
import onnxruntime
import torch
from config import AppConfig


def __detect__(ort_session, image):
    preprocessed_image = cv2.resize(image, (256, 256))
    preprocessed_image = np.rollaxis(np.array(preprocessed_image, np.float32), 2, 0)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    ort_inputs = {ort_session.get_inputs()[0].name: preprocessed_image}
    mask = ort_session.run(None, ort_inputs)
    mask = torch.sigmoid(torch.Tensor(mask[0])).detach().numpy()

    mask = np.rollaxis(mask[0], 0, 3)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask = np.where(mask > 0.7, 1, 0)
    return mask


def run_inference(config: AppConfig):
    model = onnxruntime.InferenceSession(config.checkpoint_weights)
    video_capture = cv2.VideoCapture(0)

    while True:
        # считываем кадр за кадром
        ret, img = video_capture.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = __detect__(model, img)
        blur = cv2.blur(img, (40, 40), 0)
        out = img.copy()
        out[mask == 0] = blur[mask == 0]

        cv2.imshow("Result", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    config = AppConfig.parse_raw()
    print("INFO: Inference started\n")
    run_inference(config)
    print("INFO: Inference over\n")
