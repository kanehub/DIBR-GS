import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Depth-anything-v2

class DepthEstimator():
    '''
    input PIL image
    '''
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.load_model()

    def load_model(self):
        dpt_model_path = "depth-anything/Depth-Anything-V2-Small-hf"
        self.model = AutoModelForDepthEstimation.from_pretrained(dpt_model_path).to(self.device)
        self.image_processor = AutoImageProcessor.from_pretrained(dpt_model_path)

    def predict(self, image, mode='test'):
        if mode == 'test':
            # input PIL image
            # prepare PIL image for the model
            inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
            # interpolate to original size and visualize the prediction
            post_processed_output = self.image_processor.post_process_depth_estimation(
                outputs,
                target_sizes=[(image.height, image.width)],
            )

            predicted_depth = post_processed_output[0]["predicted_depth"].squeeze()
            predicted_depth = predicted_depth.detach().cpu().numpy()
            return predicted_depth

