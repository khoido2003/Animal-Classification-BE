import sys
import torch
import torchvision

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from PIL import Image
from torch.nn import functional as F
from utils.logger import Logger
from config.animal_cfg import AnimalDataConfig
from .animal_model import AnimalModel

LOGGER = Logger(__file__, log_file='predictor.log')
LOGGER.log.info('Starting Model Serving')

class Predictor:
    def __init__(self, model_name: str, model_weight: str, device: str = 'cpu'):
        self.model_name = model_name
        self.model_weight = model_weight
        self.device = device
        self.load_model()
        self.create_transform()

    async def predict(self, image):
        pil_img = Image.open(image)

        if pil_img.mode == 'RGBA':
            pil_img = pil_img.convert('RGB')

        transformed_image = self.transforms_(pil_img).unsqueeze(0)
        output = await self.model_inference(transformed_image)
        probs, best_prob, predicted_id, predicted_class = self.output2pred(output)

        LOGGER.log_model(self.model_name)
        LOGGER.log_response(best_prob, predicted_id, predicted_class)

        torch.cuda.empty_cache()

        resp_dict = {
            'probs': probs,
            'best_prob': best_prob,
            'predicted_id': predicted_id,
            'predicted_class': predicted_class,
            'predictor_name': self.model_name
        }

        return resp_dict
    
    async def model_inference(self, input):
        input = input.to(self.device)
        with torch.no_grad():
            output = self.loaded_model(input.to(self.device)).cpu()
        return output
    
    def load_model(self):
        try:
            model = AnimalModel(num_classes=10)
            # checkpoint = torch.load("/kaggle/working/trained_models/best.pt")
            checkpoint = torch.load(self.model_weight, map_location = self.device)
            model.load_state_dict(checkpoint['model'])
            model.to(self.device)
            model.eval()

            self.loaded_model = model
        
        except Exception as e:
            LOGGER.log.error(f'Load model failed')
            LOGGER.log.error(f'Error: {e}')

            return None
        
    def create_transform(self):
        self.transforms_ = torchvision.transforms.Compose([
            torchvision.transforms.Resize((AnimalDataConfig.IMG_SIZE, AnimalDataConfig.IMG_SIZE)),
            torchvision.transforms.ToTensor()
        ])
    
    def output2pred(self, output):
        probabilities = F.softmax(output, dim=1)
        best_prob = torch.max(probabilities, 1)[0].item()
        predicted_id = torch.max(probabilities, 1)[1].item()
        predicted_class = AnimalDataConfig.ID2LABEL[predicted_id]

        return probabilities.squeeze().tolist(), round(best_prob, 6), predicted_id, predicted_class