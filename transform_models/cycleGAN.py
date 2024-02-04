import os
from PIL import Image
import sys
import torch
import torchvision.transforms as transforms

sys.path.append("transform_models/cycleGAN")

# Веса тренированной модели скачаны с
# http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/
# и подготовлены с помощью /cycleGAN/prepare_model.py

PTH_PATH = "transform_models/pth/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


# Класс cycle GAN модели
# пример создания:  model = Model_style_transfer_CGAN('model_name')
# model_name должно совпадать с одним из файлов подготовленных весов
class Model_style_transfer_CGAN:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self.load_pretrain_model()

        # преобразование изображения для подачи в предобученные модели
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(286),
                transforms.CenterCrop(256),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    # Преобразование изображения content_image и сохранение в result_image
    def get_style_transfer_image(self, content_image : str, result_image : str) -> None:
        # загружаем из файла
        content_img_batch = self.image_loader(content_image)
        # получаем вывод сети
        output = self.model.test(content_img_batch)
        # делаем обратное преобразование
        output = self.image_unloader(output)
        # сохраняем в файл
        output.save(result_image)
        # return output

    # Загрузка предобученной модели
    def load_pretrain_model(self):
        model_path = '%s.pth' % (self.model_name)
        model_path = os.path.join(PTH_PATH, model_path)
        print(f"Load pretrain CGAN model from {model_path}")
        model = torch.load(model_path, map_location=torch.device(device))
        # for p in model.parameters():
        #     p.requires_grad = False
        model.eval()  # .to(device)
        return model

    # Загрузка из img_path и преобразование изображения
    def image_loader(self, img_path: str):
        image = Image.open(img_path)
        # Сеть принимает данные батчами,
        # поэтому после преобразования нужно добавить еще одну ось
        # получается мини-батч тензор из одного изображения
        image = self.transform(image).unsqueeze(0)
        return image.to(device, torch.float)

    def image_unloader(self, img_tensor: torch.Tensor) -> Image:
        image = img_tensor.cpu().clone()  # клонируем исходный тензор
        image = image.squeeze(0)  # remove the fake batch dimension
        image = torch.clamp(
            image * 0.5 + 0.5, min=0, max=1
        )  # обратное преобразование
        tensor_to_image = transforms.Compose([transforms.ToPILImage()])
        image = tensor_to_image(image)

        return image
