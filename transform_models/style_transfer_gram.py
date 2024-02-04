"""
За основу был взят код Tutorials > Neural Transfer Using PyTorch
https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
"""

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms

# Проверим доступность Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print(device)

# выбор слоев сети VGG для вычисления style/content losses :
content_layers_default = ["conv_5"]
style_layers_default = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

# Преобразование изображений для сети VGG из документации
# https://pytorch.org/vision/master/models/generated/torchvision.models.vgg16.html#torchvision.models.vgg16

NORM_STATS = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])


# Загрузка предобученной модели
def load_transform_models() -> models.VGG:
    print("Load transform models...")
    return (
        models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
        .features.eval()
        .to(device)
    )


transform_for_VGG = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(*NORM_STATS)
    ]
)


# обратное преобразование после работы сети к обычному изображению
def denorm(img_tensors):
    return img_tensors * NORM_STATS[1][0] + NORM_STATS[0][0]


tensor_to_image = transforms.Compose([transforms.ToPILImage()])
normalize_image = lambda t: tensor_to_image(
    torch.clamp(denorm(t), min=0, max=1)
)


# Загрузка из image_name и преобразование изображения
def image_loader(image_name: str) -> torch.Tensor:
    image = Image.open(image_name)
    # Сеть принимает данные батчами, поэтому нужно добавить еще одну ось
    # получается тензор мини-батч из одного изображения вида (1, 3, 224, 224)
    image = transform_for_VGG(image).unsqueeze(0)
    return image.to(device, torch.float)


# вывод изображения на экран
def imshow(tensor: torch.Tensor, title: str = None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = tensor_to_image(image)  # normalize_image(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    # plt.pause(0.001) # pause a bit so that plots are updated


# Обратное преобрахование из тензора в обычное изображение
def image_unloader_from_tensor(tensor: torch.Tensor) -> Image:
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = tensor_to_image(image)  # normalize_image(image)
    return image


# create a module to normalize input image so we can easily put it in a
# ``nn.Sequential``
class Normalization(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std


class ContentLoss(nn.Module):
    def __init__(
        self,
        target,
    ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input: torch.Tensor) -> torch.Tensor:
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


def get_style_model_and_losses(
    cnn: models.VGG,
    normalization_mean: torch.Tensor,
    normalization_std: torch.Tensor,
    style_img: torch.Tensor,
    content_img: torch.Tensor,
    content_layers: list[str] = content_layers_default,
    style_layers: list[str] = style_layers_default,
):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std)

    # just in order to have an iterable access to or list of content/style
    # losses
    content_losses = []
    style_losses = []

    # assuming that ``cnn`` is a ``nn.Sequential``,
    # so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = "conv_{}".format(i)
        elif isinstance(layer, nn.ReLU):
            name = "relu_{}".format(i)
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = "pool_{}".format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn_{}".format(i)
        else:
            raise RuntimeError(
                "Unrecognized layer: {}".format(layer.__class__.__name__)
            )

        # print(name)
        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(
            model[i], StyleLoss
        ):
            break

    model = model[: (i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer


def run_style_transfer(
    cnn: models.VGG,
    normalization_mean: torch.Tensor,
    normalization_std: torch.Tensor,
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    input_img: torch.Tensor,
    num_steps: int = 300,
    style_weight: int = 1000000,
    content_weight:int = 1,
) -> torch.Tensor:
    """Run the style transfer."""
    print("Building the style transfer model..")
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img
    )

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    # We also put the model in evaluation mode, so that specific layers
    # such as dropout or batch normalization layers behave correctly.
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print("Optimizing..")

    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 10 == 0:
                # clear_output(wait=True)
                print("run {}:".format(run))
                print(
                    "Style Loss : {:4f} Content Loss: {:4f}".format(
                        style_score.item(), content_score.item()
                    )
                )
                print()
                # imshow(input_img, title="Output Image")
                # plt.show()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


class Model_style_transfer_gram:
    def __init__(self):
        self.model = load_transform_models()

    def get_style_transfer_image(
        self, content_image: str, style_image: str, result_image: str
    ) -> None:
        content_img_batch = image_loader(content_image)
        style_img_batch = image_loader(style_image)

        input_img_batch = content_img_batch.clone()
        output = run_style_transfer(
            self.model,
            cnn_normalization_mean,
            cnn_normalization_std,
            content_img_batch,
            style_img_batch,
            input_img_batch,
            num_steps=100,
            content_weight=1,
        )
        output = image_unloader_from_tensor(output)
        output.save(result_image)
        # return output
