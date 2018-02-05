import torchvision.transforms as transforms
from torch.autograd import Variable

import os
from PIL import Image
import numpy as np

def image_loader(image_name, imsize):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor

    image = Image.open(image_name)
    image = Variable(loader(image))
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image

def image_loader_gray(image_name, imsize):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor

    image = Image.open(image_name).convert('L')
    image = np.asarray(image)
    image = np.asarray([image,image,image])
    image = Image.fromarray(np.uint8(image).transpose(1,2,0))
    image = Variable(loader(image))
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image

def save_image(tensor, size, input_size, fname='transferred.png'):
    unloader = transforms.ToPILImage()  # reconvert into PIL image

    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    image = image.view(size)
    image = unloader(image).resize(input_size)

    out_path = os.path.join('transferred', fname)
    if not os.path.exists('transferred'):
        os.mkdir('transferred')

    image.save(out_path)
