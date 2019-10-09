"""
Made by @nizhib
"""

import torch
from torchvision import transforms
from PIL import Image


class Segmentator(object):
    size = (400, 300)

    meanstd = {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
    normalize = transforms.Normalize(**meanstd)
    preprocess = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        normalize
    ])

    def __init__(self, name):
        self.net = torch.jit.load(name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net.to(self.device)

    @torch.no_grad()
    def predict(self, image):
        image = self.preprocess(image)
        tensor = torch.stack((image,)).to(self.device)
        _, logits = self.net(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0, 1, ...]
        return probs


def main():
    seg = Segmentator("scriptmodule.pt")

    tmp = numpy.random.rand(800,600,3) * 255
    tmp = Image.fromarray(imarray.astype('uint8')).convert('RGB')
    out = seg.predict(tmp)


if __name__=="__main__":
    main()
