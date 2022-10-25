# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from facenet_pytorch import InceptionResnetV1

"""## Helper Functions

- **process(img)** : convert StyleGAN generated image for visualization by matplotlib
- **process_image_sequence(image_sequence, target)**: process image sequence for visualization by matplotlib
- **stylegan_postprocess(img)** : simple crop & normalization for StyleGAN
- **VGGPerceptualLoss(img, target)** : for computing LPIPS between generated & target image
- **CosineDistanceLoss(img, target)** : for computing cosine distance  between generated & target image

For the evaluation of generated image, we will use L2 distance, LPIPS, and cosine distance.  
For L2 distance, you can simply use nn.MSELoss().  
For the LPIPS and cosine distance, please use the above helper functions.  
You can define more custom functions for better performance!
"""

def process(img):
    """
    process img for visualization by matplotlib
    img: StyleGAN generated image tensor of shape [1,3,H,W]
    return: image tensor of shape [H,W,3]
    """
    img = img.detach().cpu()
    img = torchvision.utils.make_grid(img, nrow=1, normalize=True)
    img = img.permute(1,2,0)  # reshape to [H,W,3]
    
    return img


def process_image_sequence(image_sequence, target):
    """
    process image sequence for visualization by matplotlib
    image_sequence: list of N generated images, each of shape [1,3,H,W]
    target: target image of shape [1,3,H,W]
    return: image tensor of shape [H, (N+1)*W, 3]
    """
    N = len(image_sequence) + 1
    imseq = image_sequence + [target]
    imseq = torch.vstack(imseq)
    imseq = torchvision.utils.make_grid(imseq, nrow=N, normalize=True)
    imseq = imseq.detach().cpu()  # detach gradient & save to CPU
    imseq = imseq.permute(1,2,0)  # reshape to [H, (N+1)*W, 3]
    
    return imseq

def stylegan_postprocess(img, crop_size=192, v_offset=10, normalize=False):
    '''
    postprocessing for StyleGAN-FFHQ-256
    crops and normalizes the generated image
    '''
    _, _, cy, cx = img.shape
    assert len(img.shape) == 4 and img.shape[1] == 3, 'img must be a Bx3xHxW numpy array'
    assert cy >= crop_size and cx >= crop_size, 'crop size must be smaller than the given image'
    cy = cy // 2 + v_offset  # vertical offset
    cx = cx // 2
    w = crop_size // 2
    img = img[:, :, cy-w:cy+w, cx-w:cx+w]
    
    if normalize:
        img = 2 * (img - img.min()) / (img.max() - img.min()) - 1  # normalize -1~1
        
    return img

class VGGPerceptualLoss(nn.Module):
    """
    perceptual loss using ImageNet trained VGG-16
    can be used to compute LPIPS
    """
    DEFAULT_FEATURE_LAYERS = [0, 1, 2, 3]
    IMAGENET_RESIZE = (224, 224)

    def __init__(self, resize=True, feature_layers=None, style_layers=None):
        super().__init__()
        self.resize = resize
        self.feature_layers = feature_layers or self.DEFAULT_FEATURE_LAYERS
        self.style_layers = style_layers or []
        features = torchvision.models.vgg16(pretrained=True).features
        self.blocks = nn.ModuleList([
            features[:4].eval(),
            features[4:9].eval(),
            features[9:16].eval(),
            features[16:23].eval(),
        ])
        for param in self.parameters():
            param.requires_grad = False

    def _transform(self, tensor):
        if self.resize:
            tensor = nn.functional.interpolate(tensor, mode='bilinear', size=self.IMAGENET_RESIZE, align_corners=False)
        return tensor

    def _calculate_gram(self, tensor):
        act = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)
        return act @ act.permute(0, 2, 1)

    def forward(self, output, target):
        output, target = self._transform(output), self._transform(target)
        loss = 0.
        for i, block in enumerate(self.blocks):
            output, target = block(output), block(target)
            if i in self.feature_layers:
                loss += nn.functional.l1_loss(output, target)
            if i in self.style_layers:
                gram_output, gram_target = self._calculate_gram(output), self._calculate_gram(target)
                loss += nn.functional.l1_loss(gram_output, gram_target)
        return loss

def cosine_similarity(x, y):
    """
    Batch-wise cosine similarity between x, y of shape [B, d]
    B is the batch size, d is the feature dimension
    returns cosine similarity matrix of shape [B, B]
    """
    x_norm = F.normalize(x, dim=1)
    y_norm = F.normalize(y, dim=1)
    cosine = torch.mm(x_norm, y_norm.T).clamp(-1,1)

    return cosine


class CosineDistanceLoss(nn.Module):
    """
    Cosine distance loss using FaceNet trained from VGGFace2
    """
    FACENET_RESIZE = (160, 160)
    def __init__(self, resize=True):
        super(CosineDistanceLoss, self).__init__()
        self.resize = resize
        self.encoder = InceptionResnetV1(pretrained='vggface2')
        self.encoder.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def _transform(self, tensor):
        if self.resize:
            tensor = nn.functional.interpolate(tensor, mode='bilinear', size=self.FACENET_RESIZE, align_corners=False)
        return tensor
    
    def forward(self, output, target):
        output, target = self._transform(output), self._transform(target)
        output, target = self.encoder(output), self.encoder(target)
        loss = 1 - cosine_similarity(output, target)
        
        return loss

