from bcos_layer import BcosConv2d
from torch_radon import Radon

import gc
import math
from sklearn.metrics import classification_report, accuracy_score

import torch
from torchvision import transforms, datasets, models

#######################################################################################################LOAD DATASET#######################################################################################################
class AddInverse(torch.nn.Module):
    """To a [B, C, H, W] input add the inverse channels of the given one to it. Results in a [B, 2C, H, W] output."""

    def __init__(self):
        super().__init__()

    def forward(self, in_tensor):
        return torch.cat([in_tensor, 1 - in_tensor], dim=-3)


def load_dataset(dataset_name, batch_size, is_model_bcos):
    '''
     dataset_name: string[imagenet / cifar100]
       batch_size: int
    is_model_bcos: boolean[True / False]
    '''
    image_size = (224, 224)
    if dataset_name == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif dataset_name == 'cifar100':
        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2762]

    train_transform = [
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    validation_transform = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    if is_model_bcos:
        train_transform.append(AddInverse())
        validation_transform.append(AddInverse())
    
    train_transform = transforms.Compose(train_transform)
    validation_transform = transforms.Compose(validation_transform)

    if dataset_name == 'imagenet':
        train_dataset = datasets.ImageFolder('./../image-net-data/training-set', transform=train_transform)# '/home/public/imagenet/train'
        validation_dataset = datasets.ImageFolder('./../image-net-data/testing-set', transform=validation_transform)# '/home/public/imagenet/val'
    elif dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(root='/home/public/cifar100/', train=True, download=True, transform=train_transform)
        validation_dataset = datasets.CIFAR100(root='/home/public/cifar100/', train=False, download=True, transform=validation_transform)

    assert set(train_dataset.classes) == set(validation_dataset.classes)
    assert len(train_dataset.classes) == len(validation_dataset.classes)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    return image_size, {v: k for k, v in train_dataset.class_to_idx.items()}, len(train_dataset.classes), train_loader, validation_loader

########################################################################################################LOAD MODEL########################################################################################################
def load_model(is_model_bcos, model_name, is_model_teacher, num_classes, state_dict=None):
    """
       is_model_bcos: boolean[True/False]
          model_name: string[one of the following - resnet[18, 34, 50, 152] or densenet[121, 169] or vit_b_[16]]
    is_model_teacher: boolean[True/False]
         num_classes: int[number of classes in the dataset]
          state_dict: string[path/to/trained/model]
    """
    if is_model_bcos:
        model = torch.hub.load('B-cos/B-cos-v2', model_name, pretrained=is_model_teacher)
        # in_channels=512 this is the default value in the layer. keeping it as is.
        model.fc = BcosConv2d(in_channels=512, out_channels=num_classes)
    else:
        if model_name == 'resnet18':
            if is_model_teacher:
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet18(weights=None)
        elif model_name == 'resnet34':
            if is_model_teacher:
                model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet34(weights=None)
        elif model_name == 'resnet50':
            if is_model_teacher:
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet50(weights=None)
        elif model_name == 'resnet152':
            if is_model_teacher:
                model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet152(weights=None)
        elif model_name == 'densenet121':
            if is_model_teacher:
                model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            else:
                model = models.densenet121(weights=None)
        elif model_name == 'densenet169':
            if is_model_teacher:
                model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
            else:
                model = models.densenet169(weights=None)
        elif model_name == 'vit_b_16':
            if is_model_teacher:
                model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            else:
                model = models.vit_b_16(weights=None)

        if 'resnet' in model_name:
            model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
        elif 'densenet' in model_name:
            model.classifier = torch.nn.Linear(in_features=model.classifier.in_features, out_features=num_classes)
        elif 'vit' in model_name:
            model.heads.head = torch.nn.Linear(in_features=model.heads.head.in_features, out_features=num_classes)
        
        if state_dict:
            model.load_state_dict(torch.load(state_dict))

    return model

#####################################################################################################RadonCDT FOR SWD#####################################################################################################
class RadonCDT:
    def __init__(self, image_shape, num_projections=100, epsilon=1e-10):
        self.radon = Radon(image_shape, torch.linspace(0., 180., num_projections), image_shape, 1.0, clip_to_circle=True)
        self.epsilon = epsilon

    def _cdt(self, signal):
        sorted_signal, _ = torch.sort(signal, dim=-1)
        cdf = torch.cumsum(sorted_signal, dim=-1) / (torch.sum(sorted_signal, dim=-1, keepdim=True) + self.epsilon) # may not be necessary. check if final value is 1.
        return torch.nn.functional.interpolate(cdf.unsqueeze(1), size=signal.shape[-1], mode='linear', align_corners=True).squeeze(1) # interpolate method may not be right.

    def transform(self, image):
        radon_img = self.radon.forward(image)
        return torch.stack([self._cdt(radon_img[:, i]) for i in range(radon_img.shape[1])], dim=1)


####################################################################################################TorchRadon FOR SWD####################################################################################################
def torch_radon_transform(I, theta, device):
    B, H, W = I.shape

    diag_len = int(math.ceil(math.sqrt(H ** 2 + W ** 2)))
    pad_H, pad_W = (diag_len - H) // 2, (diag_len - W) // 2
    padded_I = torch.nn.functional.pad(I, (pad_W, pad_W + (diag_len - W - 2 * pad_W), pad_H, pad_H + (diag_len - H - 2 * pad_H)))

    radonI = torch.zeros((B, diag_len, len(theta)), device=device)

    for i, angle in enumerate(theta):
        rotated_I = rotate_image(padded_I, angle, device)
        radonI[:, :, i] = rotated_I.sum(dim=-1)

    return radonI

def rotate_image(I, angle, device):
    angle = torch.tensor(angle, device=device, dtype=torch.float32)
    theta = torch.tensor([
        [torch.cos(angle), -torch.sin(angle), 0],
        [torch.sin(angle), torch.cos(angle), 0]
    ], device=device).unsqueeze(0).repeat(I.size(0), 1, 1)

    grid = torch.nn.functional.affine_grid(theta, I.unsqueeze(1).size(), align_corners=False)
    rotated_I = torch.nn.functional.grid_sample(I.unsqueeze(1), grid, align_corners=False).squeeze(1)

    return rotated_I
    
#########################################################################################################GradCAM##########################################################################################################
class ActivationsAndGradients:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self._get_target_layer()

        self.activations = None
        self.gradients = None
        
        self.hook_handles = []
        self.hook_handles.append(self.target_layer.register_forward_hook(self._save_activation))
        self.hook_handles.append(self.target_layer.register_backward_hook(self._save_gradient))

    def _get_target_layer(self):
        if "resnet" in self.model_name:
            if self.model_name in ["resnet18", "resnet34"]:
                self.target_layer = self.model.layer4[-1].conv2
            elif self.model_name in ["resnet50", "resnet152"]:
                self.target_layer = self.model.layer4[-1].conv3
        elif "densenet" in self.model_name:
            if self.model_name in ["densenet121"]:
                self.target_layer = self.model.features.denseblock4.denselayer16.conv2
            elif self.model_name in ["densenet169"]:
                self.target_layer = self.model.features.denseblock4.denselayer32.conv2
        elif "vit" in self.model_name:
            if self.model_name in ["vit_b_16"]:
                self.target_layer = self.model.encoder.layers[-1].ln_1

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

def generate_gradcam_heatmaps(model, model_name, images, labels, image_size):
    agg = ActivationsAndGradients(model, model_name)
    outputs = model(images)
    grads = torch.zeros_like(agg.activations)

    if "vit" in model_name:
        patch_size = model._modules.get("conv_proj").kernel_size
        U, V = images.shape[2] // patch_size[0], images.shape[3] // patch_size[1]

    for i in range(len(labels)):
        class_score = outputs[i, labels[i]]
        grad = torch.autograd.grad(outputs=class_score, inputs=agg.activations, retain_graph=True, allow_unused=True)[0]
        grads[i] = grad[i]

    if "vit" in model_name:
        pooled_grads = torch.mean(grads, dim=1)
    else:
        pooled_grads = torch.mean(grads, dim=[0, 2, 3])

    activations = agg.activations
    if "vit" in model_name:
        activations = activations[:, 1:, :]
        activations = activations.reshape(activations.shape[0], -1, activations.shape[-1])
        for i in range(activations.size(2)):
            activations[:, :, i] *= pooled_grads[:, i].unsqueeze(1)
        heatmaps = torch.mean(activations, dim=2).reshape(-1, U, V)
    else:
        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_grads[i]
        heatmaps = torch.mean(activations, dim=1).squeeze()

    heatmaps = torch.relu(torch.nn.functional.interpolate(heatmaps.unsqueeze(1), size=image_size, mode='bilinear', align_corners=False).squeeze())

    min_vals = heatmaps.view(heatmaps.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1)
    max_vals = heatmaps.view(heatmaps.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1)
    
    heatmaps = (heatmaps - min_vals) / (max_vals - min_vals + 1e-10)
    heatmaps = heatmaps / heatmaps.sum(dim=(1, 2), keepdim=True)

    agg.remove_hooks()

    del agg
    del outputs
    del grads
    del pooled_grads
    del activations
    gc.collect()
    
    return heatmaps

########################################################################################################B-cos EXPLANATIONS########################################################################################################
def generate_bcos_explanations(model, images, labels):
    model.eval()
    images.requires_grad = True
    with torch.enable_grad(), model.explanation_mode():
        logits = model(images)[range(images.size()[0]), labels]
        gradients = torch.autograd.grad(outputs=logits, inputs=images, grad_outputs=torch.ones_like(logits), retain_graph=True)[0]

    bcos_explanations = (images * gradients).sum(dim=1)

    min_vals = bcos_explanations.view(bcos_explanations.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1)
    max_vals = bcos_explanations.view(bcos_explanations.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1)
    
    bcos_explanations = (bcos_explanations - min_vals) / (max_vals - min_vals + 1e-10)
    bcos_explanations = bcos_explanations / bcos_explanations.sum(dim=(1, 2), keepdim=True)

    del gradients
    del min_vals
    del max_vals
    gc.collect()

    return bcos_explanations

########################################################################################################EVALUATION########################################################################################################
def evaluate(model, validation_loader, idx_to_tag):
    device = torch.device("cuda")
    model.eval()
    model.to(device)
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = [idx_to_tag[x] for x in y_true]
    y_pred = [idx_to_tag[x] for x in y_pred]

    print(classification_report(y_true=y_true, y_pred=y_pred))
    return accuracy_score(y_true=y_true, y_pred=y_pred)