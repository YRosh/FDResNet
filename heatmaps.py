import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import PIL
from torchvision import transforms
from torchvision import datasets
from gradcam import grad_cam
from models import *
from utils import progress_bar
import random
import time
import os
import numpy as np
import torch

transform_train = transforms.Compose([
    transforms.Resize((128, 128)), # PIL.Image.BICUBIC 
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainset = datasets.ImageFolder(root="/content/content/test", transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, num_workers=1, shuffle=False)

net = ResNet101()
net = torch.nn.DataParallel(net)
checkpoint = torch.load('/content/drive/MyDrive/Resnet50_cifar10/checkpoint/caltech-retrain-actual-run-2.pth')
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']
print(best_acc, start_epoch, "Resnet34 loaded")
net.load_state_dict(checkpoint['net'])
net.eval()

net_fg = FG_ResNet101()
net_fg = torch.nn.DataParallel(net_fg)
checkpoint = torch.load('/content/drive/MyDrive/Resnet50_Rahul/checkpoint/caltech-retrain-l3-h5-run-2.pth')
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']
print(best_acc, start_epoch, "FG_Resnet 34 loaded")
net_fg.load_state_dict(checkpoint['net'])
net_fg.eval()
# for name, param in net.named_parameters():
#   print(name)

heatmaps = []

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

unnorm = UnNormalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
m = torch.nn.Softmax(dim=1)
res_wrong, fg_res_wrong, inter, union, res_right_fg_wrong, fg_right_res_wrong  = [], [], [], [], [], []
def test():
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total = 0
    l = []
    c = 0
    actual_arr = []
    fg_arr = []
    #with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        outputs_fg = net_fg(inputs)
        actual_class = targets.item()
        #print(outputs.shape)
        # loss = criterion(outputs, targets)
        # test_loss += loss.item()
        _, predicted = outputs.max(1)
        _, predicted_fg = outputs_fg.max(1)

        # if (predicted.item() != targets.item() or predicted_fg.item() != targets.item()):
        #   union.append(targets.item())

        # if (predicted.item() != targets.item()):
        #   res_wrong.append(targets.item())

        # if (predicted_fg.item() != targets.item()):
        #   fg_res_wrong.append(targets.item())

        # if (predicted.item() != targets.item() and predicted_fg.item() != targets.item()):
        #   inter.append(targets.item())
        
        # if(predicted.item() != targets.item() and predicted_fg.item() == targets.item()):
        #   res_right_fg_wrong.append(targets.item())
        
        # if(predicted.item() == targets.item() and predicted_fg.item() != targets.item()):
        #   fg_right_res_wrong.append(targets.item())

        # print("normal")
        #print(list(m(outputs).tolist())[0])
        #print(len(list(m(outputs).tolist())[0]))

        # print()
        # print("FG")
        # print(outputs_fg.max(1))
        if(predicted.item() != targets.item()) and (predicted_fg.item() == targets.item()) and (targets.item() not in l and list(m(outputs_fg).tolist())[0][actual_class] > 0.5):
            l.append(targets.item())
            inputs = inputs.squeeze()
            img = transforms.ToPILImage()(unnorm(inputs)).convert("RGB")
            img.save('/content/drive/MyDrive/Caltech-256/outputs-caltech-05-greater/image-{}.jpg'.format(c))
            c += 1
            p = list(m(outputs).tolist())[0]
            idx = p.index(max(p))
            actual_arr.append((p[actual_class], [max(p), idx]))
            fg_arr.append((list(m(outputs_fg).tolist())[0][actual_class], list(m(outputs_fg).tolist())[0][idx]))
        #     #gen_heatmaps(inputs)
        if len(l) == 10:
        #     # imgs_comb = np.hstack( (np.asarray(i) for i in heatmaps ) )
        #     # imgs_comb = PIL.Image.fromarray( imgs_comb)
        #     # imgs_comb.save('/content/drive/MyDrive/Caltech-256/heatmap-conv-1.jpg')
             break
    print(actual_arr)
    print(fg_arr)
    # print(res_wrong, len(res_wrong))
    # print(fg_res_wrong, len(fg_res_wrong))
    # print(inter, len(inter))
    # print(union, len(union))
    # print(res_right_fg_wrong, len(res_right_fg_wrong))
    # print(fg_right_res_wrong, len(fg_right_res_wrong))
    # print(res_wrong, fg_res_wrong, inter, union, res_right_fg_wrong, fg_right_res_wrong)
    # print(fg_arr)
        #print(predicted, targets)
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

def gen_heatmaps(img):
    # print(img)
    img = img.squeeze().cpu()
    # print(img)
    #print(img.shape)
    heatmap_layer = net.module.conv1
    image1 = grad_cam(net, img, heatmap_layer)

    # heatmap_layer = net.module.layer3[22].conv3
    # image2 = grad_cam(net, img, heatmap_layer)

    # heatmap_layer = net_fg.module.layer2[3].conv3
    # image3 = grad_cam(net_fg, img, heatmap_layer)

    heatmap_layer = net_fg.module.conv1
    image4 = grad_cam(net_fg, img, heatmap_layer)

    img = unnorm(img)
    img = transforms.ToPILImage()(img).convert("RGB")

    imgs_comb = np.vstack( (np.asarray(i) for i in [img, image1, image4] ) )
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    heatmaps.append(imgs_comb)

def heatmap():
    for i, folder in enumerate(os.listdir("/content/train"), 0):
        # model = torchvision.models.resnet34(pretrained=True)
        if folder in ['001.ak47', '016.boom-box', '033.cd', '087.goldfish', '107.hot-air-balloon', '080.frog', '179.scorpion-101', '224.touring-bike', '142.microwave', '175.roulette-wheel']:
            net.eval()
            file_name = os.listdir("/content/train/"+folder)[0]

            image = Image.open("/content/train/"+folder+"/"+file_name)
            input_tensor = transform_train(image)

            heatmap_layer = net.module.conv1
            image = grad_cam(net, input_tensor, heatmap_layer)
            plt.imshow(image)
            plt.savefig('/content/drive/MyDrive/Caltech-256/heatmaps/{}-input'.format(folder.split(".")[1]))

            heatmap_layer = net.module.layer1[2].conv3
            image = grad_cam(net, input_tensor, heatmap_layer)
            plt.imshow(image)
            plt.savefig('/content/drive/MyDrive/Caltech-256/heatmaps/{}-output'.format(folder.split(".")[1]))

criterion = nn.CrossEntropyLoss()
def testing():
  test_loss_res = 0
  test_loss_fg = 0
  correct_res = 0
  correct_fg = 0
  total = 0
  with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(trainloader):
          inputs, targets = inputs.to(device), targets.to(device)

          outputs = net(inputs)
          outputs_fg = net_fg(inputs)

          loss = criterion(outputs, targets)
          loss_fg = criterion(outputs_fg, targets)

          test_loss_res += loss.item()
          test_loss_fg += loss_fg.item()

          _, predicted = outputs.max(1)
          _, predicted_fg = outputs_fg.max(1)

          total += targets.size(0)
          correct_res += predicted.eq(targets).sum().item()
          correct_fg += predicted_fg.eq(targets).sum().item()
      acc = 100.*correct_res/total
      acc_fg = 100.*correct_fg/total
      print("ResNet: {}".format(acc))
      print("FGResNet: {}".format(acc_fg))


#testing()
test()
#heatmaps()