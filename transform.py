from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# usage of transforms -> tensor data type
# Use transforms.ToTensor to solve two probs
# 1. Transform usage
# 2. Tensor data type diff?

# img_path = "data/train/ants_image/0013035.jpg"
img_path = "G:\\AI\\WorkspacePython\\learnPytorch\\data\\train\\ants_image\\0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

#
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
#
writer.add_image("Tensor_img", tensor_img)

writer.close()


