from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

write = SummaryWriter("logs")
img = Image.open("pics/1.jpeg")
print(img)

# ToTensor
trans_toTensor = transforms.ToTensor()
img_tensor = trans_toTensor(img)
write.add_image("ToTensor", img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([6, 3, 2], [9, 3, 5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
write.add_image("Normalize", img_norm, 2)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
# img_resize PIL -> toTensor -> img_resize tensor
img_resize = trans_toTensor(img_resize)
write.add_image("Resize", img_resize, 0)
print(img_resize)

# Compose - resize -2
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_toTensor])
img_resize_2 = trans_compose(img)
write.add_image("Resize", img_resize_2, 1)

# RandomCrop


write.close()
