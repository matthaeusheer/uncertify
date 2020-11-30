import torchvision
from uncertify.data.np_transforms import Numpy2PILTransform, NumpyReshapeTransform


H_FLIP_TRANSFORM = torchvision.transforms.Compose([
    NumpyReshapeTransform((200, 200)),
    Numpy2PILTransform(),
    torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.RandomHorizontalFlip(p=1.0),
    torchvision.transforms.ToTensor()
])

V_FLIP_TRANSFORM = torchvision.transforms.Compose([
    NumpyReshapeTransform((200, 200)),
    Numpy2PILTransform(),
    torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.RandomVerticalFlip(p=1.0),
    torchvision.transforms.ToTensor()
])
