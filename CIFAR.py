import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda, Compose

# https://www.cs.toronto.edu/~kriz/cifar.html
# CIFAR-10 dataset consists of 60000 32x32 colour images
# in 10 classes, with 6000 images per class.
# There are 5000 training images and 1000 test images
public_train_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

public_test_data = datasets.CIFAR10(
    root="data",
    train=False,
    transform=ToTensor(),
)

# This dataset is just like the CIFAR-10,
# except it has 100 classes containing 600 images each.
# There are 500 training images and 100 testing images per class
# grouped into 20 superclasses
private_train_data = datasets.CIFAR100(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

private_test_data = datasets.CIFAR100(
    root="data",
    train=False,
    transform=ToTensor(),
)

private_subcats = [
    ("aquatic mammals", ["beaver", "dolphin", "otter", "seal", "whale"]),
    ("fish", ["aquarium fish", "flatfish", "ray", "shark", "trout"]),
    ("flowers", ["orchids", "poppies", "roses", "sunflowers", "tulips"]),
    ("food containers", ["bottles", "bowls", "cans", "cups", "plates"]),
    ("fruit and vegetables", ["apples", "mushrooms", "oranges", "pears", "sweet peppers"]),
    ("household electrical devices", ["clock", "computer keyboard", "lamp", "telephone", "television"]),
    ("household furniture", ["bed", "chair", "couch", "table", "wardrobe"]),
    ("insects", ["bee", "beetle", "butterfly", "caterpillar", "cockroach"]),
    ("large carnivores", ["bear", "leopard", "lion", "tiger", "wolf"]),
    ("large man-made outdoor things", ["bridge", "castle", "house", "road", "skyscraper"]),
    ("large natural outdoor scenes", ["cloud", "forest", "mountain", "plain", "sea"]),
    ("large omnivores and herbivores", ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"]),
    ("medium-sized mammals", ["fox", "porcupine", "possum", "raccoon", "skunk"]),
    ("non-insect invertebrates", ["crab", "lobster", "snail", "spider", "worm"]),
    ("people", ["baby", "boy", "girl", "man", "woman"]),
    ("reptiles", ["crocodile", "dinosaur", "lizard", "snake", "turtle"]),
    ("small mammals", ["hamster", "mouse", "rabbit", "shrew", "squirrel"]),
    ("trees", ["maple", "oak", "palm", "pine", "willow"]),
    ("vehicles 1", ["bicycle", "bus", "motorcycle", "pickup truck", "train"]),
    ("vehicles 2", ["lawn-mower", "rocket", "streetcar", "tank", "tractor"])
]
