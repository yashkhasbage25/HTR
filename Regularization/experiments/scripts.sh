# Lenet + MNIST
# sgd
# 98.75 + 0.03
python3 no_penalization.py -lr 1e-2 -n 30 -d MNIST -b 256 -m LeNet -r "model=LeNet,dataset=MNIST,no,id=1" --milestones 10 20 --lr_gamma 0.25 -reruns 3
# sgd + htr
# use rademacher vectors as the commented code follows
python3 ht_penalization.py -lr 1e-2 -l2 0.0 -n 30 -d MNIST -b 256 -m LeNet -r "model=LeNet,dataset=MNIST,ht,id=1" --milestones 10 20 --lr_gamma 0.25 -reruns 3 -sp 10 -gamma 1e-3 -freq 50 -hi 10
# sgd + htr + l2
# 98.847 + 0.019
python3 combined_penalization.py -lr 1e-2 -l2 1e-4 -n 30 -d MNIST -b 256 -m LeNet -r "model=LeNet,dataset=MNIST,combined,id=1" --milestones 10 20 --lr_gamma 0.25 -reruns 3 -sp 10 -gamma 1e-3 -freq 50 -hi 10

# lenet + fmnist
# sgd 
python3 no_penalization.py -lr 1e-2 -n 30 -d FashionMNIST -b 256 -m LeNet -r "model=LeNet,dataset=FashionMNIST,no,id=1" --milestones 10 20 --lr_gamma 0.25 -reruns 3
# sgd + htr
python3 ht_penalization.py -lr 1e-2 -l2 0.0 -n 40 -d FashionMNIST -b 256 -m LeNet -r "model=LeNet,dataset=FashionMNIST,ht,id=1" --milestones 10 20 --lr_gamma 0.25 -reruns 3 -sp 20 -gamma 1e-5 -freq 50 -hi 10
# sgd + htr + l2
python3 combined_penalization.py -lr 1e-2 -l2 1e-13 -n 30 -d FashionMNIST -b 256 -m LeNet -r "model=LeNet,dataset=FashionMNIST,combined,id=1" --milestones 10 20 --lr_gamma 0.25 -reruns 3 -sp 20 -gamma 1e-5 -freq 50 -hi 10

# lenet + svhn
# sgd
python3 no_penalization.py -lr 1e-1 -n 50 -d SVHN -b 256 -m LeNet -r "model=LeNet,dataset=SVHN,no,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3
# sgd + htr
python3 ht_penalization.py -lr 1e-1 -l2 0.0 -n 50 -d SVHN -b 256 -m LeNet -r "model=LeNet,dataset=SVHN,ht,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3 -sp 0 -gamma 1e-7 -freq 50 -hi 10
# sgd + htr + l2
python3 combined_penalization.py -lr 1e-1 -l2 1e-3 -n 50 -d SVHN -b 256 -m LeNet -r "model=LeNet,dataset=SVHN,combined,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3 -sp 0 -gamma 1e-8 -freq 50 -hi 10

# vgg11_bn + cifar100
# sgd
python3 no_penalization.py -lr 1e-2 -n 50 -d CIFAR100 -b 64 -m VGG11_bn -r "model=VGG11_bn,dataset=CIFAR100,no,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3
# sgd + htr
python3 ht_penalization.py -lr 1e-2 -l2 0.0 -n 60 -d CIFAR100 -b 64 -m VGG11_bn -r "model=VGG11_bn,dataset=CIFAR100,ht,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3 -sp 0 -gamma 1e-6 -freq 50 -hi 5
# sgd + l2 + htr
python3 combined_penalization.py -lr 1e-2 -l2 1e-4 -n 50 -d CIFAR100 -b 64 -m VGG11_bn -r "model=VGG11_bn,dataset=CIFAR100,combined,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3 -sp 0 -gamma 1e-6 -freq 50 -hi 5

# resnet + cifar10
# sgd 
python3 no_penalization.py -lr 1e-2 -n 40 -d CIFAR10 -b 64 -m ResNet18 -r "model=ResNet18,dataset=CIFAR10,no,id=1" --milestones 20 30 --lr_gamma 0.25 -reruns 3
# sgd + htr
python3 ht_penalization.py -lr 1e-2 -l2 0.0 -n 40 -d CIFAR10 -b 64 -m ResNet18 -r "model=ResNet18,dataset=CIFAR10,ht,id=1" --milestones 20 30 --lr_gamma 0.25 -reruns 3 -sp 10 -gamma 1e-6 -freq 50 -hi 3
# sgd + htr + l2
python3 combined_penalization.py -lr 1e-2 -l2 1e-4 -n 40 -d CIFAR10 -b 64 -m ResNet18 -r "model=ResNet18,dataset=CIFAR10,combined,id=1" --milestones 20 30 --lr_gamma 0.25 -reruns 3 -sp 0 -gamma 1e-12 -freq 50 -hi 3

# vgg11 + cifar10
# sgd 
python3 no_penalization.py -lr 1e-2 -n 50 -d CIFAR10 -b 64 -m VGG11 -r "model=VGG11,dataset=CIFAR10,no,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3
# sgd + htr
python3 ht_penalization.py -lr 1e-2 -l2 0.0 -n 50 -d CIFAR10 -b 64 -m VGG11 -r "model=VGG11,dataset=CIFAR10,ht,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3 -sp 0 -gamma 1e-6 -freq 50 -hi 5
# sgd + htr + l2
python3 combined_penalization.py -lr 1e-2 -l2 1e-12 -n 50 -d CIFAR10 -b 64 -m VGG11 -r "model=VGG11,dataset=CIFAR10,combined,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3 -sp 0 -gamma 1e-6 -freq 50 -hi 5

# resnet34 + cifar100
# sgd
python3 no_penalization.py -lr 1e-2 -n 50 -d CIFAR100 -b 64 -m ResNet34 -r "model=ResNet34,dataset=CIFAR100,no,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3
# sgd + htr
python3 ht_penalization.py -lr 1e-2 -l2 0.0 -n 50 -d CIFAR10 -b 64 -m ResNet34 -r "model=ResNet34,dataset=CIFAR100,ht,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3 -sp 0 -gamma 1e-3 -freq 50 -hi 5


# vgg13 + cifar10
# sgd 
python3 no_penalization.py -lr 1e-4 -n 50 -d CIFAR10 -b 64 -m VGG13 -r "model=VGG13,dataset=CIFAR10,no,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3
# sgd + l2 
python3 l2_penalization.py -lr 1e-2 -l2 1e-4 -n 50 -d CIFAR10 -b 64 -m VGG13 -r "model=VGG13,dataset=CIFAR10,l2,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3

# vgg19 + cifar10
# sgd 
python3 no_penalization.py -lr 1e-2 -n 50 -d CIFAR10 -b 64 -m VGG19 -r "model=VGG19,dataset=CIFAR10,no,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3

# vgg13 + svhn
# sgd
python3 no_penalization.py -lr 5e-4 -n 50 -d SVHN -b 64 -m VGG13 -r "model=VGG13,dataset=SVHN,no,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3
# sgd + htr
python3 ht_penalization.py -lr 5e-4 -l2 0.0 -n 50 -d SVHN -b 64 -m VGG13 -r "model=VGG13,dataset=SVHN,ht,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3 -sp 0 -gamma 1e-3 -freq 50 -hi 5
# sgd + htr + l2
python3 combined_penalization.py -lr 5e-4 -l2 1e-3 -n 50 -d SVHN -b 64 -m VGG13 -r "model=VGG13,dataset=SVHN,combined,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3 -sp 0 -gamma 1e-6 -freq 50 -hi 5

# vgg11 + svhn
# sgd 
python3 no_penalization.py -lr 1e-2 -n 50 -d SVHN -b 64 -m VGG11 -r "model=VGG11,dataset=SVHN,no,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3

# vgg16 + svhn
# sgd
python3 no_penalization.py -lr 5e-4 -n 50 -d SVHN -b 64 -m VGG16 -r "model=VGG16,dataset=SVHN,no,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3


#####################################
# middle layer
#####################################
# vgg11bn + cifar100
python3 middle_penalization.py -lr 1e-2 -l2 0.0 -n 50 -d CIFAR100 -b 64 -m VGG11_bn -r "model=VGG11_bn,dataset=CIFAR100,mid,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3 -sp 0 -gamma 1e-7 -freq 50 -hi 10 --middle
# resnet18 + cifar10
python3 middle_penalization.py -lr 1e-2 -l2 0.0 -n 40 -d CIFAR10 -b 64 -m ResNet18 -r "model=ResNet18,dataset=CIFAR10,mid,id=1" --milestones 20 30 --lr_gamma 0.25 -reruns 3 -sp 10 -gamma 1e-5 -freq 50 -hi 3 --middle
# lenet + fash
python3 middle_penalization.py -lr 1e-2 -l2 0.0 -n 40 -d FashionMNIST -b 256 -m LeNet -r "model=LeNet,dataset=FashionMNIST,mid,id=1" --milestones 10 20 --lr_gamma 0.25 -reruns 3 -sp 20 -gamma 1e-6 -freq 50 -hi 10 --middle
# lenet + mnist
python3 middle_penalization.py -lr 1e-2 -l2 0.0 -n 30 -d MNIST -b 256 -m LeNet -r "model=LeNet,dataset=MNIST,mid,id=1" --milestones 10 20 --lr_gamma 0.25 -reruns 3 -sp 10 -gamma 1e-4 -freq 50 -hi 10 --middle 
# vgg13 + svhn 
python3 middle_penalization.py -lr 5e-4 -l2 0.0 -n 50 -d SVHN -b 64 -m VGG13 -r "model=VGG13,dataset=SVHN,mid,id=1" --milestones 20 30 --lr_gamma 0.25 -reruns 3 -sp 0 -gamma 1e-6 -freq 50 -hi 5 --middle