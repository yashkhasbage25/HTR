# Lenet + MNIST
# sgd
# 1.00 +- 0.05
python3 no_penalization.py -lr 1e-2 -n 30 -d MNIST -b 256 -m LeNet -r "model=LeNet,dataset=MNIST,no,id=1" --milestones 10 20 --lr_gamma 0.25 -reruns 3
# sgd + l2
# 0.92 +- 0.03
python3 l2_penalization.py -lr 1e-2 -l2 1e-4 -n 30 -d MNIST -b 256 -m LeNet -r "model=LeNet,dataset=MNIST,l2,id=1" --milestones 10 20 --lr_gamma 0.25 -reruns 3
# sgd + htr
# 0.86 +- 0.09
python3 ht_penalization.py -lr 1e-2 -l2 0.0 -n 30 -d MNIST -b 256 -m LeNet -r "model=LeNet,dataset=MNIST,ht,id=1" --milestones 10 20 --lr_gamma 0.25 -reruns 3 -sp 10 -gamma 1e-3 -freq 50 -hi 10
# sgd + htr + l2
python3 combined_penalization.py -lr 1e-2 -l2 1e-4 -n 30 -d MNIST -b 256 -m LeNet -r "model=LeNet,dataset=MNIST,combined,id=1" --milestones 10 20 --lr_gamma 0.25 -reruns 3 -sp 10 -gamma 1e-3 -freq 50 -hi 10

# lenet + fmnist
# sgd 
# 5.43 +- 0.10
python3 no_penalization.py -lr 1e-2 -n 30 -d FashionMNIST -b 256 -m LeNet -r "model=LeNet,dataset=FashionMNIST,no,id=1" --milestones 10 20 --lr_gamma 0.25 -reruns 3
# sgd + l2
# 5.41 +- 0.16
python3 l2_penalization.py -lr 1e-2 -l2 1e-4 -n 30 -d FashionMNIST -b 256 -m LeNet -r "model=LeNet,dataset=FashionMNIST,l2,id=1" --milestones 10 20 --lr_gamma 0.25 -reruns 3
# sgd + htr
# 4.78 +- 0.15
python3 ht_penalization.py -lr 1e-2 -l2 0.0 -n 30 -d FashionMNIST -b 256 -m LeNet -r "model=LeNet,dataset=FashionMNIST,ht,id=1" --milestones 10 20 --lr_gamma 0.25 -reruns 3 -sp 10 -gamma 1e-3 -freq 50 -hi 10
# sgd + htr + l2
python3 combined_penalization.py -lr 1e-2 -l2 1e-5 -n 30 -d FashionMNIST -b 256 -m LeNet -r "model=LeNet,dataset=FashionMNIST,combined,id=1" --milestones 10 20 --lr_gamma 0.25 -reruns 3 -sp 10 -gamma 1e-3 -freq 50 -hi 10

# lenet + cifar10
# sgd
# 30.44 +- 1.68
python3 no_penalization.py -lr 1e-2 -n 50 -d CIFAR10 -b 256 -m LeNet -r "model=LeNet,dataset=CIFAR10,no,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3
# sgd + l2 
# 30.38 +- 1.75
python3 l2_penalization.py -lr 1e-2 -l2 1e-4 -n 50 -d CIFAR10 -b 256 -m LeNet -r "model=LeNet,dataset=CIFAR10,l2,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3
# sgd + htr
# 29.20 +- 0.38
python3 ht_penalization.py -lr 1e-2 -l2 0.0 -n 50 -d CIFAR10 -b 256 -m LeNet -r "model=LeNet,dataset=CIFAR10,ht,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3 -sp 20 -gamma 1e-3 -freq 50 -hi 10
# sgd + l2 + htr
python combined_penalization.py -lr 1e-2 -l2 1e-4 -n 50 -d CIFAR10 -b 256 -m LeNet -r "model=LeNet,dataset=CIFAR10,combined,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3 -sp 20 -gamma 1e-3 -freq 50 -hi 10

# lenet + svhn
# sgd
python3 no_penalization.py -lr 1e-1 -n 50 -d SVHN -b 256 -m LeNet -r "model=LeNet,dataset=SVHN,no,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3
# sgd + l2
python3 l2_penalization.py -lr 1e-1 -l2 1e-4 -n 50 -d SVHN -b 256 -m LeNet -r "model=LeNet,dataset=SVHN,l2,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3
# sgd + htr
python3 ht_penalization.py -lr 1e-1 -l2 0.0 -n 50 -d SVHN -b 256 -m LeNet -r "model=LeNet,dataset=SVHN,ht,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3 -sp 0 -gamma 1e-3 -freq 50 -hi 10
# sgd + htr + l2
python3 combined_penalization.py -lr 1e-1 -l2 1e-4 -n 50 -d SVHN -b 256 -m LeNet -r "model=LeNet,dataset=SVHN,combined,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3 -sp 20 -gamma 1e-3 -freq 50 -hi 10

# vgg11 + fmnist
# sgd
python3 no_penalization.py -lr 1e-4 -n 40 -d FashionMNIST -b 64 -m VGG11 -r "model=VGG11,dataset=FashionMNIST,no,id=1" --milestones 10 25 --lr_gamma 0.1 -reruns 3
# sgd + l2
python3 l2_penalization.py -lr 1e-4 -l2 1e-6 -n 40 -d FashionMNIST -b 64 -m VGG11 -r "model=VGG11,dataset=FashionMNIST,l2,id=1" --milestones 10 25 --lr_gamma 0.1 -reruns 3
# sgd + htr 
python3 ht_penalization.py -lr 1e-4 -l2 0.0 -n 40 -d FashionMNIST -b 64 -m VGG11 -r "model=VGG11,dataset=FashionMNIST,ht,id=1" --milestones 10 25 --lr_gamma 0.1 -reruns 3 -sp 10 -gamma 1e-3 -freq 50 -hi 10
# sgd + l2 + htr
python3 combined_penalization.py -lr 1e-4 -l2 1e-6 -n 40 -d FashionMNIST -b 64 -m VGG11 -r "model=VGG11,dataset=FashionMNIST,combined,id=1" --milestones 10 25 --lr_gamma 0.1 -reruns 3 -sp 10 -gamma 1e-3 -freq 50 -hi 10

# vgg11_bn + cifar100
# sgd
python3 no_penalization.py -lr 1e-2 -n 50 -d CIFAR100 -b 64 -m VGG11_bn -r "model=VGG11_bn,dataset=CIFAR100,no,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3
# sgd + l2
python3 l2_penalization.py -lr 1e-2 -l2 1e-5 -n 50 -d CIFAR100 -b 64 -m VGG11_bn -r "model=VGG11_bn,dataset=CIFAR100,l2,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3
# sgd + htr
python3 ht_penalization.py -lr 1e-2 -l2 0.0 -n 50 -d CIFAR100 -b 64 -m VGG11_bn -r "model=VGG11_bn,dataset=CIFAR100,ht,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3 -sp 20 -gamma 1e-6 -freq 50 -hi 10
# sgd + l2 + htr
python3 combined_penalization.py -lr 1e-2 -l2 1e-5 -n 50 -d CIFAR100 -b 64 -m VGG11_bn -r "model=VGG11_bn,dataset=CIFAR100,combined,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3 -sp 20 -gamma 1e-6 -freq 50 -hi 10

# resnet + cifar10
# sgd 
python3 no_penalization.py -lr 1e-2 -n 40 -d CIFAR10 -b 64 -m ResNet18 -r "model=ResNet18,dataset=CIFAR10,no,id=1" --milestones 10 25 --lr_gamma 0.25 -reruns 3
# sgd + l2
python3 l2_penalization.py -lr 1e-2 -l2 1e-4 -n 40 -d CIFAR10 -b 16 -m ResNet18 -r "model=ResNet18,dataset=CIFAR10,l2,id=1" --milestones 10 25 --lr_gamma 0.25 -reruns 3
# sgd + htr
python3 ht_penalization.py -lr 1e-2 -l2 0.0 -n 40 -d CIFAR10 -b 16 -m ResNet18 -r "model=ResNet18,dataset=CIFAR10,ht,id=1" --milestones 10 25 --lr_gamma 0.25 -reruns 3 -sp 0 -gamma 1e-4 -freq 50 -hi 5
# sgd + htr + l2
python3 combined_penalization.py -lr 1e-2 -l2 1e-4 -n 40 -d CIFAR10 -b 16 -m ResNet18 -r "model=ResNet18,dataset=CIFAR10,combined,id=1" --milestones 10 25 --lr_gamma 0.25 -reruns 3 -sp 0 -gamma 1e-4 -freq 50 -hi 5

# vgg11 + cifar10
# sgd 
python3 no_penalization.py -lr 1e-2 -n 50 -d CIFAR10 -b 64 -m VGG11 -r "model=VGG11,dataset=CIFAR10,no,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3
# sgd + l2
python3 l2_penalization.py -lr 1e-2 -l2 1e-4 -n 50 -d CIFAR10 -b 64 -m VGG11 -r "model=VGG11,dataset=CIFAR10,l2,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3
# sgd + htr
python3 ht_penalization.py -lr 1e-2 -l2 0.0 -n 50 -d CIFAR10 -b 64 -m VGG11 -r "model=VGG11,dataset=CIFAR10,ht,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3 -sp 0 -gamma 1e-4 -freq 50 -hi 5
# sgd + htr + l2
python3 combined_penalization.py -lr 1e-2 -l2 1e-4 -n 50 -d CIFAR10 -b 64 -m VGG11 -r "model=VGG11,dataset=CIFAR10,combined,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3 -sp 0 -gamma 1e-4 -freq 50 -hi 5

# vgg11bn + fash
# sgd
python3 no_penalization.py -lr 1e-2 -n 40 -d CIFAR10 -b 64 -m VGG11_bn -r "model=VGG11_bn,dataset=FashionMNIST,no,id=1" --milestones 20 30 --lr_gamma 0.25 -reruns 3
# sgd + l2
python3 l2_penalization.py -lr 1e-2 -l2 1e-4 -n 40 -d CIFAR10 -b 64 -m VGG11_bn -r "model=VGG11_bn,dataset=FashionMNIST,l2,id=1" --milestones 20 30 --lr_gamma 0.25 -reruns 3

#####################################
# middle layer
#####################################
# vgg11bn + cifar100
python3 middle_penalization.py -lr 1e-2 -l2 0.0 -n 50 -d CIFAR100 -b 64 -m VGG11_bn -r "model=VGG11_bn,dataset=CIFAR100,mid,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3 -sp 20 -gamma 1e-6 -freq 50 -hi 10 --middle
# resnet18 + cifar10
python3 middle_penalization.py -lr 1e-2 -l2 0.0 -n 40 -d CIFAR10 -b 64 -m ResNet18 -r "model=ResNet18,dataset=CIFAR10,mid,id=1" --milestones 20 30 --lr_gamma 0.25 -reruns 3 -sp 10 -gamma 1e-6 -freq 50 -hi 3 --middle
# lenet + fash
python3 middle_penalization.py -lr 1e-2 -l2 0.0 -n 40 -d FashionMNIST -b 256 -m LeNet -r "model=LeNet,dataset=FashionMNIST,mid,id=1" --milestones 10 20 --lr_gamma 0.25 -reruns 3 -sp 20 -gamma 1e-3 -freq 50 -hi 10 --middle
# lenet + mnist
python3 middle_penalization.py -lr 1e-2 -l2 0.0 -n 30 -d MNIST -b 256 -m LeNet -r "model=LeNet,dataset=MNIST,mid,id=1" --milestones 10 20 --lr_gamma 0.25 -reruns 3 -sp 0 -gamma 1e-3 -freq 50 -hi 10 --middle 
# lenet + svhn
python3 middle_penalization.py -lr 1e-1 -l2 0.0 -n 50 -d SVHN -b 256 -m LeNet -r "model=LeNet,dataset=SVHN,mid,id=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3 -sp 0 -gamma 1e-7 -freq 50 -hi 10 --middle 
# vgg11+cifar10
python3 middle_penalization.py -lr 1e-2 -l2 0.0 -n 50 -d CIFAR10 -b 64 -m VGG11 -r "model=VGG11,dataset=CIFAR10,mid=1" --milestones 20 35 --lr_gamma 0.25 -reruns 3 -sp 0 -gamma 1e-6 -freq 50 -hi 5 --middle 