# LayerHessians

Directory strcture:
```bash
.
./datasets/  # to store datasets
./experiments # experimentation code and results
./hessian # code for core hessian operations
./models # code for networks
./utils # utils
```
libraries used:

pytorch
torchvision
seaborn 
pandas
matplotlib
numpy
scipy
sklearn

Ubuntu-16.04
python3 only

### Matching full and layerwise Hessian
```bash

# this will generate eigenspectrums throughout the training
python3 training_eigenspectrum.py -lr 1e-2 -l2 0 -n 50 -o sgd -d MNIST -b 256 -m LeNet --cuda 2 -r "model=LeNet,dataset=MNIST,te,run1"
python3 training_eigenspectrum.py -lr 5e-3 -l2 5e-5 -n 50 -o sgd -d MNIST -b 128 -m VGG11_bn --cuda 2 -r "model=VGG11_bn,dataset=MNIST,te,run1"
python3 training_eigenspectrum.py -lr 1e-2 -l2 1e-4 -n 50 -o sgd -d MNIST -b 128 -m ResNet18 --cuda 2 -r "model=ResNet18,dataset=MNIST,te,run1"

python3 training_eigenspectrum.py -lr 1e-2 -l2 0 -n 50 -o sgd -d FashionMNIST -b 256 -m LeNet --cuda 2 -r "model=LeNet,dataset=FashionMNIST,te,run1"
python3 training_eigenspectrum.py -lr 1e-3 -l2 0 -n 50 -o sgd -d FashionMNIST -b 256 -m VGG11_bn --cuda 2 -r "model=VGG11_bn,dataset=FashionMNIST,te,run1"
python3 training_eigenspectrum.py -lr 5e-3 -l2 5e-5 -n 50 -o sgd -d FashionMNIST -b 128 -m ResNet18 --cuda 2 -r "model=ResNet18,dataset=FashionMNIST,te,run1"

python3 training_eigenspectrum.py -lr 1e-2 -l2 0 -n 50 -o sgd -d SVHN -b 128 -m LeNet --cuda 2 -r "model=LeNet,dataset=SVHN,te,run1"
python3 training_eigenspectrum.py -lr 1e-2 -l2 1e-4 -n 50 -o sgd -d SVHN -b 128 -m VGG11_bn --cuda 2 -r "model=VGG11_bn,dataset=SVHN,te,run1"
python3 training_eigenspectrum.py -lr 1e-3 -l2 0 -n 50 -o sgd -d SVHN -b 256 -m ResNet18 --cuda 2 -r "model=ResNet18,dataset=SVHN,te,run1"

python3 training_eigenspectrum.py -lr 1e-2 -l2 1e-4 -n 50 -o sgd -d CIFAR10 -b 256 -m LeNet --cuda 2 -r "model=LeNet,dataset=CIFAR10,te,run1"
python3 training_eigenspectrum.py -lr 1e-2 -l2 0 -n 50 -o sgd -d CIFAR10 -b 256 -m VGG11_bn --cuda 2 -r "model=VGG11_bn,dataset=CIFAR10,te,run1"
python3 training_eigenspectrum.py -lr 1e-2 -l2 1e-4 -n 50 -o sgd -d CIFAR10 -b 128 -m ResNet18 --cuda 2 -r "model=ResNet18,dataset=CIFAR10,te,run1"

# this will plot the computed eigenspectrums 
python3 plot_training_eigenspectrum.py -r "model=LeNet,dataset=MNIST,te,run1" -m LeNet -d MNIST
python3 plot_training_eigenspectrum.py -r "model=LeNet,dataset=FashionMNIST,te,run1" -m LeNet -d FashionMNIST
python3 plot_training_eigenspectrum.py -r "model=LeNet,dataset=SVHN,te,run1" -m LeNet -d SVHN
python3 plot_training_eigenspectrum.py -r "model=LeNet,dataset=CIFAR10,te,run1" -m LeNet -d CIFAR10

python3 plot_training_eigenspectrum.py -r "model=VGG11_bn,dataset=MNIST,te,run1" -m VGG11_bn -d MNIST
python3 plot_training_eigenspectrum.py -r "model=VGG11_bn,dataset=FashionMNIST,te,run1" -m VGG11_bn -d FashionMNIST
python3 plot_training_eigenspectrum.py -r "model=VGG11_bn,dataset=SVHN,te,run1" -m VGG11_bn -d SVHN
python3 plot_training_eigenspectrum.py -r "model=VGG11_bn,dataset=CIFAR10,te,run1" -m VGG11_bn -d CIFAR10

python3 plot_training_eigenspectrum.py -r "model=ResNet18,dataset=MNIST,te,run1" -m ResNet18 -d MNIST 
python3 plot_training_eigenspectrum.py -r "model=ResNet18,dataset=FashionMNIST,te,run1" -m ResNet18 -d FashionMNIST 
python3 plot_training_eigenspectrum.py -r "model=ResNet18,dataset=SVHN,te,run1" -m ResNet18 -d SVHN 
python3 plot_training_eigenspectrum.py -r "model=ResNet18,dataset=CIFAR10,te,run1" -m ResNet18 -d CIFAR10 

# this will use the above generated eigenspectrums to create a single plot
python3 match_distros.py
```

Apart from several intermediate figures and files, these commands will result in two figures match_distros_peak.png (for comparison at peak epoch) and match_distros_wasserstein.png (for comparison througout the training). 

### Reproducing HTR results


* LeNet + CIFAR10
```bash
# sgd
# 30.44 +- 1.68
python3 no_penalization.py -lr 1e-2 -n 400 -d CIFAR10 -b 256 -m LeNet -r "model=LeNet,dataset=CIFAR10,no,id=1" --milestones 200 300 --lr_gamma 0.1 -reruns 5
# sgd + l2 
# 30.38 +- 1.75
python3 l2_penalization.py -lr 1e-2 -l2 1e-6 -n 400 -d CIFAR10 -b 256 -m LeNet -r "model=LeNet,dataset=CIFAR10,l2,id=1" --milestones 200 300 --lr_gamma 0.1 -reruns 5
# sgd + htr
# 29.20 +- 0.38
python3 ht_penalization.py -lr 1e-2 -l2 0 -n 500 -d CIFAR10 -b 256 -m LeNet -r "model=LeNet,dataset=CIFAR10,ht,id=1" --milestones 300 400 --lr_gamma 0.1 -reruns 5 -sp 15 -gamma 1e-3 -freq 50 -hi 10
```

* LeNet + MNIST
```bash
# sgd
# 1.00 +- 0.05
python3 no_penalization.py -lr 1e-2 -n 400 -d MNIST -b 256 -m LeNet -r "model=LeNet,dataset=MNIST,no,id=1" --milestones 200 300 --lr_gamma 0.1 -reruns 5
# sgd + l2
# 0.92 +- 0.03
python3 l2_penalization.py -lr 1e-2 -n 500 -d MNIST -b 256 -m LeNet -r "model=LeNet,dataset=MNIST,l2,id=1" --milestones 300 400 --lr_gamma 0.1 -reruns 5
# sgd + htr
# 0.86 +- 0.09
python3 ht_penalization.py -lr 1e-2 -l2 0 -n 400 -d MNIST -b 256 -m LeNet -r "model=LeNet,dataset=MNIST,ht,id=1" --milestones 200 300 --lr_gamma 0.1 -reruns 5 -sp 15 -gamma 1e-3 -freq 50 -hi 10
```

* LeNet + FashionMNIST
```bash
# sgd 
# 5.43 +- 0.10
python3 no_penalization.py -lr 1e-2 -n 400 -d FashionMNIST -b 256 -m LeNet -r "model=LeNet,dataset=FashionMNIST,no,id=1" --milestones 200 300 --lr_gamma 0.1 -reruns 5
# sgd + l2
# 5.41 +- 0.16
python3 l2_penalization.py -lr 1e-2 -l2 1e-5 -n 400 -d FashionMNIST -b 256 -m LeNet -r "model=LeNet,dataset=FashionMNIST,l2,id=1" --milestones 200 300 --lr_gamma 0.1 -reruns 5
# sgd + htr
# 4.78 +- 0.15
python3 ht_penalization.py -lr 1e-2 -l2 0 -n 400 -d FashionMNIST -b 256 -m LeNet -r "model=LeNet,dataset=FashionMNIST,ht,id=1" --milestones 250 350 --lr_gamma 0.1 -reruns 5
```

* VGG11-bn + FashionMNIST
```bash
# sgd
# 7.72 +- 0.06
python3 no_penalization.py -lr 5e-3 -n 100 -d FashionMNIST -b 64 -m VGG11_bn -r "model=VGG11_bn,dataset=FashionMNIST,no,id=1" --milestones 50 75 --lr_gamma 0.1 -reruns 5
# sgd + l2 
# 7.69 +- 0.15
python3 l2_penalization.py -lr 5e-3 -l2 1e-4 -n 100 -d FashionMNIST -b 64 -m VGG11_bn -r "model=VGG11_bn,dataset=FashionMNIST,l2,id=1" --milestones 50 75 --lr_gamma 0.1 -reruns 5
# sgd + htr 
# 7.67 +- 0.07
python3 ht_penalization.py -lr 5e-3 -l2 0 -n 100 -d FashionMNIST -b 64 -m VGG11_bn -r "model=VGG11_bn,dataset=FashionMNIST,ht,id=1" --milestones 50 75 --lr_gamma 0.1 -reruns 5 
```

* VGG11 + CIFAR10
```bash
# sgd 
# 24.27 +- 1.49
python3 no_penalization.py -lr 5e-3 -n 100 -d CIFAR10 -b 64 -m VGG11 -r "model=VGG11,dataset=CIFAR10,no,id=1" --milestones 60 80 --lr_gamma 0.2 -reruns 5
# sgd + l2
# 23.88 +- 0.59
python3 l2_penalization.py -lr 5e-3 -l2 1e-6 -n 100 -d CIFAR10 -b 64 -m VGG11 -r "model=VGG11,dataset=CIFAR10,no,l2,id=1" --milestones 50 75 --lr_gamma 0.1 -reruns 5
# sgd + htr 
# 21.95 +- 0.20
python3 ht_penalization.py -lr 1e-2 -l2 0 -n 100 -d CIFAR10 -b 64 -m VGG11 -r "model=VGG11,dataset=CIFAR10,ht,id=1" --milestones 50 75 --lr_gamma 0.1 -reruns 3 -sp 30 -gamma 1e-8 -freq 50 -hi 10 
```

* ResNet18 + CIFAR10
```bash
# sgd
# 19.38 +- 0.5
python3 no_penalization.py -lr 1e-2 -n 100 -d CIFAR10 -b 16 -m ResNet18 -r "model=ResNet18,dataset=CIFAR10,no,id=1" --milestones 50 75 --lr_gamma 0.1 -reruns 5
# sgd + l2 
# 18.64 +- 0.63
python3 l2_penalization.py -lr 1e-2 -l2 1e-7 -n 100 -d CIFAR10 -b 16 -m ResNet18 -r "model=ResNet18,dataset=CIFAR10,l2,id=1" --milestones 50 75 --lr_gamma 0.1 -reruns 5
# sgd + htr 
# 18.36 +- 0.27
python3 ht_penalization.py -lr 1e-2 -l2 0 -n 40 -d CIFAR10 -b 8 -m ResNet18 -r "model=ResNet18,dataset=CIFAr10,ht,id=1" --milestones 1000 --lr_gamma 0.1 -reruns 10 -sp 8 -gamma 1e-5 -freq 50 -hi 10
```

* VGG11_bn + CIFAR100
```bash
# sgd
# 55.28 +- 0.34 
python3 no_penalization.py -lr 1r-2 -n 100 -d CIFAR100 -b 128 -m VGG11_bn -r "model=VGG11_bn,dataset=CIFAR100,no,id=1" --milestones 1000 --lr_gamma 0.1 -reruns 3
# sgd + l2 
# 54.96 +- 0.46
python3 l2_penalization.py -lr 1e-2 -l2 1e-10 -n 100 -d CIFAR100 -b 128 -m VGG11_bn -r "model=VGG11_bn,dataset=CIFAR100,l2,id=1" --milestones 1000 --lr_gamma 0.1 -reruns 3 
# sgd + htr
# 54.92 +- 0.35
python3 ht_penalization.py -lr 1e-2 -l2 0 -n 100 -d CIFAR100 -b 128 -m VGG11_bn -r "model=VGG11_bn,dataset=CIFAR100,ht,id=1" --milestones 1000 --lr_gamma 0.1 -reruns 3 -sp 30 -gamma 1e-6 -freq 50 -hi 5
```

Note that, HTR and L2 gain better test accuracy even at lower train accuracy, when comaring with Vanilla-SGD. The differences in train parameters are mainly to get same train accuarcy in all three cases. 

Use plot_penalization.py to print the training stats along with accuracies across reruns. 

```bash

# LeNet + MNIST
python3 plot_penalization.py -r "model=LeNet,dataset=MNIST,no,id=1"
python3 plot_penalization.py -r "model=LeNet,dataset=MNIST,l2,id=1"
python3 plot_penalization.py -r "model=LeNet,dataset=MNIST,ht,id=1"

# LeNet + FashionMNIST
python3 plot_penalization.py -r "model=LeNet,dataset=FashionMNIST,no,id=1" 
python3 plot_penalization.py -r "model=LeNet,dataset=FashionMNIST,l2,id=1"
python3 plot_penalization.py -r "model=LeNet,dataset=FashionMNIST,ht,id=1"

# LeNet + SVHN
python3 plot_penalization.py -r "model=LeNet,dataset=SVHN,no,id=1"
python3 plot_penalization.py -r "model=LeNet,dataset=SVHN,l2,id=1"
python3 plot_penalizatoin.py -r "model=LeNet,dataset=SVHN,ht,id=1"

# LeNet + CIFAR10
python3 plot_penalization.py -r "model=LeNet,dataset=CIFAR10,no,id=1"
python3 plot_penalization.py -r "model=LeNet,dataset=CIFAR10,l2,id=1"
python3 plot_penalization.py -r "model=LeNet,dataset=CIFAR10,ht,id=1"

# VGG11 + CIFAR10
python3 plot_penalization.py -r "model=VGG11,dataset=CIFAR10,no,id=1"
python3 plot_penalization.py -r "model=VGG11,dataset=CIFAR10,l2,id=1"
python3 plot_penalization.py -r "model=VGG11,dataset=CiFAR10,ht,id=1"

# VGG11_bn + FashionMNIST
python3 plot_penalization.py -r "model=VGG11_bn,dataset=FashionMNIST,no,id=1"
python3 plot_penalization.py -r "model=VGG11_bn,dataset=FashionMNIST,l2,id=1"
python3 plot_penalization.py -r "model=VGG11_bn,dataset=FashionMNIST,ht,id=1"

# ResNet18 + CIFAR10
python3 plot_penalization.py -r "model=ResNet18,dataset=CIFAR10,no,id=1"
python3 plot_penalization.py -r "model=ResNet18,dataset=CIFAR10,l2,id=1"
python3 plot_penalization.py -r "model=ResNet18,dataset=CIFAR10,ht,id=1"

# VGG11_bn + CIFAR100
python3 plot_penalization.py -r "model=VGG11_bn,dataset=CIFAR100,no,id=1"
python3 plot_penalization.py -r "model=VGG11_bn,dataset=CIFAR100,l2,id=1"
python3 plot_penalization.py -r "model=VGG11_bn,dataset=CIFAR100,ht,id=1"

```
