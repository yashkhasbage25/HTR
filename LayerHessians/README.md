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

