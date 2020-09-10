model_choices = ['VGG%d' % x for x in [11, 13, 16, 19]] \
    + ['VGG%d_bn' % x for x in [11, 13, 16, 19]] \
    + ['ResNet18', 'DenseNet3_40'] \
    + ['LeNet'] 