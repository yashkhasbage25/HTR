#! /usr/bin/env python3

import torch.nn.functional as F

def svhn_ce_loss(output, truth):

    digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits = output
    digits_labels = truth

    # length_cross_entropy = F.cross_entropy(length_logits, length_labels)
    # print(digit1_logits.shape, digits_labels)
    digit1_cross_entropy = F.cross_entropy(digit1_logits, digits_labels[0])
    digit2_cross_entropy = F.cross_entropy(digit2_logits, digits_labels[1])
    digit3_cross_entropy = F.cross_entropy(digit3_logits, digits_labels[2])
    digit4_cross_entropy = F.cross_entropy(digit4_logits, digits_labels[3])
    digit5_cross_entropy = F.cross_entropy(digit5_logits, digits_labels[4])
    
    loss = digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy + digit5_cross_entropy
    
    return loss