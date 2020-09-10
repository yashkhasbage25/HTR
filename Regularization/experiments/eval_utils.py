import torch

def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res

def evaluate_model(model, criterion, dataloader, device, dataset_size, num_classes):

    model.eval()
    all_outputs = torch.zeros(0, num_classes).to(device)
    all_targets = torch.zeros(0).to(device)

    with torch.no_grad():
        for batch, truth in dataloader:

            output = model(batch.to(device))
            _, preds = torch.max(output, 1)
            all_outputs = torch.cat((all_outputs, output), dim=0)
            all_targets = torch.cat((all_targets, truth.float().to(device)), dim=0)

    acc = accuracy(all_outputs, all_targets, topk=(1, 5,))

    return acc
