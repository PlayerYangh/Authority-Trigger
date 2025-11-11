"""
Utility Classes for Training and Evaluation

Contains helper classes for metric tracking, such as AverageMeter and Accuracy,
used across different training scripts.
"""

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def __str__(self): 
        return f'{self.avg:.6f}'

class Accuracy(object):
    """Computes classification accuracy"""
    def __init__(self):
        self.correct, self.count = 0, 0
    def update(self, output, label):
        preds = output.data.argmax(dim=1)
        self.correct += preds.eq(label.data).sum().item()
        self.count += output.size(0)
    @property
    def accuracy(self):
        return self.correct / self.count if self.count > 0 else 0.0
    def __str__(self):
        return f'{self.accuracy * 100:.2f}%'