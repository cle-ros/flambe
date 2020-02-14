import torch

from flambe.metric.metric import Metric


class Rank(Metric):

    def compute(self, pred: torch.Tensor, target: torch.Tensor) \
            -> torch.Tensor:
        """Computes the rank of the correct response

        Parameters
        ----------
        pred: Tensor
            input logits of shape (B x N)
        target: LongTensor
            target tensor of shape (B) or (B x N)

        Returns
        -------
        rank: torch.Tensor
            average rank of the correct response

        """
        # If 2-dimensional, select the highest score in each row
        if len(target.size()) == 2:
            target = target.argmax(dim=1)

        ranked_scores = torch.argsort(pred, dim=1)
        rank = (ranked_scores == target).to(torch.uint8).argmax(dim=1)
        # offsetting by 1 for natural counting
        return pred.shape[1] - rank.float().mean() + 1
