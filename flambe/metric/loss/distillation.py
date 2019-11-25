from typing import Tuple, Callable, Optional

import torch
import torch.nn.functional as F

from flambe.metric import Metric


class DistillationLoss(Metric):
    """Distillation loss module."""

    def __init__(self,
                 alpha_kl: float = 0.5,
                 temperature: float = 1,
                 objective_fn: Optional[Callable] = None) -> None:
        """Initialize loss function.

        Parameters
        ----------
        alpha_kl: float
            The weight of the KL term, should be between 0 and 1.
            Defaults to 0.5.
        temperature: float
            Temperature applied to the student and teacher
            distributions. Defaults to 1.
        objective_fn: Callable, Optional
            The loss function to use over the logits for the student.
            By default, uses Pytorch's ``CrossEntropyLoss``.

        """
        self.objective_fn = objective_fn or torch.nn.CrossEntropyLoss(reduction=None)
        self.alpha_kl = alpha_kl
        self.temp = temperature

    def compute(self,
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the per-sample loss.

        Parameters
        ----------
        student_logits: torch.Tensor
            Pre-softmax student logits, with the target dimension
            as last dimension. The first N-1 dimensions are flattened
            if not already.
        teacher_predictions: torch.Tensor
            Pre-softmax teacher logits with the target dimension
            as last dimension. The first N-1 dimensions are flattened
            if not already.
        targets: torch.Tensor
            True targets

        Returns
        -------
        torch.Tensor
            The total loss, interpolated between the student and KL
        torch.Tensor
            The student loss over the given objective function
        torch.Tensor)
            The KL loss between the student and the teacher

        """
        # Flatten if not already
        student_logits = student_logits.view(-1, student_logits.size(-1))
        teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
        targets = targets.view(-1, student_logits.size(-1))

        # Calculate the softmaxes from predictions
        student_pred = F.log_softmax(student_logits / self.temp, dim=1)
        teacher_pred = F.softmax(teacher_logits / self.temp, dim=1)

        # Compute the losses
        student_loss = self.objective_fn(student_logits, targets)
        kl_loss = F.kl_div(student_pred, teacher_pred, reduction=None)

        loss = (1 - self.alpha_kl) * student_loss + self.alpha_kl * (self.temp ** 2) * kl_loss
        return loss, student_loss, kl_loss
