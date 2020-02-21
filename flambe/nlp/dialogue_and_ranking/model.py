from typing import Optional, Tuple

import torch
from torch import Tensor

from flambe.nn import Embedder, Module


class RankingModel(Module):
    """Implements a standard classifier.

    The classifier is composed of an encoder module, followed by
    a fully connected output layer, with a dropout layer in between.

    Attributes
    ----------
    embedder_1: Embedder
        The embedder layer for the first sentence
    embedder_2: Optional[Embedder]
        The embedder layer for the second sentence. If not specified, embedder_1 is shared for both text fields.
    output_module : Optional[Module]
        The output layer which can be used instead of the inner-batch dot product to compute a score.
        Needs to return a score for each text in text_2, so that output_dim will be batch x batch.
        If not specified, defaults to inner-batch dot products.
    """

    def __init__(self,
                 embedder_1: Embedder,
                 embedder_2: Optional[Embedder] = None,
                 output_module: Optional[Module] = None,
                 ) -> None:
        """Initialize the TextClassifier model.

        Parameters
        ----------
        embedder_1: Embedder
            The embedder layer for the first sentence
        embedder_2: Optional[Embedder]
            The embedder layer for the second sentence. If not specified, embedder_1 is shared for both text fields.
        output_module : Optional[Module]
            The output layer which can be used instead of the inner-batch dot product to compute a score.
            Needs to return a score for each combination of texts in text1 and text2, so that output_dim
            will be batch x batch. If not specified, defaults to inner-batch dot products.

        """
        super().__init__()

        self.embedder_1 = embedder_1
        self.embedder_2 = embedder_2
        self.output_module = output_module

    def _forward_batch_negatives(self, text_1: Tensor, text_2: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode context and response vectors.
        Then either:
            - use custom output module (e.g., polyencoder), if output_module specified
            - compute dot product between each pair of encodings
        Returns score and cross-entropy labels

        Arguments
        ----------
        text_1: Tensor
            The first text (oftentimes defining the context) of shape
            (batch x seq_len)
        text_2: Tensor
            The second text (oftentimes defining the responses) of shape
            (batch x seq_len)

        Returns
        ----------
        Tuple[Tensor, Tensor]
            - Either:
                Text 2 ranked for text1: batch_dim x batch_dim
                The output from output_module
            - The labels: a longtensor with the correct response indices
        """
        # 1st: embed. Produces an encoding, possibly unpooled.
        # Both are (batch_dim x seq_len x embedding_dim)
        text_1_encodings = self.embedder_1(text_1)
        text_2_encodings = self.embedder_2(text_2) \
            if self.embedder_2 is not None \
            else self.embedder_1(text_2)
        if self.output_module is not None:
            # 2a: (optional) custom output computation. Example usage: PolyEncoder
            scores = self.output_module(text_1, text_2, text_1_encodings, text_2_encodings)
        else:
            # 2b: (default) dot products as inner-batch scores
            scores = text_1_encodings @ text_2_encodings.transpose(0, 1)
        labels = torch.arange(text_1.size(0), device=text_1.device)
        return scores, labels

    def _forward_provided_candidates(self, text_1: Tensor, text_2: Tensor,
                                     labels: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode context and response vectors.
        Then either:
            - use custom output module (e.g., polyencoder),
              if output_module specified
            - compute dot product between each pair of encodings
        Returns score and cross-entropy labels

        Arguments
        ----------
        text_1: Tensor
            The first text (oftentimes defining the context) of shape
            (batch x seq_len)
        text_2: Tensor
            The second text (oftentimes defining the responses) of shape
            (batch x seq_len)
        labels: Tensor
            The labels. Can only be provided in combination with a
            text_2 tensor that contains a list of candidates.

        Returns
        ----------
        Tuple[Tensor, Tensor]
            - Either:
                Text 2 ranked for text1: batch_dim x batch_dim
                The output from output_module
            - The labels: a longtensor with the correct response indices
        """
        # 0th: reshape candidates
        # text 2 is now (batch_dim * num_cands x seq_len)
        batch_dim, num_cands, seq_len = text_2.shape
        text_2_rs = text_2.reshape(batch_dim * num_cands, seq_len)
        # 1st: embed. Produces an encoding, possibly unpooled.
        # text 1 is (batch_dim x seq_len x embedding_dim)
        # text 2 is (batch_dim * num_cands x seq_len x embedding_dim)
        text_1_encodings = self.embedder_1(text_1)
        text_2_encodings = self.embedder_2(text_2_rs) \
            if self.embedder_2 is not None \
            else self.embedder_1(text_2_rs)
        # change text 2 shape to:
        # if no pooling was applied:
        #   (batch_dim x num_cands x seq_len x embedding_dim)
        # if pooling was applied:
        #   (batch_dim x num_cands x embedding_dim)
        text_2_encodings = text_2_encodings.reshape(batch_dim, num_cands, seq_len, -1) \
            if len(text_2_encodings.shape) == 3 \
            else text_2_encodings.reshape(batch_dim, num_cands, -1)
        if self.output_module is not None:
            # 2a: (optional) custom output computation. Example usage: PolyEncoder
            scores = self.output_module(text_1, text_2, text_1_encodings, text_2_encodings)
        else:
            # 2b: (default) dot products as inner-batch scores
            scores = torch.bmm(text_1_encodings.unsqueeze(1),
                               text_2_encodings.transpose(1, 2)).squeeze(1)
        return scores, labels

    def forward(self, text_1: Tensor, text_2: Tensor,
                labels: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Encode context and response vectors.

        Then compute the score either for provided candidates
        or for the batch as candidates.

        The score can be computed by:
            - use custom output module (e.g., polyencoder),
              if output_module specified
            - the dot product between each pair of encodings
        Returns score and cross-entropy labels

        Arguments
        ----------
        text_1: Tensor
            The first text (oftentimes defining the context) of shape
            (batch x seq_len)
        text_2: Tensor
            The second text (oftentimes defining the responses) of shape
            (batch x seq_len) OR (batch x num_candidates x seq_len)
        labels: Optional[Tensor]
            The labels. Can only be provided in combination with a
            text_2 tensor that contains a list of candidates.

        Returns
        ----------
        Tuple[Tensor, Tensor]
            - Either:
                Text 2 ranked for text1: batch_dim x batch_dim
                The output from output_module
            - The labels: a longtensor with the correct response indices
        """
        # 2 cases: candidates and labels provided, or not
        if labels is not None and len(text_2.shape) == 3:
            # 1st case: provided
            return self._forward_provided_candidates(text_1, text_2, labels)
        elif labels is None and len(text_2.shape) == 2:
            # 2nd case: no candidates and no negatives provided
            return self._forward_batch_negatives(text_1, text_2)
        else:
            # this should not happen
            raise ValueError('RankingModel either needs provided candidates'
                             'for each sample, _and_ provided labels,\n'
                             'OR no candidates and no labels.')
