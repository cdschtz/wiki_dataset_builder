import torch
from torch.nn import functional as F


def generate(
        model,
        input_ids=None,
        max_length=None,
        do_sample=None,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_ids=None,
        length_penalty=None,
        num_return_sequences=None,
):
    do_sample = True
    eos_token_ids = [0]

    batch_size = input_ids.shape[0]
    cur_len = input_ids.shape[1]
    vocab_size = 50257

    if num_return_sequences != 1:
        # Expand input to num return sequences
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_return_sequences, cur_len)
        input_ids = input_ids.contiguous().view(
            batch_size * num_return_sequences, cur_len
        )  # (batch_size * num_return_sequences, cur_len)
        effective_batch_size = batch_size * num_return_sequences
    else:
        effective_batch_size = batch_size

    if num_beams > 1:
        output = generate_beam_search(
            model,
            input_ids,
            cur_len,
            max_length,
            do_sample,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            pad_token_id,
            eos_token_ids,
            effective_batch_size,
            length_penalty,
            num_beams,
            vocab_size,
        )
    else:
        return

    if num_return_sequences != 1:
        output = output.view(batch_size, num_return_sequences, -1)
    return output


def generate_beam_search(
        model,
        input_ids,
        cur_len,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size,
        length_penalty,
        num_beams,
        vocab_size,
):
    input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, cur_len)
    input_ids = input_ids.contiguous().view(batch_size * num_beams, cur_len)  # (batch_size * num_beams, cur_len)

    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=False) for _ in range(batch_size)
    ]

    # scores for each sentence in the beam
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

    # cache compute states
    past = None

    # done sentences
    done = [False for _ in range(batch_size)]

    while cur_len < max_length:
        model_inputs = prepare_inputs_for_generation(input_ids, past=past)
        outputs = model(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
        scores = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

        # if model has past, then set the past variable to speed up decoding
        if do_output_past(outputs):
            past = outputs[1]

        # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            for i in range(batch_size * num_beams):
                for previous_token in set(input_ids[i].tolist()):
                    # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                    if scores[i, previous_token] < 0:
                        scores[i, previous_token] *= repetition_penalty
                    else:
                        scores[i, previous_token] /= repetition_penalty

        if do_sample:
            # Temperature (higher temperature => more likely to sample low probability tokens)
            if temperature != 1.0:
                scores = scores / temperature
            # Top-p/top-k filtering
            scores = top_k_top_p_filtering(
                scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
            )  # (batch_size * num_beams, vocab_size)
            # Sample 2 next words for each beam (so we have some spare tokens and match output of greedy beam search)
            next_words = torch.multinomial(F.softmax(scores, dim=-1), num_samples=2)  # (batch_size * num_beams, 2)
            # Compute next scores
            _scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
            _scores = torch.gather(_scores, -1, next_words)  # (batch_size * num_beams, 2)
            next_scores = _scores + beam_scores[:, None].expand_as(_scores)  # (batch_size * num_beams, 2)
            # Match shape of greedy beam search
            next_words = next_words.view(batch_size, 2 * num_beams)  # (batch_size, 2 * num_beams)
            next_scores = next_scores.view(batch_size, 2 * num_beams)  # (batch_size, 2 * num_beams)
        else:
            # do greedy beam search
            scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
            assert scores.size() == (batch_size * num_beams, vocab_size)
            # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            _scores = _scores.view(batch_size, num_beams * vocab_size)  # (batch_size, num_beams * vocab_size)
            next_scores, next_words = torch.topk(_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

        assert next_scores.size() == next_words.size() == (batch_size, 2 * num_beams)

        # next batch beam content
        # list of (batch_size * num_beams) tuple(next hypothesis score, next word, current position in the batch)
        next_batch_beam = []

        # for each sentence
        for batch_ex in range(batch_size):

            # if we are done with this sentence
            done[batch_ex] = done[batch_ex] or generated_hyps[batch_ex].is_done(next_scores[batch_ex].max().item())
            if done[batch_ex]:
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                continue

            # next sentence beam content
            next_sent_beam = []

            # next words for this sentence
            for idx, score in zip(next_words[batch_ex], next_scores[batch_ex]):

                # get beam and word IDs
                beam_id = idx // vocab_size
                word_id = idx % vocab_size

                # end of sentence, or next word
                if word_id.item() in eos_token_ids or cur_len + 1 == max_length:
                    generated_hyps[batch_ex].add(
                        input_ids[batch_ex * num_beams + beam_id, :cur_len].clone(), score.item()
                    )
                else:
                    next_sent_beam.append((score, word_id, batch_ex * num_beams + beam_id))

                # the beam for next step is full
                if len(next_sent_beam) == num_beams:
                    break

            # update next beam content
            assert len(next_sent_beam) == 0 if cur_len + 1 == max_length else num_beams
            if len(next_sent_beam) == 0:
                next_sent_beam = [(0, pad_token_id, 0)] * num_beams  # pad the batch
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_ex + 1)

        # sanity check / prepare next batch
        assert len(next_batch_beam) == batch_size * num_beams
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_words = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])

        # re-order batch
        input_ids = input_ids[beam_idx, :]
        input_ids = torch.cat([input_ids, beam_words.unsqueeze(1)], dim=-1)

        # re-order internal states
        if past:
            reordered_past = []
            for layer_past in past:
                # get the correct batch idx from layer past batch dim
                # batch dim of `past` and `mems` is at 2nd position
                reordered_layer_past = [layer_past[:, i].unsqueeze(1).clone().detach() for i in beam_idx]
                reordered_layer_past = torch.cat(reordered_layer_past, dim=1)
                # check that shape matches
                assert reordered_layer_past.shape == layer_past.shape
                reordered_past.append(reordered_layer_past)
            past = tuple(reordered_past)

        # update current length
        cur_len = cur_len + 1

        # stop when we are done with each sentence
        if all(done):
            break

    tgt_len = input_ids.new(batch_size)
    best = []

    for i, hypotheses in enumerate(generated_hyps):
        best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
        tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
        best.append(best_hyp)

    # generate target batch
    decoded = input_ids.new(batch_size, tgt_len.max().item()).fill_(pad_token_id)
    for i, hypo in enumerate(best):
        decoded[i, : tgt_len[i] - 1] = hypo
        decoded[i, tgt_len[i] - 1] = eos_token_ids[0]

    return decoded


class BeamHypotheses(object):
    def __init__(self, n_hyp, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_length ** self.length_penalty


def prepare_inputs_for_generation(input_ids, **kwargs):
    return {"input_ids": input_ids}


def do_output_past(outputs):
    has_output_past = True
    has_mem_len = False

    if has_output_past and not has_mem_len and len(outputs) > 1:
        return True
    # elif has_mem_len and self.config.mem_len > 0 and len(outputs) > 1:
    #     return True

    return False


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits