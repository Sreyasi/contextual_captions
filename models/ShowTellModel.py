# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .CaptionModel import CaptionModel
import pdb

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.rnn_size = config['rnn_hidden_dim']
        self.W_h = nn.Linear(self.rnn_size*2, self.rnn_size, bias=False)
        self.W_s = nn.Linear(self.rnn_size, self.rnn_size, bias=False)
        self.b_attn = nn.Parameter(torch.zeros(self.rnn_size))
        self.v = nn.Linear(self.rnn_size, 1, bias=False)
#         self.v = nn.Linear(self.rnn_size, 1, bias=True)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.W_h.weight.data.uniform_(-initrange, initrange)
        self.W_h.weight.data.uniform_(-initrange, initrange)
        self.v.weight.data.uniform_(-initrange, initrange)

    def forward(self, encoder_states, decoder_state):
        # print(encoder_states.size())
        # print(decoder_state.size())
        encoder_ctx = self.W_h(encoder_states)
        # print(encoder_ctx.size())
        decoder_ctx = self.W_s(decoder_state)
        decoder_ctx = decoder_ctx.expand_as(encoder_ctx)
        # print(decoder_ctx.size())
        # print(self.b_attn.size())
        attn_energy = self.v(encoder_ctx + decoder_ctx + self.b_attn.expand_as(decoder_ctx)).squeeze(2)
#         attn_energy = self.v(encoder_ctx + decoder_ctx).squeeze(2) #TODO
        # print(attn_energy.size())
        attn_weights = F.softmax(attn_energy, dim=1).unsqueeze(2)
        # TODO: hard thresholding - if prob less than x, don't consider
        # print(attn_weights.size())
        context_vec = torch.sum(encoder_states * attn_weights, dim=1)
        # print(context_vec.size())
        # print("----------------------------")
        return context_vec

class ShowTellModel(CaptionModel):
    def __init__(self, config, vocab):
        super(ShowTellModel, self).__init__()
        self.vocab_size = len(vocab['w2i'])
        self.input_encoding_size = config['rnn_input_dim']
        self.rnn_type = config['rnn_type']
        self.rnn_size = config['rnn_hidden_dim']
        self.num_layers = config['rnn_layers']
        self.drop_prob_rnn = config['layer_dropout']
        self.drop_prob_lm = config['ff_dropout']
        self.seq_length = config['max_cap_len']
        self.fc_feat_size = config['image_emb_dim'] * 2
        self.emb_dim = config['word_emb_dim']
        self.config = config
        self.ss_prob = 0.0  # Schedule sampling probability

        self.img_embed = nn.Linear(self.fc_feat_size, self.input_encoding_size)
        self.paragraph_encoder = getattr(nn, self.rnn_type.upper())(
                self.input_encoding_size, self.rnn_size,
                self.num_layers, bias=False, dropout=self.drop_prob_rnn, bidirectional=True,
                batch_first=True)
        self.decoder = getattr(nn, self.rnn_type.upper())(
                self.input_encoding_size, self.rnn_size,
                self.num_layers, bias=False, dropout=self.drop_prob_rnn,
                batch_first=True)
        self.embed = nn.Embedding(self.vocab_size, self.emb_dim)
#         self.noun_embed = nn.Embedding(2, self.emb_dim) # binary noun_pos_vec vector projected to emb_dim
#         self.ner_embed = nn.Embedding(2, self.emb_dim) # binary ner_pos_vec vector projected to emb_dim
        # self.pers_embed = nn.Embedding(self.personality_size,
        #                                self.emb_dim)
        self.attn = Attention(config)
        self.logit = nn.Sequential(nn.Linear(self.rnn_size*3, self.rnn_size),
                                   nn.ReLU(),
                                   nn.Linear(self.rnn_size, self.vocab_size))
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit[0].bias.data.fill_(0)
        self.logit[0].weight.data.uniform_(-initrange, initrange)
        self.logit[2].bias.data.fill_(0)
        self.logit[2].weight.data.uniform_(-initrange, initrange)

    def init_decoder_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'lstm':
            return (Variable(weight.new(self.num_layers, bsz,
                                        self.rnn_size).zero_(), requires_grad=True),
                    Variable(weight.new(self.num_layers, bsz,
                                        self.rnn_size).zero_(), requires_grad=True))
        else:
            return Variable(weight.new(self.num_layers, bsz,
                                       self.rnn_size).zero_(), requires_grad=True)

    def init_encoder_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'lstm':
            return (Variable(weight.new(self.num_layers*2, bsz,
                                        self.rnn_size).zero_(), requires_grad=True),
                    Variable(weight.new(self.num_layers*2, bsz,
                                        self.rnn_size).zero_(), requires_grad=True))
        else:
            return Variable(weight.new(self.num_layers*2, bsz,
                                       self.rnn_size).zero_(), requires_grad=True)

#     def forward(self, fc_feats, seq, paragraph, noun_pos, ner_pos): # extra input - ner_pos: positions of nouns/NEs
#     def forward(self, fc_feats, seq, paragraph, noun_pos): # extra input - ner_pos: positions of nouns
    def forward(self, fc_feats, seq, paragraph):
        batch_size = fc_feats.size(0)
        decoder_hidden = self.init_decoder_hidden(batch_size)
        encoder_hidden = self.init_encoder_hidden(batch_size)

        # Encode the input paragraph
        encoder_state = []
        for t in range(self.config['max_par_len']):
            x_t = self.embed(paragraph[:, t]).unsqueeze(1)
#             x_t = x_t + self.noun_embed(noun_pos[:, t].unsqueeze(1))
#             x_t = x_t + self.ner_embed(ner_pos[:, t].unsqueeze(1)) # additional emphasis for named entities
            output, encoder_hidden = self.paragraph_encoder(x_t, encoder_hidden)
            encoder_state.append(output)
        encoder_state = torch.cat(encoder_state, dim=1)
        # print(encoder_state.size())

        outputs = []
        for i in range(seq.size(1)):
            if i == 0:
                xt = self.img_embed(fc_feats).unsqueeze(1)
            else:
                # otherwise no need to sample
                if self.training and i >= 2 and self.ss_prob > 0.0:
                    sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, i-1].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, i-1].data.clone()
                        # fetch prev distribution: shape Nx(M+1)
                        # prob_prev = torch.exp(outputs[-1].data.index_select(
                        # 0, sample_ind))
                        # it.index_copy_(0, sample_ind,
                        # torch.multinomial(prob_prev, 1).view(-1))
                        prob_prev = torch.exp(outputs[-1].data)
                        it.index_copy_(0, sample_ind, torch.multinomial(
                                prob_prev, 1).view(-1).index_select(
                                        0, sample_ind))
                        it = Variable(it, requires_grad=False)
                else:
                    it = seq[:, i-1].clone()
                # break if all the sequences end
                # if i >= 2 and seq[:, i-1].data.sum() == 0:
                #     break
                # x_personality = self.pers_embed(personality)
                xt = self.embed(it).unsqueeze(1)

            output, decoder_hidden = self.decoder(xt, decoder_hidden)

            # Apply attention weights on the encoder hidden states
            context_vec = self.attn(encoder_state, output)

            input = self.dropout(torch.cat((context_vec, output.squeeze(1)), dim=1))
            logits = self.logit(input)
            output = F.log_softmax(logits, dim=1)
            outputs.append(output)

        return torch.cat([_.unsqueeze(1) for _ in outputs[1:]], 1).contiguous()

    def get_logprobs_decoder_hidden(self, it, x_personality, decoder_hidden):
        # 'it' is Variable containing a word index
        x_word = self.embed(it)
        x_personality = x_word.expand_as(x_word)
        xt = torch.cat((x_word, x_personality), dim=1)

        output, decoder_hidden = self.decoder(xt.unsqueeze(1), decoder_hidden)
        logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(1))))

        return logprobs, decoder_hidden

    def sample_beam(self, fc_feats, paragraph, beam_size, opt={}):
        opt['beam_size'] = beam_size
        batch_size = fc_feats.size(0)
        # otherwise this corner case causes a few headaches down the road.
        assert beam_size <= self.vocab_size + 1, 'assume this for now'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            decoder_hidden = self.init_decoder_hidden(beam_size)
            for t in range(2):
                if t == 0:
                    xt = self.img_embed(fc_feats[k:k+1]).expand(
                            beam_size, self.input_encoding_size)
                elif t == 1:  # input <bos>
                    it = torch.ones(1).long().cuda()
                    x_personality = self.pers_embed(Variable(personality[k],
                                                             requires_grad=False))
                    x_word = self.embed(Variable(it,
                                                 requires_grad=False))
                    xt = torch.cat((x_word, x_personality), 1)
                    xt = xt.expand(beam_size, -1)

                output, decoder_hidden = self.decoder(xt.unsqueeze(1), decoder_hidden)
                logprobs = F.log_softmax(
                        self.logit(self.dropout(output.squeeze(1))))

            self.done_beams[k] = self.beam_search(decoder_hidden, logprobs, x_personality, opt=opt)
            # the first beam has highest cumulative score
            seq[:, k] = self.done_beams[k][0]['seq']
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

#     def sample(self, fc_feats, paragraph, noun_pos, ner_pos, sample_max=True, temperature=1.0): # extra input - noun_pos, ner_pos- positions of nouns/NEs in paragraph
#     def sample(self, fc_feats, paragraph, noun_pos, sample_max=True, temperature=1.0): # extra input - noun_pos: positions of nouns in paragraph

    def sample(self, fc_feats, paragraph, sample_max=True, temperature=1.0):
        batch_size = fc_feats.size(0)
        decoder_hidden = self.init_decoder_hidden(batch_size)
        encoder_hidden = self.init_encoder_hidden(batch_size)

        # Encode the input paragraph
        encoder_state = []
        for t in range(self.config['max_par_len']):
            x_t = self.embed(paragraph[:, t]).unsqueeze(1)
#             x_t = x_t + self.noun_embed(noun_pos[:, t].unsqueeze(1))
#             x_t = x_t + self.ner_embed(ner_pos[:, t].unsqueeze(1))
            output, encoder_hidden = self.paragraph_encoder(x_t, encoder_hidden)
            encoder_state.append(output)
        encoder_state = torch.cat(encoder_state, dim=1)

        seq = []
        seqLogprobs = []
        for t in range(self.seq_length):
            if t == 0:
                xt = self.img_embed(fc_feats).unsqueeze(1)
            elif t == 1: # input <sos>
                it = torch.ones(batch_size).long().cuda()
                # x_personality = self.pers_embed(Variable(personality,
                #                                                  requires_grad=False))
                xt = self.embed(Variable(it, requires_grad=False)).unsqueeze(1)
                # xt = torch.cat((x_word, x_personality), 2)
            else:
                if sample_max:
                    sampleLogprobs, it = torch.max(logprobs.data, 1)
                    it = it.view(-1).long()
                else:
                    if temperature == 1.0:
                        # fetch prev distribution: shape Nx(M+1)
                        prob_prev = torch.exp(logprobs.data).cpu()
                    else:
                        # scale logprobs by temperature
                        prob_prev = torch.exp(torch.div(logprobs.data,
                                                        temperature)).cpu()
                    it = torch.multinomial(prob_prev, 1).cuda()
                    # gather the logprobs at sampled positions
                    sampleLogprobs = logprobs.gather(1, Variable(
                            it, requires_grad=False))
                    # and flatten indices for downstream processing
                    it = it.view(-1).long()

                # x_personality = self.pers_embed(Variable(personality,
                #                                                  requires_grad=False))
                xt = self.embed(Variable(it, requires_grad=False)).unsqueeze(1)
                # xt = torch.cat((x_word, x_personality), 2)

            if t >= 2:
                # stop when all finished
                # if t == 2:
                #     unfinished = it > 0
                # else:
                #     unfinished = unfinished * (it > 0)
                # if unfinished.sum() == 0:
                #     break
                # it = it * unfinished.type_as(it)
                seq.append(it)  # seq[t] the input of t+2 time step
                seqLogprobs.append(sampleLogprobs.view(-1))

            output, decoder_hidden = self.decoder(xt, decoder_hidden)
            # Apply attention weights on the encoder hidden states
            context_vec = self.attn(encoder_state, output)

            input = self.dropout(torch.cat((context_vec, output.squeeze(1)), dim=1))
            logits = self.logit(input)
            logprobs = F.log_softmax(logits, dim=1)

            # logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(1))))

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat(
                [_.unsqueeze(1) for _ in seqLogprobs], 1)