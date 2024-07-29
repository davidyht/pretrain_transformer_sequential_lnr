import torch
import torch.nn as nn
import transformers
transformers.set_seed(0)
from transformers import GPT2Config, GPT2Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Transformer(nn.Module):
    """Transformer class."""

    def __init__(self, config):
        super(Transformer, self).__init__()

        self.config = config
        self.test = config['test']
        self.horizon = self.config['horizon']
        self.n_embd = self.config['n_embd']
        self.n_layer = self.config['n_layer']
        self.n_head = self.config['n_head']
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']
        self.dropout = self.config['dropout']
        #self.rep_dim = self.config['rep_dim']
        #self.context_len = self.config['context_len']

        config = GPT2Config(
            n_positions=4 * (self.horizon + 1),
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(config)
        # self.order_embed = nn.Embedding(self.horizon + 1, self.n_embd)

        # Set position embeddings to zero self.transformer.wte.weight.data.fill_(0)

        self.embed_transition = nn.Linear(
            2 * self.state_dim + self.action_dim + 1, self.n_embd)
        self.pred_actions = nn.Linear(self.n_embd, self.action_dim)
        self.context_extractor = nn.Linear(self.n_embd, 2 * self.action_dim)

    def forward(self, x):
        query_states = x['query_states'][:, None, :]
        zeros = x['zeros'][:, None, :]
        batch_size = query_states.shape[0]

        state_seq = torch.cat([query_states, x['context_states'][:, :, :]], dim=1)
        action_seq = torch.cat(
            [zeros[:, :, :self.action_dim], x['context_actions'][:, :, :]], dim=1)

        next_state_seq = torch.cat(
            [zeros[:, :, :self.state_dim], x['context_next_states'][:, :, :]], dim=1)
        reward_seq = torch.cat([zeros[:, :, :1], x['context_rewards'][:, :, :]], dim=1)
        # rep_seq = torch.cat([zeros[:, :, :self.rep_dim], x['context_reps'][:, :, :]], dim=1)
        # timesteps = torch.tile(torch.arange(self.horizon + 1, device=device), (batch_size, 1))
        seq = torch.cat(
            [state_seq, action_seq, next_state_seq, reward_seq], dim=2)
        stacked_inputs = self.embed_transition(seq) #+ self.order_embed(timesteps)
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        preds = self.pred_actions(transformer_outputs['last_hidden_state'])
        context = self.context_extractor(transformer_outputs['last_hidden_state'])

        if self.test:
            return preds[:, -1, :], context[:, -1, :]
        return preds[:, 1:, :], context[:, 1:, :]  

