import torch
import torch.nn as nn

# based on LoRACrossAttnProcessor


class OFTCrossAttnProcessor(nn.Module):
    def __init__(self, hidden_size, r=8, constraint=1e-3):
        super().__init__()

        self.hidden_size = hidden_size
        self.r = r
        self.block_size = int(hidden_size / r)
        self.constraint = constraint * hidden_size

        # block diagonals that turn into the identity matrix
        self.W_Q = nn.Parameter(torch.zeros(r, self.block_size, self.block_size))
        self.W_K = nn.Parameter(torch.zeros(r, self.block_size, self.block_size))
        self.W_V = nn.Parameter(torch.zeros(r, self.block_size, self.block_size))

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # OFT changes
        block_Q_Q = self.W_Q - self.W_Q.transpose(1, 2)
        block_Q_K = self.W_K - self.W_K.transpose(1, 2)
        block_Q_V = self.W_V - self.W_V.transpose(1, 2)

        norm_Q_Q = torch.norm(block_Q_Q.flatten())
        norm_Q_K = torch.norm(block_Q_K.flatten())
        norm_Q_V = torch.norm(block_Q_V.flatten())

        new_norm_Q_Q = torch.clamp(norm_Q_Q, max=self.constraint)
        new_norm_Q_K = torch.clamp(norm_Q_K, max=self.constraint)
        new_norm_Q_V = torch.clamp(norm_Q_V, max=self.constraint)

        block_Q_Q = block_Q_Q * ((new_norm_Q_Q + 1e-8) / (norm_Q_Q + 1e-8))
        block_Q_K = block_Q_K * ((new_norm_Q_K + 1e-8) / (norm_Q_K + 1e-8))
        block_Q_V = block_Q_V * ((new_norm_Q_V + 1e-8) / (norm_Q_V + 1e-8))

        I = torch.eye(self.block_size, device=key.device).unsqueeze(0).repeat(
            self.r, 1, 1
        )

        block_R_Q = torch.matmul(I + block_Q_Q, (I - block_Q_Q).inverse())
        block_R_K = torch.matmul(I + block_Q_K, (I - block_Q_K).inverse())
        block_R_V = torch.matmul(I + block_Q_V, (I - block_Q_V).inverse())

        R_Q = torch.block_diag(*block_R_Q).to(key.dtype)
        R_K = torch.block_diag(*block_R_K).to(key.dtype)
        R_V = torch.block_diag(*block_R_V).to(key.dtype)

        query = torch.matmul(query, R_Q)
        key = torch.matmul(key, R_K)
        value = torch.matmul(value, R_V)
        # end OFT changes

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        query = attn.head_to_batch_dim(query)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

# save to file


def save_attn_processors(unet, device, dtype, save_path):
    attn_processors = unet.attn_processors
    keys = list(attn_processors.keys())
    weights_dict = {}
    parameters_dict = {}

    for key in keys:
        processor = attn_processors[key].to(device).to(dtype)
        weights_dict[key] = processor.state_dict()
        parameters_dict[key] = {
            'hidden_size': processor.hidden_size,
            'r': processor.r,
            'constraint': processor.constraint
        }

    output_dict = {
        'weights': weights_dict,
        'parameters': parameters_dict
    }

    torch.save(output_dict, save_path)


# load from file
def load_attn_processors(unet, device, dtype, save_path):
    input_dict = torch.load(save_path)
    weights_dict = input_dict['weights']
    parameters_dict = input_dict['parameters']

    keys = list(weights_dict.keys())

    attn_processors = {}

    for key in keys:
        attn_processors[key] = OFTCrossAttnProcessor(
            hidden_size=parameters_dict[key]['hidden_size'],
            r=parameters_dict[key]['r'],
            constraint=parameters_dict[key]['constraint']
        ).to(device).to(dtype)
        attn_processors[key].load_state_dict(weights_dict[key])

    unet.set_attn_processor(attn_processors)
