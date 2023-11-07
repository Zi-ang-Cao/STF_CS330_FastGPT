import torch
import torch.nn.functional as F
import transformers
import utils

class LMEnv:

    def __init__(self, args):
        self.model, self.tok = utils.get_model_and_tokenizer(args.model_name)
        self.input_ids = None
        assert isinstance(self.model, transformers.GPT2LMHeadModel)
        self.stop_tokens = utils.stop_tokens(self.tok)

    def do_sample(
        self,
        max_tokens: int,
        sampling_mode: str = "likelihood"
    ) -> List[int]:
        sampled_tokens = []
        cum_logits = []
        n = 0
        cur_input = self.input_ids
        past_kvs = None
        with torch.inference_mode():
            while n < max_tokens:
                outputs = self.model(cur_input, past_key_values=past_kvs, use_cache=True)
                local_logits = outputs.logits[:, -1, :]
                cum_logits.append(local_logits)
                if sampling_mode == "argmax":
                    sampled_token = torch.argmax(local_logits, dim=-1)
                elif sampling_mode == "likelihood":
                    sampled_token = torch.multinomial(F.softmax(local_logits, dim=-1), num_samples=1)
                if sampled_token[0].item() in self.stop_tokens:
                    break
                sampled_tokens.append(sampled_token[0])
                cur_input = sampled_token.unsqueeze(0)
                past_kvs = outputs.past_key_values
                n += 1
        # (Batch Size=1, Input + Output Seq Len)
        new_tokens = torch.cat((self.input_ids, torch.tensor(sampled_tokens).unsqueeze(dim=0)), dim=1) 
        # (Seq Len, Batch Size=1, Vocab Size)
        new_logits = torch.stack(cum_logits, dim=0) 
        return new_tokens, new_logits

    def step():
        new_tokens, new_logits = self.do_sample(max_tokens=1)
        self.input_ids = new_tokens

        pass
