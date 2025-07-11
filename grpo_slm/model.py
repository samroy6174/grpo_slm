import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ReasoningModel:
    def __init__(self, model_name="gpt2", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(self, prompts, **gen_kwargs):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    def log_probs(self, prompts, responses):
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True)
        dec = self.tokenizer(responses, return_tensors="pt", padding=True)
        input_ids = torch.cat([enc.input_ids, dec.input_ids[:, 1:]], dim=1).to(self.device)
        attention = torch.ones_like(input_ids).to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention, labels=input_ids)
        # outputs.loss is cross-entropy averaged
        # to get token-wise log-probs:
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        logp = -torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none"
        ).view(shift_labels.size())
        # sum per sample:
        return logp.sum(dim=1)
