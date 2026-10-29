"""
Generation module (paper III.E, III.F): wraps a local causal LM so that
attention weights and token log-probabilities are actually accessible
(this is why the framework needs an open-weight model rather than an
API-only model like Groq/OpenAI, which never expose internals).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import torch


@dataclass
class GenerationResult:
    text: str
    generated_token_ids: "object"          # torch.LongTensor [gen_len]
    input_token_ids: "object"              # torch.LongTensor [prompt_len]
    attentions: "object"                   # tuple(layers) of [1, heads, gen_len, seq_len] (see note below)
    token_logprobs: List[float]             # per generated token, log P(t_i)
    doc_token_spans: List[tuple]            # [(start, end)] per context document in input_token_ids


class LLMGenerator:
    """Loads any causal-LM from Hugging Face Hub (Llama/Mistral/Qwen/Phi
    family all work). Designed to run in Colab/Kaggle with a GPU, but also
    degrades sensibly on CPU.

    Efficiency notes:
    - quantize="4bit"/"8bit" (GPU only, needs `bitsandbytes`) cuts VRAM
      usage roughly in half (8-bit) or to a quarter (4-bit) vs fp16, which
      is what actually lets a 3B-7B model fit on a free-tier T4 instead of
      needing an A100. Accuracy impact is typically <1-2 F1 points for
      4-bit NF4 on most QA-style tasks -- worth it for the resource
      trade-off, but re-measure once you have real eval numbers.
    - fp16 (the default on GPU) roughly halves memory and latency vs fp32
      with no accuracy loss on inference (only training is precision
      sensitive here).
    - attn_implementation="sdpa" (PyTorch's fused scaled-dot-product
      attention) is faster than the default "eager" path, but note it can
      change what `output_attentions` returns on some model families --
      kept as "eager" here specifically because this project depends on
      reading real attention weights (Eq. 7); don't switch this without
      checking the attentions tuple still comes back non-empty.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 device: str = "cuda", dtype: str = "auto", max_new_tokens: int = 256,
                 quantize: Optional[str] = None, adapter_path: Optional[str] = None):
        """`adapter_path`: directory produced by train/train_lora.py. When
        set, loads the base model then applies the LoRA adapter on top --
        this is how a fine-tuned generator plugs into the SOAR-RAG pipeline
        (SOARRAGPipeline doesn't need to know or care whether the generator
        it's calling is the base model or a fine-tuned one)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = torch.float16 if (dtype == "auto" and self.device == "cuda") else torch.float32

        quant_config = None
        if quantize in ("4bit", "8bit"):
            if self.device != "cuda":
                raise ValueError("quantize='4bit'/'8bit' requires a CUDA GPU (bitsandbytes has no CPU kernel).")
            from transformers import BitsAndBytesConfig
            if quantize == "4bit":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
                )
            else:
                quant_config = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype if quant_config is None else None,
            quantization_config=quant_config,
            output_attentions=True,
            attn_implementation="eager",  # required for reliable output_attentions -- see docstring
        )
        if quant_config is None:
            self.model = self.model.to(self.device)

        self.adapter_path = adapter_path
        if adapter_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            print(f"Loaded LoRA adapter from {adapter_path}")

        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def build_prompt(self, query: str, documents: List[str]) -> tuple[str, list[tuple]]:
        """Builds C0 = Q + R0 (Eq. 5). Returns the prompt text plus the
        character spans of each document within the prompt, later converted
        to token spans so attention can be attributed back to documents.

        IMPORTANT: for *-Instruct models (Qwen2.5-Instruct etc.), the model
        was fine-tuned to expect chat-template-formatted input
        (<|im_start|>system/user/assistant turns). Feeding it a raw
        completion-style string (no chat template) is out-of-distribution
        for an instruct model -- in practice this caused the model to
        sometimes hallucinate its own system/user turns into the *answer*
        instead of actually answering (observed directly: SOAR outputs on
        some questions contained leaked text like "You are an AI
        assistant... I will now provide the answer according to the
        guidelines above" -- that's the model completing what it thinks a
        missing chat template should look like, not a real answer).

        Fix: build the same header/context/question content as before, but
        wrap it in the tokenizer's real chat template when the model has
        one. The character spans are computed on `user_content` first
        (unchanged from before) and then shifted by wherever that exact
        substring ends up inside the final templated prompt string, so the
        attention-attribution mechanism (Eq. 7/8) is unaffected by this
        fix -- it just gets fed a prompt the model can actually follow.
        """
        header = "Context:\n"
        spans, cursor, body = [], 0, ""
        for i, doc in enumerate(documents):
            tag = f"[DOC{i}] "
            piece = f"{tag}{doc}\n"
            start = len(header) + cursor
            body += piece
            cursor += len(piece)
            end = len(header) + cursor
            spans.append((start, end))
        user_content = (
            f"{header}{body}\nQuestion: {query}\n"
            "Answer in a short phrase or single sentence using ONLY the context above. "
            "Do not explain your reasoning, do not repeat the question, and do not add "
            "any text after the answer."
        )

        has_chat_template = getattr(self.tokenizer, "chat_template", None)
        if has_chat_template:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You answer questions using ONLY the context the user provides. "
                        "Always respond with a short, direct phrase or single sentence -- "
                        "never explanations, never meta-commentary, never repeat these "
                        "instructions."
                    ),
                },
                {"role": "user", "content": user_content},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            offset = prompt.find(user_content)
            if offset == -1:
                # Should not happen (apply_chat_template inserts content
                # verbatim), but never let a span bug crash the run --
                # fall back to un-shifted spans against the raw content.
                prompt, offset = user_content, 0
        else:
            # Base (non-chat) models: no template exists, behave as before.
            prompt, offset = user_content, 0

        shifted_spans = [(s + offset, e + offset) for (s, e) in spans]
        return prompt, shifted_spans

    def _char_to_token_spans(self, prompt: str, char_spans: List[tuple], enc) -> List[tuple]:
        spans = []
        for (cs, ce) in char_spans:
            ts = enc.char_to_token(cs) or 0
            te = enc.char_to_token(max(ce - 1, cs)) or ts
            spans.append((ts, te + 1))
        return spans

    @torch.inference_mode()  # faster than no_grad(): also disables version-counter tracking
    def generate(self, query: str, documents: List[str]) -> GenerationResult:
        prompt, char_spans = self.build_prompt(query, documents)
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
        doc_token_spans = self._char_to_token_spans(prompt, char_spans, enc.encodings[0]) if enc.encodings else []

        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            output_attentions=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
        input_len = enc["input_ids"].shape[1]
        gen_ids = out.sequences[0][input_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # token-level log-probabilities P(t_i) for confidence estimation (Eq. 14)
        logprobs = []
        for step, score in enumerate(out.scores):
            logp = torch.log_softmax(score[0], dim=-1)
            tok_id = gen_ids[step] if step < len(gen_ids) else None
            if tok_id is not None:
                logprobs.append(float(logp[tok_id].item()))

        # attentions: out.attentions is a tuple per generation step, each a
        # tuple per layer of [batch, heads, q_len, k_len]. Only the FIRST
        # decoding step has q_len == full prompt length (later steps only
        # attend from the single newest token), so that first step is what
        # gives a real per-document attention distribution over the prompt
        # (matches Eq. 7/8). Later steps could be summed in for a fuller
        # picture at extra compute cost; kept simple here.
        first_step_attn = out.attentions[0] if out.attentions else None

        return GenerationResult(
            text=text.strip(),
            generated_token_ids=gen_ids,
            input_token_ids=enc["input_ids"][0],
            attentions=first_step_attn,
            token_logprobs=logprobs,
            doc_token_spans=doc_token_spans,
        )
