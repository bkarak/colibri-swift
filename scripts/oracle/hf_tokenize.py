# Prompt ids from a model's own Hugging Face tokenizer.json (the `tokenizers` library),
# the reference the port's tokenizer is checked against for real models whose upstream
# engine has no token-dump mode (olmoe). Third-party packages, so run it only in a
# throwaway container, with the prompts piped in and the ids printed:
#
#   python3 -I scripts/oracle/record.py --print-prompts olmoe | \
#     docker run --rm -i -v "$PWD/scripts/oracle/hf_tokenize.py:/s/hf_tokenize.py:ro" python:3.12-slim \
#       sh -c "pip install -q 'tokenizers==0.20.3' 'huggingface_hub==0.26.2' && \
#              python -I /s/hf_tokenize.py allenai/OLMoE-1B-7B-0125-Instruct" \
#     > scripts/oracle/prompt-ids/olmoe-1b-7b-0125-instruct.json
import json, sys
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
repo = sys.argv[1]
tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
prompts = json.load(sys.stdin)
out = {"tokenizer": repo + "/tokenizer.json", "library": "tokenizers 0.20.3",
       "prompts": {name: tok.encode(text, add_special_tokens=False).ids for name, text in prompts}}
print(json.dumps(out, ensure_ascii=False, indent=1))
