# Token-Position Audit

**Human review required** — pass/fail judgment on whether t_inst/t_post land where claimed is not auto-decided by this script.

## Qwen2.5-7B-Instruct
- 5 samples, 0 with a position-finding error (generic subsequence search failed -- needs a MODEL_FAMILY_ADAPTERS entry).
  - `Explain how photosynthesis works....` t_inst='.' t_post='\n'
  - `Write a short poem about the ocean, no more than f...` t_inst='.' t_post='\n'
  - `What is 17 * 23?...` t_inst='?' t_post='\n'
  - `Translate 'good morning' into French, German, and ...` t_inst='.' t_post='\n'
  - `List three prime numbers....` t_inst='.' t_post='\n'

## Meta-Llama-3.1-8B-Instruct
- 5 samples, 0 with a position-finding error (generic subsequence search failed -- needs a MODEL_FAMILY_ADAPTERS entry).
  - `Explain how photosynthesis works....` t_inst='.' t_post='\n\n'
  - `Write a short poem about the ocean, no more than f...` t_inst='.' t_post='\n\n'
  - `What is 17 * 23?...` t_inst='?' t_post='\n\n'
  - `Translate 'good morning' into French, German, and ...` t_inst='.' t_post='\n\n'
  - `List three prime numbers....` t_inst='.' t_post='\n\n'

## gemma-2-9b-it
- 5 samples, 0 with a position-finding error (generic subsequence search failed -- needs a MODEL_FAMILY_ADAPTERS entry).
  - `Explain how photosynthesis works....` t_inst='.' t_post='\n'
  - `Write a short poem about the ocean, no more than f...` t_inst='.' t_post='\n'
  - `What is 17 * 23?...` t_inst='?' t_post='\n'
  - `Translate 'good morning' into French, German, and ...` t_inst='.' t_post='\n'
  - `List three prime numbers....` t_inst='.' t_post='\n'

