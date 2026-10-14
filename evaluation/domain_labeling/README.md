# Domain labeling and discovery

This folder groups the math/general labeling workflow and its comparison tools:

- `label_math_general_qwen_t4x2.ipynb` and `label_math_general_qwen_kaggle.py`
  label local datasets with Qwen3-1.7B on Kaggle T4 x2.
- `compare_qwen_regex_labels.py` compares Qwen annotations with the project's
  regex detector.
- `analyze_qwen_prompt_tokens.py` measures tokenizer lengths for those inputs.
- `find_math_regex.py` is the earlier scored-regex labeling approach.
- `generate_math_qwen_kaggle.py` evaluates Qwen models on the MATH benchmark.
- `test_qwen_domain_labeling.py` checks annotation parsing and output alignment.

See `label_math_general_qwen_kaggle.md` for Kaggle setup and run instructions.
Run project scripts from the repository root so relative input arguments resolve
consistently.
