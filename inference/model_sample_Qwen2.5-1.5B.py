# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

def model_sample(model_name):
	tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)
	model = AutoModelForCausalLM.from_pretrained(model_name)
	messages = [
		{"role": "user", "content": "你好"},
	]
	inputs = tokenizer.apply_chat_template(
		messages,
		add_generation_prompt=True,
		tokenize=True,
		return_dict=True,
		return_tensors="pt",
	).to(model.device)

	outputs = model.generate(**inputs, max_new_tokens=40)
	print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))


if __name__ == "__main__":

	# print("------------------- Original Pretrain Sample ------------------- \n")

	# model_name = "/root/projects/happy-llm/ZeroLLM/autodl-tmp/model/Qwen2.5-1.5B"
	# model_sample(model_name)



	# print("------------------- Pretrain Sample ------------------- \n")
	# model_name = "/root/projects/happy-llm/ZeroLLM/autodl-tmp/model/pretrain_Qwen2.5-1.5B/output/checkpoint-390"
	# model_sample(model_name)

	print("\n ------------------- SFT Sample ------------------- \n")
	model_name = "/root/projects/happy-llm/ZeroLLM/autodl-tmp/model/sft_train_Qwen2.5-1.5B/output/checkpoint-900"
	model_sample(model_name)