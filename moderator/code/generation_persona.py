import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizerA = AutoTokenizer.from_pretrained(model_id)
modelA = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
tokenizerB = AutoTokenizer.from_pretrained(model_id)
modelB = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

def generate_reply(messages, model, tokenizer):
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_tokens = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

def check_for_deal_proposal(text): return "Submit-Deal:" in text
def check_for_acceptance(text): return any(k in text.lower() for k in ["i accept", "accept the deal"])
def check_for_rejection(text): return any(k in text.lower() for k in ["i reject", "reject the deal", "not accept"])

# Load formatted test data
input_filename = "../data/split/casino_test_formatted.json"
with open(input_filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

all_results = []

# === Loop through each dialogue ===
for idx, dialogue in enumerate(data):
    initial_text = dialogue["chat_logs"][0]["text"]
    p1 = dialogue["participant_info"]["mturk_agent_1"]["value2issue"]
    p2 = dialogue["participant_info"]["mturk_agent_2"]["value2issue"]

    def make_system_msg(name, persona):
        return f"""
You are {name}, a camper negotiating how to fairly split camping supplies: Food, Firewood, and Water.
You want to maximize your personal gain without revealing your preferences.

**Persona Priorities**:
- Highest Priority: {persona["High"]}
- Medium Priority: {persona["Medium"]}
- Lowest Priority: {persona["Low"]}

Behave like a real human who values these items accordingly. Use emotional appeals, logical reasoning, or personal stories, but don’t reveal your item priorities directly.

The conversation should feel natural and human-like. You must strictly follow the format when proposing or responding to deals.
"""

    # Add system and initial user messages
    botA_messages = [{"role": "system", "content": make_system_msg("Player A", p1)},
                     {"role": "user", "content": initial_text}]
    botB_messages = [{"role": "system", "content": make_system_msg("Player B", p2)},
                     {"role": "assistant", "content": initial_text}]

    turn_count = 0
    deal_accepted = False
    last_speaker = "B"  # A starts first

    while turn_count < 15 and not deal_accepted:
        turn_count += 1

        if last_speaker == "B":
            speaker, model, tokenizer = "A", modelA, tokenizerA
            msgs, other_msgs = botA_messages, botB_messages
        else:
            speaker, model, tokenizer = "B", modelB, tokenizerB
            msgs, other_msgs = botB_messages, botA_messages

        reply = generate_reply(msgs, model, tokenizer)
        print(f"Player {speaker}:\n{reply}\n{'-'*50}")
        msgs.append({"role": "assistant", "content": reply})
        other_msgs.append({"role": "user", "content": reply})

        # Check if previous message was a proposal
        other_last_assistant = [m for m in other_msgs if m["role"] == "assistant"]
        if other_last_assistant:
            last_reply = other_last_assistant[-1]["content"]
            if check_for_deal_proposal(last_reply):
                if check_for_acceptance(reply):
                    deal_accepted = True
                    print(f"Player {speaker} ACCEPTED the deal.")
                elif check_for_rejection(reply):
                    print(f"Player {speaker} REJECTED the deal.")

        last_speaker = speaker

    print(f"Negotiation ended. Deal accepted? {deal_accepted}")
    print("="*70)

    all_results.append({
        "dialogue_index": idx,
        "initial_user_text": initial_text,
        "BotA": {"conversation": botA_messages},
        "BotB": {"conversation": botB_messages},
        "deal_accepted": deal_accepted
    })

# Save results
output_filename = "negotiation_results_test_all.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"Saved {len(all_results)} negotiation results to {output_filename}")
