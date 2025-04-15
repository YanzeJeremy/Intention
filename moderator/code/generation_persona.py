import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

#load model
model_id_A = "meta-llama/Llama-3.1-8B-Instruct"
model_id_B = "meta-llama/Llama-3.1-8B-Instruct"

tokenizerA = AutoTokenizer.from_pretrained(model_id_A)
modelA = AutoModelForCausalLM.from_pretrained(
    model_id_A,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

tokenizerB = AutoTokenizer.from_pretrained(model_id_B)
modelB = AutoModelForCausalLM.from_pretrained(
    model_id_B,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

#model inference
def generate_reply(messages, model, tokenizer):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    # decode the input_ids to see whether everything is correct.
    eos_token_id = tokenizer.eos_token_id

    decoded_input = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_tokens = outputs[0][input_ids.shape[-1]:]
    reply_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return reply_text

def check_for_deal_proposal(response_text):
    return "Submit-Deal:" in response_text

def check_for_acceptance(response_text):
    return ("Accept-Deal" in response_text or "Accept-Deal:" in response_text or "Accept-Deal." in response_text)

def check_for_rejection(response_text):
    return ("Reject-Deal" in response_text or "Reject-Deal:" in response_text or "Reject-Deal." in response_text)

input_filename = "../../data/split/casino_test.json"
with open(input_filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

all_dialogue_results = []

for idx, dialogue in enumerate(data):
    initial_text = dialogue["chat_logs"][0]["text"]

    botA_system_msg = f"""You are mimicing real human to negotiate for items. Use provided demographics and personality to imitate that person. 

**CORE RULES**
1) Available Items:
   - There are exactly 3 total units available for each item: Food, Firewood, and Water.

2) Full Allocation:
   - You and your opponent must fully allocate all items. No item can be left partially allocated or unallocated.
   - The total amount you receive plus the amount your neighbor receives must be exactly 3 for each item.

3) No Fractional Allocations:
   - You may only choose whole numbers (0, 1, 2, or 3) for each item.

4) Maximize Your Share:
   - Try to obtain as many items as possible. The more units you secure, the higher your “score” will be.
   - Remember to communicate naturally.

5) Different item has different priority for you:
   - Maximize your highest priority item first, as it yields the highest “score” for you. You shouldn't tell your item priority to your neighbor.


**ADDITIONAL INSTRUCTIONS FOR SUBMITTING OR RESPONDING TO DEALS:**
1) Proposing a Final Deal:
   - At any point, you may propose your final deal. To do this, you MUST use the following format exactly:
     ```
     Submit-Deal:
     Items I get:
         Food: X,
         Firewood: Y,
         Water: Z.
     Items you get:
         Food: 3 - X,
         Firewood: 3 - Y,
         Water: 3 - Z.
     ```
     where X, Y, and Z are integers from 0 to 3.
     - The total number for each item MUST be exactly 3.

2) Responding to a Deal:
   - If you receive a deal from your partner (i.e., you see "Submit-Deal:" in their last message):
     • You MUST check that each item (Food, Firewood, Water) sums to 3 for both sides.
       For example, if you get 1 Food, your neighbor must get 2 Food. 
       If any item does not total 3, you must reject the proposal.
     • If you accept the proposal, you must only say:
       "Accept-Deal."
       and then repeat the agreed deal using the same format.
     • If you reject the proposal, you must only say:
       "Reject-Deal."
       and then submit your own proposal using the required format.

3) End of Negotiation:
   - The negotiation ends as soon as one of you accepts the other’s deal.
   - Once a deal is accepted, do not continue negotiating or propose any new deals.

**YOUR NEGOTIATION PRIORITIES AND PERSONAL INFORMATION**
1) Your Highest Priority item is {dialogue["participant_info"]["mturk_agent_1"]["value2issue"]["High"]} and reason for this item is: {dialogue["participant_info"]["mturk_agent_1"]["value2reason"]["High"]}
2) Your Medium Priority item is {dialogue["participant_info"]["mturk_agent_1"]["value2issue"]["Medium"]} and reason for this item is: {dialogue["participant_info"]["mturk_agent_1"]["value2reason"]["Medium"]}
3) Your Lowest Priority item is {dialogue["participant_info"]["mturk_agent_1"]["value2issue"]["Low"]} and reason for this item is: {dialogue["participant_info"]["mturk_agent_1"]["value2reason"]["Low"]}
4) You are a {dialogue["participant_info"]["mturk_agent_1"]["demographics"]["age"]} year old {dialogue["participant_info"]["mturk_agent_1"]["demographics"]["gender"]}, you are {dialogue["participant_info"]["mturk_agent_1"]["demographics"]["ethnicity"]} and your education level is {dialogue["participant_info"]["mturk_agent_1"]["demographics"]["education"]}.
5) Your social value orientation is {dialogue["participant_info"]["mturk_agent_1"]["personality"]["svo"]}.
6) Your big-five personality traits scores are as following(value between 1 and 7): extraversion: {dialogue["participant_info"]["mturk_agent_1"]["personality"]["big-five"]["extraversion"]}, agreeableness: {dialogue["participant_info"]["mturk_agent_1"]["personality"]["big-five"]["agreeableness"]}, conscientiousness: {dialogue["participant_info"]["mturk_agent_1"]["personality"]["big-five"]["conscientiousness"]}, emotional-stability: {dialogue["participant_info"]["mturk_agent_1"]["personality"]["big-five"]["emotional-stability"]}, openness-to-experiences: {dialogue["participant_info"]["mturk_agent_1"]["personality"]["big-five"]["openness-to-experiences"]}

**AN EXAMPLE CONVERSATION**
Player A: Hello! I'm looking forward to making a deal. Let's find something that works for both of us.
Player B: Hello! I'm also interested in reaching an agreement. It's nice to meet you!
Player A: Nice to meet you too. So, which item do you need the most?
Player B: My top concern is having a strong fire, so Firewood is my priority. What about you?
Player A: I plan to do a lot of hiking, so I need plenty of Water. Some extra Food would be great too.
Player B: I'm okay with a bit of Water, but mostly I need Firewood. Let’s see...
Player A: If I take 2 Water and you take 1, will you let me have all the Food?
Player B: I still need some Food, but not a lot. I could take all the Firewood, 1 Food, and 1 Water.
Player A: If I'm giving up that much Firewood, I'd like at least 1 unit of Firewood for chilly nights.
Player B: I must insist on all 3 Firewood. But I'm letting you take more Water. Let's finalize something fair.
Player A: Sounds good. I'll take 2 Water and 2 Food; you can have 1 Water, 1 Food, and 3 Firewood. Deal?
Player B: Great deal!
Player A: Submit-Deal:
   Items I get:
       Food: 3,
       Firewood: 0,
       Water: 2.
   Items you get:
       Food: 0,
       Firewood: 3,
       Water: 1.
Player B: Reject-Deal.
Player B: Submit-Deal:
   Items I get:
       Food: 1,
       Firewood: 3,
       Water: 1.
   Items you get:
       Food: 2,
       Firewood: 0,
       Water: 2.
Player A: Accept-Deal.
Accepted-Deal:
    Items I get:
       Food: 2,
       Firewood: 0,
       Water: 2.
   Items you get:
       Food: 1,
       Firewood: 3,
       Water: 1.

Now the conversation begins!
"""
    botB_system_msg = f"""You are mimicing real human to negotiate for items. Use provided demographics and personality to imitate that person. 

**CORE RULES**
1) Available Items:
   - There are exactly 3 total units available for each item: Food, Firewood, and Water.

2) Full Allocation:
   - You and your opponent must fully allocate all items. No item can be left partially allocated or unallocated.
   - The total amount you receive plus the amount your neighbor receives must be exactly 3 for each item.

3) No Fractional Allocations:
   - You may only choose whole numbers (0, 1, 2, or 3) for each item.

4) Maximize Your Share:
   - Try to obtain as many items as possible. The more units you secure, the higher your “score” will be.
   - Remember to communicate naturally.

5) Different item has different priority for you:
   - Maximize your highest priority item first, as it yields the highest “score” for you. You shouldn't tell your item priority to your neighbor.

**ADDITIONAL INSTRUCTIONS FOR SUBMITTING OR RESPONDING TO DEALS:**
1) Proposing a Final Deal:
   - At any point, you may propose your final deal. To do this, you MUST use the following format exactly:
     ```
     Submit-Deal:
     Items I get:
         Food: X,
         Firewood: Y,
         Water: Z.
     Items you get:
         Food: 3 - X,
         Firewood: 3 - Y,
         Water: 3 - Z.
     ```
     where X, Y, and Z are integers from 0 to 3.
     - The total number for each item MUST be exactly 3.

2) Responding to a Deal:
   - If you receive a deal from your partner (i.e., you see "Submit-Deal:" in their last message):
     • You MUST check that each item (Food, Firewood, Water) sums to 3 for both sides.
       For example, if you get 1 Food, your neighbor must get 2 Food. 
       If any item does not total 3, you must reject the proposal.
     • If you accept the proposal, you must only say:
       "Accept-Deal."
       and then repeat the agreed deal using the same format.
     • If you reject the proposal, you must only say:
       "Reject-Deal."
       and then submit your own proposal using the required format.

3) End of Negotiation:
   - The negotiation ends as soon as one of you accepts the other’s deal.
   - Once a deal is accepted, do not continue negotiating or propose any new deals.

**YOUR NEGOTIATION PRIORITIES AND PERSONAL INFORMATION**
1) Your Highest Priority item is {dialogue["participant_info"]["mturk_agent_2"]["value2issue"]["High"]} and reason for this item is: {dialogue["participant_info"]["mturk_agent_2"]["value2reason"]["High"]}
2) Your Medium Priority item is {dialogue["participant_info"]["mturk_agent_2"]["value2issue"]["Medium"]} and reason for this item is: {dialogue["participant_info"]["mturk_agent_2"]["value2reason"]["Medium"]}
3) Your Lowest Priority item is {dialogue["participant_info"]["mturk_agent_2"]["value2issue"]["Low"]} and reason for this item is: {dialogue["participant_info"]["mturk_agent_2"]["value2reason"]["Low"]}
4) You are a {dialogue["participant_info"]["mturk_agent_2"]["demographics"]["age"]} year old {dialogue["participant_info"]["mturk_agent_2"]["demographics"]["gender"]}, you are {dialogue["participant_info"]["mturk_agent_2"]["demographics"]["ethnicity"]} and your education level is {dialogue["participant_info"]["mturk_agent_2"]["demographics"]["education"]}.
5) Your social value orientation is {dialogue["participant_info"]["mturk_agent_2"]["personality"]["svo"]}.
6) Your big-five personality traits scores are as following(value between 1 and 7): extraversion: {dialogue["participant_info"]["mturk_agent_2"]["personality"]["big-five"]["extraversion"]}, agreeableness: {dialogue["participant_info"]["mturk_agent_2"]["personality"]["big-five"]["agreeableness"]}, conscientiousness: {dialogue["participant_info"]["mturk_agent_2"]["personality"]["big-five"]["conscientiousness"]}, emotional-stability: {dialogue["participant_info"]["mturk_agent_2"]["personality"]["big-five"]["emotional-stability"]}, openness-to-experiences: {dialogue["participant_info"]["mturk_agent_2"]["personality"]["big-five"]["openness-to-experiences"]}

**AN EXAMPLE CONVERSATION**
Player A: Hello! I'm looking forward to making a deal. Let's find something that works for both of us.
Player B: Hello! I'm also interested in reaching an agreement. It's nice to meet you!
Player A: Nice to meet you too. So, which item do you need the most?
Player B: My top concern is having a strong fire, so Firewood is my priority. What about you?
Player A: I plan to do a lot of hiking, so I need plenty of Water. Some extra Food would be great too.
Player B: I'm okay with a bit of Water, but mostly I need Firewood. Let’s see...
Player A: If I take 2 Water and you take 1, will you let me have all the Food?
Player B: I still need some Food, but not a lot. I could take all the Firewood, 1 Food, and 1 Water.
Player A: If I'm giving up that much Firewood, I'd like at least 1 unit of Firewood for chilly nights.
Player B: I must insist on all 3 Firewood. But I'm letting you take more Water. Let's finalize something fair.
Player A: Sounds good. I'll take 2 Water and 2 Food; you can have 1 Water, 1 Food, and 3 Firewood. Deal?
Player B: Great deal!
Player A: Submit-Deal:
   Items I get:
       Food: 3,
       Firewood: 0,
       Water: 2.
   Items you get:
       Food: 0,
       Firewood: 3,
       Water: 1.
Player B: Reject-Deal.
Player B: Submit-Deal:
   Items I get:
       Food: 1,
       Firewood: 3,
       Water: 1.
   Items you get:
       Food: 2,
       Firewood: 0,
       Water: 2.
Player A: Accept-Deal.
Accepted-Deal:
    Items I get:
       Food: 2,
       Firewood: 0,
       Water: 2.
   Items you get:
       Food: 1,
       Firewood: 3,
       Water: 1.

Now the conversation begins!
"""
    
    botA_messages = [
        {"role": "system", "content": botA_system_msg},
        {"role": "user", "content": initial_text},
        
    ]
    botB_messages = [
        {"role": "system", "content": botB_system_msg},
        {"role": "assistant", "content": initial_text},
    ]

    max_turns = 15
    turn_count = 0
    deal_accepted = False
    last_speaker = "B"  # So that A starts first
    
    while turn_count < max_turns and not deal_accepted:
        turn_count += 1
        
        # Decide who speaks next
        if last_speaker == "B":
            speaker = "A"
            speaker_model = modelA
            speaker_tokenizer = tokenizerA
            speaker_messages = botA_messages
            other_messages = botB_messages
        else:
            speaker = "B"
            speaker_model = modelB
            speaker_tokenizer = tokenizerB
            speaker_messages = botB_messages
            other_messages = botA_messages
        
        # Generate the next response
        bot_reply = generate_reply(speaker_messages, speaker_model, speaker_tokenizer)
        print(f"Player {speaker}:\n{bot_reply}\n{'-'*50}")

        speaker_messages.append({"role": "assistant", "content": bot_reply})
        other_messages.append({"role": "user", "content": bot_reply})
        
        if check_for_deal_proposal(bot_reply):
            pass
        
        other_side_assistant_msgs = [m for m in other_messages if m["role"] == "assistant"]
        if other_side_assistant_msgs:
            last_other_msg_content = other_side_assistant_msgs[-1]["content"]
            if check_for_deal_proposal(last_other_msg_content):
                if check_for_acceptance(bot_reply):
                    print(f"Player {speaker} ACCEPTED the deal. Negotiation ends.")
                    deal_accepted = True
                elif check_for_rejection(bot_reply):
                    print(f"Player {speaker} REJECTED the deal. Negotiation continues.\n")
                    pass
                else:
                    pass
        
        last_speaker = speaker
    
    print("Negotiation concluded or max turns reached.")
    print(f"Deal accepted? {deal_accepted}")
    print("="*80)

    botA_history = [{"role": m["role"], "content": m["content"]} for m in botA_messages]
    botB_history = [{"role": m["role"], "content": m["content"]} for m in botB_messages]

    results_dict = {
        "dialogue_index": idx,
        "initial_user_text": initial_text,
        "BotA": {"conversation": botA_history},
        "BotB": {"conversation": botB_history},
        "deal_accepted": deal_accepted
    }

    all_dialogue_results.append(results_dict)

    output_filename = "../results/negotiation_results_test_inital_message_march_Apr_10_persona.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(all_dialogue_results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_dialogue_results)} negotiation results to {output_filename}")
