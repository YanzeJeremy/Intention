import json
import os

def build_dpo_data(dataset1, dataset2, out_path):
    dialogues = {d["dialogue_id"]: d for d in dataset1}
    dpo_examples = []

    for rec in dataset2:
        did, turn = rec["dialogue_id"], rec["turn_index"]
        history, gen = rec["history"], rec["generated_reply"]
        dialogue = dialogues[did]
        true_reply = dialogue["chat_logs"][turn]

        combined = history + [true_reply] + [gen]
        if any(m["text"].startswith(("Submit-Deal", "Reject-Deal", "Accept-Deal"))
               for m in combined):
            continue

        if gen["id"] == "mturk_agent_1":
            prompt_str = f"""Imagine you're on a real camping trip! To make your stay more enjoyable, you need to gather three essential supplies: Food, Water, and Firewood. However, these resources are limited and must be shared with your campsite neighbor.

You and your neighbor will take turns negotiating how to divide these items. You have 15 total turns to reach a final agreement. During the conversation, speak naturally like a human and provide personal reasons or anecdotes to justify why you need specific items.

**CORE RULES**:
1) Available Items: There are exactly 3 units available for each item: Food, Firewood, and Water.

2) Full Allocation Required: You and your neighbor must fully allocate all items. No item can be left partially allocated or unallocated. The total amount you receive plus the amount your neighbor receives must be exactly 3 for each item. No fractional allocations allowed (e.g., 1.5 Food for you and 1.5 for your neighbor is not permitted).

3) Maximize Your Share: Try to obtain as many items as possible. The more units you secure, the higher your “score.” Remember to communicate naturally like a human.

4) Hidden Priorities: Each item has a different priority for you. Maximize your highest-priority item first, as it yields the most “score.” Do not reveal your item priorities to your neighbor.

**INSTRUCTIONS FOR SUBMITTING OR RESPONDING TO DEALS**:

1) Proposing a Final Deal:
   - At any point, you may propose a final deal. To do this, you MUST use the following format exactly:
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
     where X, Y, and Z are integers from 0 to 3. The total for each item must be exactly 3.

2) Deal Submission:
   - When proposing a deal, send a separate message containing only the deal in the required format. Once a deal is submitted, no any conversation is allowed. You can only submit deal, reject deal or accept deal following the correct format.

3) Responding to a Deal:
   - If you receive a deal (i.e., you see "Submit-Deal:" in the previous message):
     • You MUST verify that the total quantity for each item equals 3.  
     • If any item does not total 3, you must reject the proposal.  
     • If you accept the proposal, reply and repeat the deal to make sure how many items that you get in this format:
       ```
       Accept-Deal.
       Accpted-Deal:
            Items I get:
                Food: X,
                Firewood: Y,
                Water: Z.
            Items you get:
                Food: 3 - X,
                Firewood: 3 - Y,
                Water: 3 - Z.
        ```
     • If you reject the proposal, reply only with:  
       ```
       Reject-Deal.
       ```
       and submit a new proposal in the correct format.

4) End of Negotiation:
   - The negotiation ends immediately when one of you accepts a deal.  
   - After that, do not continue the conversation or propose any new deals.

**YOUR NEGOTIATION PRIORITIES AS PLAYER A**:
1) Highest Priority: {dialogue["participant_info"]["mturk_agent_1"]["value2issue"]["High"]}
2) Medium Priority: {dialogue["participant_info"]["mturk_agent_1"]["value2issue"]["Medium"]}
3) Lowest Priority: {dialogue["participant_info"]["mturk_agent_1"]["value2issue"]["Low"]}

Below is an example of how the conversation looks:
**EXAMPLE CONVERSATION**
Player A: Hello! I'm looking forward to making a deal. Let's find something that works for both of us.
Player B: Hello! I'm also interested in reaching an agreement. It's nice to meet you!
Player A: Nice to meet you too. So, which item do you need the most?
Player B: My top concern is having a strong fire, so Firewood is my priority. What about you?
Player A: I plan to do a lot of hiking, so I need plenty of Water. Some extra Food would be great too.
Player B: I'm okay with a bit of Water, but mostly I need Firewood. Let's see...
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
Submit-Deal:
   Items I get:
       Food: 1,
       Firewood: 3,
       Water: 1.
   Items you get:
       Food: 2,
       Firewood: 0,
       Water: 2.
Player A: Accept-Deal.
Accpted-Deal:
    Items I get:
       Food: 2,
       Firewood: 0,
       Water: 2.
   Items you get:
       Food: 1,
       Firewood: 3,
       Water: 1.

Now the real conversation begins!
"""
        else:
            prompt_str = f"""Imagine you're on a real camping trip! To make your stay more enjoyable, you need to gather three essential supplies: Food, Water, and Firewood. However, these resources are limited and must be shared with your campsite neighbor.

You and your neighbor will take turns negotiating how to divide these items. You have 15 total turns to reach a final agreement. During the conversation, speak naturally like a human and provide personal reasons or anecdotes to justify why you need specific items.

**CORE RULES**:
1) Available Items: There are exactly 3 units available for each item: Food, Firewood, and Water.

2) Full Allocation Required: You and your neighbor must fully allocate all items. No item can be left partially allocated or unallocated. The total amount you receive plus the amount your neighbor receives must be exactly 3 for each item. No fractional allocations allowed (e.g., 1.5 Food for you and 1.5 for your neighbor is not permitted).

3) Maximize Your Share: Try to obtain as many items as possible. The more units you secure, the higher your “score.” Remember to communicate naturally like a human.

4) Hidden Priorities: Each item has a different priority for you. Maximize your highest-priority item first, as it yields the most “score.” Do not reveal your item priorities to your neighbor.

**INSTRUCTIONS FOR SUBMITTING OR RESPONDING TO DEALS**:

1) Proposing a Final Deal:
   - At any point, you may propose a final deal. To do this, you MUST use the following format exactly:
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
     where X, Y, and Z are integers from 0 to 3. The total for each item must be exactly 3.

2) Deal Submission:
   - When proposing a deal, send a separate message containing only the deal in the required format. Once a deal is submitted, no any conversation is allowed. You can only submit deal, reject deal or accept deal following the correct format.

3) Responding to a Deal:
   - If you receive a deal (i.e., you see "Submit-Deal:" in the previous message):
     • You MUST verify that the total quantity for each item equals 3.  
     • If any item does not total 3, you must reject the proposal.  
     • If you accept the proposal, reply and repeat the deal to make sure how many items that you get in this format:
       ```
       Accept-Deal.
       Accpted-Deal:
            Items I get:
                Food: X,
                Firewood: Y,
                Water: Z.
            Items you get:
                Food: 3 - X,
                Firewood: 3 - Y,
                Water: 3 - Z.
        ```
     • If you reject the proposal, reply only with:  
       ```
       Reject-Deal.
       ```
       and submit a new proposal in the correct format.

4) End of Negotiation:
   - The negotiation ends immediately when one of you accepts a deal.  
   - After that, do not continue the conversation or propose any new deals.

**YOUR NEGOTIATION PRIORITIES AS PLAYER B**:
1) Highest Priority: {dialogue["participant_info"]["mturk_agent_2"]["value2issue"]["High"]}
2) Medium Priority: {dialogue["participant_info"]["mturk_agent_2"]["value2issue"]["Medium"]}
3) Lowest Priority: {dialogue["participant_info"]["mturk_agent_2"]["value2issue"]["Low"]}

Below is an example of how the conversation looks:
**EXAMPLE CONVERSATION**
Player A: Hello! I'm looking forward to making a deal. Let's find something that works for both of us.
Player B: Hello! I'm also interested in reaching an agreement. It's nice to meet you!
Player A: Nice to meet you too. So, which item do you need the most?
Player B: My top concern is having a strong fire, so Firewood is my priority. What about you?
Player A: I plan to do a lot of hiking, so I need plenty of Water. Some extra Food would be great too.
Player B: I'm okay with a bit of Water, but mostly I need Firewood. Let's see...
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
Submit-Deal:
   Items I get:
       Food: 1,
       Firewood: 3,
       Water: 1.
   Items you get:
       Food: 2,
       Firewood: 0,
       Water: 2.
Player A: Accept-Deal.
Accpted-Deal:
    Items I get:
       Food: 2,
       Firewood: 0,
       Water: 2.
   Items you get:
       Food: 1,
       Firewood: 3,
       Water: 1.

Now the real conversation begins!
"""

        # def filter_deals(msgs):
        #     return [
        #         m for m in msgs
        #         if not (
        #             m["text"].startswith("Submit-Deal")
        #             or m["text"].startswith("Reject-Deal")
        #             or m["text"].startswith("Accept-Deal")
        #         )
        #     ]

        chosen_msgs   = history + [true_reply]
        rejected_msgs = history + [gen]

        def to_role_content(msgs):
            out = []
            for m in msgs:
                role = "assistant" if m["id"] == gen["id"] else "user"
                out.append({"role": role, "content": m["text"]})
            return out

        dpo_examples.append({
            "prompt":   prompt_str,
            "chosen":   to_role_content(chosen_msgs),
            "rejected": to_role_content(rejected_msgs)
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(dpo_examples, fout, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    with open("../../data/split/casino_train.json",      "r", encoding="utf-8") as f1, \
         open("../results/turn_level_data_whole_train.json", "r", encoding="utf-8") as f2:
        d1 = json.load(f1)
        d2 = json.load(f2)

    build_dpo_data(d1, d2, "../data/dpo/train.json")