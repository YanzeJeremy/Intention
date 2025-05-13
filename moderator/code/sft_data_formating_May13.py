import json

# input_filename = "../../data/split/casino_train.json"
# with open(input_filename, 'r', encoding='utf-8') as f:
#     data = json.load(f)

# all_dialogue_results = []

# for idx, dialogue in enumerate(data):
#     botA_system_msg = f"""Imagine you're on a real camping trip! To make your stay more enjoyable, you need to gather three essential supplies: Food, Water, and Firewood. However, these resources are limited and must be shared with your campsite neighbor.

# You and your neighbor will take turns negotiating how to divide these items. You have 15 total turns to reach a final agreement. During the conversation, speak naturally like a human and provide personal reasons or anecdotes to justify why you need specific items.

# **CORE RULES**:
# 1) Available Items: There are exactly 3 units available for each item: Food, Firewood, and Water.

# 2) Full Allocation Required: You and your neighbor must fully allocate all items. No item can be left partially allocated or unallocated. The total amount you receive plus the amount your neighbor receives must be exactly 3 for each item. No fractional allocations allowed (e.g., 1.5 Food for you and 1.5 for your neighbor is not permitted).

# 3) Maximize Your Share: Try to obtain as many items as possible. The more units you secure, the higher your “score.” Remember to communicate naturally like a human.

# 4) Hidden Priorities: Each item has a different priority for you. Maximize your highest-priority item first, as it yields the most “score.” Do not reveal your item priorities to your neighbor.

# **INSTRUCTIONS FOR SUBMITTING OR RESPONDING TO DEALS**:

# 1) Proposing a Final Deal:
#    - At any point, you may propose a final deal. To do this, you MUST use the following format exactly:
#      ```
#      Submit-Deal:
#      Items I get:
#          Food: X,
#          Firewood: Y,
#          Water: Z.
#      Items you get:
#          Food: 3 - X,
#          Firewood: 3 - Y,
#          Water: 3 - Z.
#      ```
#      where X, Y, and Z are integers from 0 to 3. The total for each item must be exactly 3.

# 2) Deal Submission:
#    - When proposing a deal, send a separate message containing only the deal in the required format. Once a deal is submitted, no any conversation is allowed. You can only submit deal, reject deal or accept deal following the correct format.

# 3) Responding to a Deal:
#    - If you receive a deal (i.e., you see "Submit-Deal:" in the previous message):
#      • You MUST verify that the total quantity for each item equals 3.  
#      • If any item does not total 3, you must reject the proposal.  
#      • If you accept the proposal, reply and repeat the deal to make sure how many items that you get in this format:
#        ```
#        Accept-Deal.
#        Accpted-Deal:
#             Items I get:
#                 Food: X,
#                 Firewood: Y,
#                 Water: Z.
#             Items you get:
#                 Food: 3 - X,
#                 Firewood: 3 - Y,
#                 Water: 3 - Z.
#         ```
#      • If you reject the proposal, reply only with:  
#        ```
#        Reject-Deal.
#        ```
#        and submit a new proposal in the correct format.

# 4) End of Negotiation:
#    - The negotiation ends immediately when one of you accepts a deal.  
#    - After that, do not continue the conversation or propose any new deals.

# **YOUR NEGOTIATION PRIORITIES AS PLAYER A**:
# 1) Highest Priority: {dialogue["participant_info"]["mturk_agent_1"]["value2issue"]["High"]}
# 2) Medium Priority: {dialogue["participant_info"]["mturk_agent_1"]["value2issue"]["Medium"]}
# 3) Lowest Priority: {dialogue["participant_info"]["mturk_agent_1"]["value2issue"]["Low"]}

# Below is an example of how the conversation looks:
# **EXAMPLE CONVERSATION**
# Player A: Hello! I'm looking forward to making a deal. Let's find something that works for both of us.
# Player B: Hello! I'm also interested in reaching an agreement. It's nice to meet you!
# Player A: Nice to meet you too. So, which item do you need the most?
# Player B: My top concern is having a strong fire, so Firewood is my priority. What about you?
# Player A: I plan to do a lot of hiking, so I need plenty of Water. Some extra Food would be great too.
# Player B: I'm okay with a bit of Water, but mostly I need Firewood. Let's see...
# Player A: If I take 2 Water and you take 1, will you let me have all the Food?
# Player B: I still need some Food, but not a lot. I could take all the Firewood, 1 Food, and 1 Water.
# Player A: If I'm giving up that much Firewood, I'd like at least 1 unit of Firewood for chilly nights.
# Player B: I must insist on all 3 Firewood. But I'm letting you take more Water. Let's finalize something fair.
# Player A: Sounds good. I'll take 2 Water and 2 Food; you can have 1 Water, 1 Food, and 3 Firewood. Deal?
# Player B: Great deal!
# Player A: Submit-Deal:
#    Items I get:
#        Food: 3,
#        Firewood: 0,
#        Water: 2.
#    Items you get:
#        Food: 0,
#        Firewood: 3,
#        Water: 1.
# Player B: Reject-Deal.
# Submit-Deal:
#    Items I get:
#        Food: 1,
#        Firewood: 3,
#        Water: 1.
#    Items you get:
#        Food: 2,
#        Firewood: 0,
#        Water: 2.
# Player A: Accept-Deal.
# Accpted-Deal:
#     Items I get:
#        Food: 2,
#        Firewood: 0,
#        Water: 2.
#    Items you get:
#        Food: 1,
#        Firewood: 3,
#        Water: 1.

# Now the real conversation begins!
# """
#     botB_system_msg = f"""Imagine you're on a real camping trip! To make your stay more enjoyable, you need to gather three essential supplies: Food, Water, and Firewood. However, these resources are limited and must be shared with your campsite neighbor.

# You and your neighbor will take turns negotiating how to divide these items. You have 15 total turns to reach a final agreement. During the conversation, speak naturally like a human and provide personal reasons or anecdotes to justify why you need specific items.

# **CORE RULES**:
# 1) Available Items: There are exactly 3 units available for each item: Food, Firewood, and Water.

# 2) Full Allocation Required: You and your neighbor must fully allocate all items. No item can be left partially allocated or unallocated. The total amount you receive plus the amount your neighbor receives must be exactly 3 for each item. No fractional allocations allowed (e.g., 1.5 Food for you and 1.5 for your neighbor is not permitted).

# 3) Maximize Your Share: Try to obtain as many items as possible. The more units you secure, the higher your “score.” Remember to communicate naturally like a human.

# 4) Hidden Priorities: Each item has a different priority for you. Maximize your highest-priority item first, as it yields the most “score.” Do not reveal your item priorities to your neighbor.

# **INSTRUCTIONS FOR SUBMITTING OR RESPONDING TO DEALS**:

# 1) Proposing a Final Deal:
#    - At any point, you may propose a final deal. To do this, you MUST use the following format exactly:
#      ```
#      Submit-Deal:
#      Items I get:
#          Food: X,
#          Firewood: Y,
#          Water: Z.
#      Items you get:
#          Food: 3 - X,
#          Firewood: 3 - Y,
#          Water: 3 - Z.
#      ```
#      where X, Y, and Z are integers from 0 to 3. The total for each item must be exactly 3.

# 2) Deal Submission:
#    - When proposing a deal, send a separate message containing only the deal in the required format. Once a deal is submitted, no any conversation is allowed. You can only submit deal, reject deal or accept deal following the correct format.

# 3) Responding to a Deal:
#    - If you receive a deal (i.e., you see "Submit-Deal:" in the previous message):
#      • You MUST verify that the total quantity for each item equals 3.  
#      • If any item does not total 3, you must reject the proposal.  
#      • If you accept the proposal, reply and repeat the deal to make sure how many items that you get in this format:
#        ```
#        Accept-Deal.
#        Accpted-Deal:
#             Items I get:
#                 Food: X,
#                 Firewood: Y,
#                 Water: Z.
#             Items you get:
#                 Food: 3 - X,
#                 Firewood: 3 - Y,
#                 Water: 3 - Z.
#         ```
#      • If you reject the proposal, reply only with:  
#        ```
#        Reject-Deal.
#        ```
#        and submit a new proposal in the correct format.

# 4) End of Negotiation:
#    - The negotiation ends immediately when one of you accepts a deal.  
#    - After that, do not continue the conversation or propose any new deals.

# **YOUR NEGOTIATION PRIORITIES AS PLAYER B**:
# 1) Highest Priority: {dialogue["participant_info"]["mturk_agent_2"]["value2issue"]["High"]}
# 2) Medium Priority: {dialogue["participant_info"]["mturk_agent_2"]["value2issue"]["Medium"]}
# 3) Lowest Priority: {dialogue["participant_info"]["mturk_agent_2"]["value2issue"]["Low"]}

# Below is an example of how the conversation looks:
# **EXAMPLE CONVERSATION**
# Player A: Hello! I'm looking forward to making a deal. Let's find something that works for both of us.
# Player B: Hello! I'm also interested in reaching an agreement. It's nice to meet you!
# Player A: Nice to meet you too. So, which item do you need the most?
# Player B: My top concern is having a strong fire, so Firewood is my priority. What about you?
# Player A: I plan to do a lot of hiking, so I need plenty of Water. Some extra Food would be great too.
# Player B: I'm okay with a bit of Water, but mostly I need Firewood. Let's see...
# Player A: If I take 2 Water and you take 1, will you let me have all the Food?
# Player B: I still need some Food, but not a lot. I could take all the Firewood, 1 Food, and 1 Water.
# Player A: If I'm giving up that much Firewood, I'd like at least 1 unit of Firewood for chilly nights.
# Player B: I must insist on all 3 Firewood. But I'm letting you take more Water. Let's finalize something fair.
# Player A: Sounds good. I'll take 2 Water and 2 Food; you can have 1 Water, 1 Food, and 3 Firewood. Deal?
# Player B: Great deal!
# Player A: Submit-Deal:
#    Items I get:
#        Food: 3,
#        Firewood: 0,
#        Water: 2.
#    Items you get:
#        Food: 0,
#        Firewood: 3,
#        Water: 1.
# Player B: Reject-Deal.
# Submit-Deal:
#    Items I get:
#        Food: 1,
#        Firewood: 3,
#        Water: 1.
#    Items you get:
#        Food: 2,
#        Firewood: 0,
#        Water: 2.
# Player A: Accept-Deal.
# Accpted-Deal:
#     Items I get:
#        Food: 2,
#        Firewood: 0,
#        Water: 2.
#    Items you get:
#        Food: 1,
#        Firewood: 3,
#        Water: 1.

# Now the real conversation begins!
# """

# def format_content(chat):
#         if chat['text'] == "Submit-Deal":
#             return (
#                 f"Submit-Deal:\n"
#                 f"Items I get:\n"
#                 f"    Food: {chat['task_data']['issue2youget']['Food']},\n"
#                 f"    Firewood: {chat['task_data']['issue2youget']['Firewood']},\n"
#                 f"    Water: {chat['task_data']['issue2youget']['Water']}.\n"
#                 f"Items you get:\n"
#                 f"    Food: {chat['task_data']['issue2theyget']['Food']},\n"
#                 f"    Firewood: {chat['task_data']['issue2theyget']['Firewood']},\n"
#                 f"    Water: {chat['task_data']['issue2theyget']['Water']}."
#             )
#         elif chat['text'] in ("Reject-Deal",):
#             return "Reject-Deal."
#         elif chat['text'] in ("Accept-Deal",):
#             return (
#                 f"Accept-Deal.\n"
#                 f"Accpted-Deal:\n"
#                 f"    Items I get:\n"
#                 f"        Food: {last_chat['task_data']['issue2theyget']['Food']},\n"
#                 f"        Firewood: {last_chat['task_data']['issue2theyget']['Firewood']},\n"
#                 f"        Water: {last_chat['task_data']['issue2theyget']['Water']}.\n"
#                 f"    Items you get:\n"
#                 f"        Food: {last_chat['task_data']['issue2youget']['Food']},\n"
#                 f"        Firewood: {last_chat['task_data']['issue2youget']['Firewood']},\n"
#                 f"        Water: {last_chat['task_data']['issue2youget']['Water']}."
#             )
#         else:
#             return chat['text']

#     chat_logs = dialogue["chat_logs"]
#     last_chat = None

#     for turn_idx, chat in enumerate(chat_logs):
#         if chat['id'] == 'mturk_agent_1':
#             system_msg = {"role": "system", "content": botA_system_msg}
#             assistant_id, user_id = 'mturk_agent_1', 'mturk_agent_2'
#         else:
#             system_msg = {"role": "system", "content": botB_system_msg}
#             assistant_id, user_id = 'mturk_agent_2', 'mturk_agent_1'

#         if chat['id'] == assistant_id:
#             messages = [system_msg]
#             for prev in chat_logs[:turn_idx]:
#                 role = "assistant" if prev['id'] == assistant_id else "user"
#                 messages.append({
#                     "role": role,
#                     "content": format_content(prev)
#                 })

#             messages.append({
#                 "role": "assistant",
#                 "content": format_content(chat)
#             })

#             all_dialogue_results.append({
#                 "id": f"{idx}_{turn_idx}",
#                 "messages": messages
#             })

#         last_chat = chat

# output_filename = "../data/sft/train_1.json"
# with open(output_filename, "w", encoding="utf-8") as f:
#     json.dump(all_dialogue_results, f, indent=2, ensure_ascii=False)

# import json

def format_content(chat, last_chat=None):
    """
    根据 chat['text'] 和 chat['task_data'] 生成消息内容。
    对 Submit-Deal / Reject-Deal / Accept-Deal 三种特殊消息，
    按模板填充；否则原样返回 chat['text']。
    """
    txt = chat['text']
    if txt == "Submit-Deal":
        you = chat['task_data']['issue2youget']
        they = chat['task_data']['issue2theyget']
        return (
            "Submit-Deal:\n"
            f"Items I get:\n"
            f"    Food: {you['Food']},\n"
            f"    Firewood: {you['Firewood']},\n"
            f"    Water: {you['Water']}.\n"
            f"Items you get:\n"
            f"    Food: {they['Food']},\n"
            f"    Firewood: {they['Firewood']},\n"
            f"    Water: {they['Water']}."
        )
    elif txt == "Reject-Deal":
        return "Reject-Deal."
    elif txt == "Accept-Deal" and last_chat is not None:
        you = last_chat['task_data']['issue2theyget']
        they = last_chat['task_data']['issue2youget']
        return (
            "Accept-Deal.\n"
            "Accpted-Deal:\n"
            f"    Items I get:\n"
            f"        Food: {you['Food']},\n"
            f"        Firewood: {you['Firewood']},\n"
            f"        Water: {you['Water']}.\n"
            f"    Items you get:\n"
            f"        Food: {they['Food']},\n"
            f"        Firewood: {they['Firewood']},\n"
            f"        Water: {they['Water']}."
        )
    else:
        return txt


input_filename = "../../data/split/casino_train.json"
output_filename = "../data/sft/train.json"

with open(input_filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

all_dialogue_results = []

for idx, dialogue in enumerate(data):
    botA_system_msg = f"""Imagine you're on a real camping trip! To make your stay more enjoyable, you need to gather three essential supplies: Food, Water, and Firewood. However, these resources are limited and must be shared with your campsite neighbor.

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
    botB_system_msg = f"""Imagine you're on a real camping trip! To make your stay more enjoyable, you need to gather three essential supplies: Food, Water, and Firewood. However, these resources are limited and must be shared with your campsite neighbor.

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

    chat_logs = dialogue["chat_logs"]
    last_chat = None

    for turn_idx, chat in enumerate(chat_logs):
        if chat['id'] == 'mturk_agent_1':
            system_msg = {"role": "system", "content": botA_system_msg}
            assistant_id = 'mturk_agent_1'
        else:
            system_msg = {"role": "system", "content": botB_system_msg}
            assistant_id = 'mturk_agent_2'

        if chat['id'] == assistant_id:
            messages = [system_msg]
            for prev in chat_logs[:turn_idx]:
                role = "assistant" if prev['id'] == assistant_id else "user"
                messages.append({
                    "role": role,
                    "content": format_content(prev, last_chat)
                })

            messages.append({
                "role": "assistant",
                "content": format_content(chat, last_chat)
            })

            all_dialogue_results.append({
                "id": f"{idx}_{turn_idx}",
                "messages": messages
            })

        last_chat = chat

with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(all_dialogue_results, f, indent=2, ensure_ascii=False)
