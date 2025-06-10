import json
import re
import textwrap

with open("../results/negotiation_results_test_inital_message_Apr_22_original.json", "r", encoding="utf-8") as f:
    data = json.load(f)

deal_hdr_re = re.compile(r"accept[-\s]?deal\.", re.IGNORECASE)

def split_sections(text):
    try:
        mine_txt  = text.split("Items I get:")[1].split("Items you get:")[0]
        yours_txt = text.split("Items you get:")[1]
    except IndexError:
        return None, None

    def grab_counts(block):
        counts = {}
        for itm in priority_points:
            m = re.search(rf"{itm}:\s*(\d+)", block, flags=re.IGNORECASE)
            counts[itm] = int(m.group(1)) if m else None
        return counts
    return grab_counts(mine_txt), grab_counts(yours_txt)

def violates_rules(mine, yours):
    """True if any value missing, out of range, or not summing to 3."""
    for itm in priority_points:
        a, b = mine.get(itm), yours.get(itm)
        if a is None or b is None:          # missing field
            return True
        if not (0 <= a <= 3 and 0 <= b <= 3):
            return True
        if a + b != 3:                      # must total exactly 3
            return True
    return False


priority_points = {
    "Food": 5,
    "Firewood": 4,
    "Water": 3,
}

def parse_deal_block(text):
    """
    Given a block containing 'Items I get:', 
    returns a dict {item: count}.
    """
    items = {}
    for item in priority_points:
        m = re.search(rf"{item}:\s*(\d+)", text)
        items[item] = int(m.group(1)) if m else 0
    return items

total_score_sum = 0
num_with_deal = 0
num_no_deal = 0
total = 0
invalid_dialogues = 0
total_dialogues   = len(data)
for dialog in data:
    botb_conv = dialog.get("BotB", {}).get("conversation", [])
    total = total + len(botb_conv)
    assistant_msgs = [m for m in botb_conv if m.get("role") == "assistant"]

    if not assistant_msgs:
        # no assistant messages at all
        num_no_deal += 1
        continue

    last_content = assistant_msgs[-1]["content"]
    if "Items I get:" not in last_content:
        # last assistant message isn't a deal
        num_no_deal += 1
        continue

    # parse & score
    received = parse_deal_block(last_content)
    score = sum(priority_points[item] * count
                for item, count in received.items())

    total_score_sum += score
    num_with_deal += 1

    has_invalid = False
    for msg in botb_conv:
        if msg["role"] not in ("assistant", "user"):
            continue
        txt = msg["content"]
        if deal_hdr_re.search(txt) and "Items I get:" in txt:
            mine, yours = split_sections(txt)
            if mine is None or violates_rules(mine, yours):
                has_invalid = True
                break      # one violation flags the whole dialogue

    if has_invalid:
        invalid_dialogues += 1

avg_length = total/100
avg_score = total_score_sum / num_with_deal if num_with_deal else 0
validation_rate = invalid_dialogues / total_dialogues * 100

print("\n=== Summary ===")
print(f"Turn-to-Deals: {avg_length}")
print(f"Dialogues with deals: {num_with_deal}")
print(f"Dialogues without deals: {num_no_deal}")
print(f"Average score (over dialogues with deals): {avg_score:.2f}")
print(f"Validation error dialogues : {invalid_dialogues}")
print(f"Validation Error Rate      : {validation_rate:.1f}%")


input_filename = "../../data/split/casino_test_formatted.json"
with open(input_filename, 'r', encoding='utf-8') as f:
    data2 = json.load(f)
count = 0
real_points = []
real_points_sum = []
fake_points = []
total = 0
for m in data2:
    total = total+len(m["chat_logs"])
    real_points_sum.append({m["participant_info"]["mturk_agent_2"]["outcomes"]["points_scored"]})
real_points_sum_2 = [next(iter(x)) for x in real_points_sum]
print(f"whole real points are {sum(real_points_sum_2)}")
average = sum(real_points_sum_2) / len(real_points_sum_2)
print(f"whole average points are {average}")
print(total/100)

