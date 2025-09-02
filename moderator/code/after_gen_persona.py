import re
import json

deal_pattern = re.compile(
    r"Submit-Deal:\s*"
    r"Items\s+I\s+get:\s*"
    r"Food:\s*(\d+),\s*Firewood:\s*(\d+),\s*Water:\s*(\d+)\.\s*"
    r"Items\s+you\s+get:\s*"
    r"Food:\s*(\d+),\s*Firewood:\s*(\d+),\s*Water:\s*(\d+)\.",
    re.IGNORECASE | re.DOTALL
)

deal_block_pattern = re.compile(
    r"(Submit-Deal:.*?)(?=\n\n|\Z)",
    re.IGNORECASE | re.DOTALL
)

def extract_deals_from_messages(messages):
    extracted = []
    for turn in messages:
        text = turn["content"]
        blocks = deal_block_pattern.findall(text)
        for block in blocks:
            m = deal_pattern.search(block)
            if m:
                g = m.groups()
                parsed = {
                    "my_food": int(g[0]),
                    "my_firewood": int(g[1]),
                    "my_water": int(g[2]),
                    "your_food": int(g[3]),
                    "your_firewood": int(g[4]),
                    "your_water": int(g[5])
                }
            else:
                parsed = None
            extracted.append({
                "raw_deal": block.strip(),
                "parsed": parsed
            })
    return extracted

def is_valid_deal(deal):
    if deal is None:
        return False
    return (
        (deal["my_food"] + deal["your_food"] == 3) and
        (deal["my_firewood"] + deal["your_firewood"] == 3) and
        (deal["my_water"] + deal["your_water"] == 3)
    )

def deals_are_symmetric(dealA, dealB):
    if dealA is None or dealB is None:
        return False
    return (
        dealA["my_food"] == dealB["your_food"] and
        dealA["my_firewood"] == dealB["your_firewood"] and
        dealA["my_water"] == dealB["your_water"] and
        dealA["your_food"] == dealB["my_food"] and
        dealA["your_firewood"] == dealB["my_firewood"] and
        dealA["your_water"] == dealB["my_water"]
    )

def main():
    with open("../results/negotiation_results_test_all_messages.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    with open("output_persona_log.txt", "w", encoding="utf-8") as out:
        for record in data:
            dialogue_idx = record["dialogue_index"]
            botA = record["BotA"]

            print(f"--- Dialogue Index: {dialogue_idx} ---", file=out)

            first_msg = botA["conversation"][0]["content"]
            persona_A_text = None
            persona_B_text = None

            if "Agent A's persona:" in first_msg and "Agent B's persona:" in first_msg:
                print("🧠 Persona conditioning detected in prompt.", file=out)
                lines = first_msg.split("\n")
                for line in lines:
                    if "Agent A's persona:" in line:
                        persona_A_text = line.strip().replace("Agent A's persona:", "").strip()
                    elif "Agent B's persona:" in line:
                        persona_B_text = line.strip().replace("Agent B's persona:", "").strip()
                if persona_A_text:
                    print(f"  👤 Agent A Persona: {persona_A_text}", file=out)
                if persona_B_text:
                    print(f"  👤 Agent B Persona: {persona_B_text}", file=out)
            else:
                print("⚠️ No persona conditioning found in prompt.", file=out)

            deals_A = extract_deals_from_messages(botA["conversation"])
            print("\nAll Deals proposed by BotA (parsed):", file=out)
            for d in deals_A:
                if d["parsed"]:
                    print(f"    -> {d['parsed']}", file=out)

            # === Unique Deals Only ===
            unique_deals_set = set()
            for d in deals_A:
                parsed = d["parsed"]
                if parsed:
                    unique_deals_set.add(tuple(sorted(parsed.items())))
            unique_deals = [dict(deal) for deal in unique_deals_set]

            print("\n✅ Unique Deals proposed by BotA:", file=out)
            for deal in unique_deals:
                print(f"    🔄 {deal}", file=out)

            # === Final two deals ===
            last_two_A = deals_A[-2:] if len(deals_A) >= 2 else deals_A
            print("\nLast two deals for BotA:", file=out)
            for d in last_two_A:
                print(f"  -> {d['parsed']}", file=out)

            if len(last_two_A) >= 2:
                final_A_2 = last_two_A[-2]["parsed"]
                final_A = last_two_A[-1]["parsed"]
            elif len(last_two_A) == 1:
                final_A_2 = None
                final_A = last_two_A[0]["parsed"]
            else:
                final_A_2 = None
                final_A = None

            print("\nValidity checks for the last two deals:", file=out)
            if final_A:
                print(f"  Final A valid: {is_valid_deal(final_A)}", file=out)
            if final_A_2:
                print(f"  Final A-2 valid: {is_valid_deal(final_A_2)}", file=out)

            if final_A and final_A_2:
                if deals_are_symmetric(final_A, final_A_2):
                    print("Comparison (last two deals): They describe the SAME final outcome.", file=out)
                else:
                    print("Comparison (last two deals): They differ.", file=out)
            else:
                print("Comparison: Not enough valid deals to compare.", file=out)

            print(f"\ndeal_accepted: {record['deal_accepted']}", file=out)
            print("=== End of Dialogue ===\n", file=out)

if __name__ == "__main__":
    main()
