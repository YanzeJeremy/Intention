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
    """
    Given a list of messages (each containing 'role' and 'content'),
    find all blocks that start with 'Submit-Deal:'.
    
    Returns a list of dictionaries for each deal found, containing:
      - 'raw_deal': the raw text of the deal
      - 'parsed': a dict with numeric fields if we can parse them via deal_pattern (or None otherwise).
    """
    extracted = []
    for turn in messages:
        text = turn["content"]
        
        # Find each block that starts with 'Submit-Deal:'
        blocks = deal_block_pattern.findall(text)
        for block in blocks:
            # Attempt a more detailed parse with deal_pattern
            m = deal_pattern.search(block)
            if m:
                # Each group is a numeric value in order: 
                # my_food, my_firewood, my_water, your_food, your_firewood, your_water
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
    """
    Check if the deal satisfies the condition that (my_* + your_*) == 3 for 
    Food, Firewood, and Water.
    """
    if deal is None:
        return False
    
    return (
        (deal["my_food"] + deal["your_food"] == 3) and
        (deal["my_firewood"] + deal["your_firewood"] == 3) and
        (deal["my_water"] + deal["your_water"] == 3)
    )

def deals_are_symmetric(dealA, dealB):
    """
    Check if dealA and dealB describe the same final allocation,
    given that 'my_*' in dealA corresponds to 'your_*' in dealB (and vice versa).
    Returns True if they match, False otherwise.
    """
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
    with open("negotiation_results_test_all.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Open the output file for writing:
    with open("output_last2_log.txt", "w", encoding="utf-8") as out:
        for record in data:
            dialogue_idx = record["dialogue_index"]
            botA = record["BotA"]

            print(f"--- Dialogue Index: {dialogue_idx} ---", file=out)

            # Extract any deals proposed in the conversation
            deals_A = extract_deals_from_messages(botA["conversation"])
            
            print("All Deals proposed by BotA (parsed):", file=out)
            for d in deals_A:
                if d["parsed"]:
                    print(f"    -> {d['parsed']}", file=out)

            # Keep only the last two deals from BotA (if they exist)
            last_two_A = deals_A[-2:] if len(deals_A) >= 2 else deals_A

            print("\nLast two deals for BotA:", file=out)
            for d in last_two_A:
                print(f"  -> {d['parsed']}", file=out)

            # Assign final deals
            if len(last_two_A) >= 2:
                final_A_2 = last_two_A[-2]["parsed"]
                final_A   = last_two_A[-1]["parsed"]
            elif len(last_two_A) == 1:
                final_A_2 = None
                final_A   = last_two_A[0]["parsed"]
            else:
                final_A_2 = None
                final_A   = None
            
            # 1) Check validity of the deals
            print("\nValidity checks for the last two deals:", file=out)
            if final_A:
                print(f"  Final A valid: {is_valid_deal(final_A)}", file=out)
            if final_A_2:
                print(f"  Final A-2 valid: {is_valid_deal(final_A_2)}", file=out)

            # 2) Compare them with deals_are_symmetric (only if both exist)
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
