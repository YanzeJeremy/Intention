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
        role = turn["role"]
        
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
                "role":role,
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
    list_correct = []
    reached_deal_A = []
    reached_deal_B = []
    with open("../results/negotiation_results_test_inital_message_march_Apr_10_no_persona.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Open the output file for writing:
    with open("../results/summary_initial_messages_Apr_10_no_persona.txt", "w", encoding="utf-8") as out:
        for record in data:
            dialogue_idx = record["dialogue_index"]
            botA = record["BotA"]

            print(f"--- Dialogue Index: {dialogue_idx} ---", file=out)

            # Extract any deals proposed in the conversation
            deals_A = extract_deals_from_messages(botA["conversation"])

            print("All Deals proposed by BotA (parsed):", file=out)
            for d in deals_A:
                if d["parsed"] and d['role']:
                    print(f"{d['role']} -> {d['parsed']}", file=out)

            # Keep only the last two deals from BotA (if they exist)
            last_two_A = deals_A[-2:] if len(deals_A) >= 2 else deals_A

            print("\nLast two deals for BotA:", file=out)
            for d in last_two_A:
                print(f"{d['role']}-> {d['parsed']}", file=out)

            # Assign final deals
            if len(last_two_A) >= 2:
                role_A_2 = last_two_A[-2]["role"]
                role_A = last_two_A[-1]["role"]
                final_A_2 = last_two_A[-2]["parsed"]
                final_A   = last_two_A[-1]["parsed"]
            elif len(last_two_A) == 1:
                final_A_2 = None
                final_A   = last_two_A[0]["parsed"]
            else:
                final_A_2 = None
                final_A   = None

            # 1) Check validity of the deals
            valid_final_A = is_valid_deal(final_A)
            valid_final_A_2 = is_valid_deal(final_A_2)

            print("\nValidity checks for the last two deals:", file=out)
            if final_A:
                print(f"  Final A valid: {valid_final_A}", file=out)
            if final_A_2:
                print(f"  Final A-2 valid: {valid_final_A_2}", file=out)

            # 2) Compare them with deals_are_symmetric (only if both exist)
            if final_A and final_A_2:
                symmetric = deals_are_symmetric(final_A, final_A_2)
                if symmetric:
                    print("Comparison (last two deals): They describe the SAME final outcome.", file=out)
                else:
                    print("Comparison (last two deals): They differ.", file=out)

                if valid_final_A and valid_final_A_2 and symmetric:
                    list_correct.append(dialogue_idx)
                    reached_deal_A.append({'role':role_A, 'deal':final_A})
                    print(f"Dialogue index with symmetric and valid final deals: {dialogue_idx}", file=out)

            else:
                print("Comparison: Not enough valid deals to compare.", file=out)

            print(f"\ndeal_accepted: {record['deal_accepted']}", file=out)
            print("=== End of Dialogue ===\n", file=out)
    
    input_filename = "../../data/split/casino_test_formatted.json"
    with open(input_filename, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    count = 0
    real_points = []
    real_points_sum = []
    fake_points = []
    for m in data2:
        real_points_sum.append({m["participant_info"]["mturk_agent_2"]["outcomes"]["points_scored"]})
    # real_points_sum_2 = [next(iter(x)) for x in real_points_sum]
    # print(f"whole real points are {sum(real_points_sum_2)}")
    # average = sum(real_points_sum_2) / len(real_points_sum_2)
    # print(f"whole average points are {average}")
    for i in range(len(data2)):
        High = {data2[i]["participant_info"]["mturk_agent_1"]["value2issue"]["High"]}
        Medium = {data2[i]["participant_info"]["mturk_agent_1"]["value2issue"]["Medium"]}
        Low = {data2[i]["participant_info"]["mturk_agent_1"]["value2issue"]["Low"]}
        real_points.append({data2[i]["participant_info"]["mturk_agent_1"]["outcomes"]["points_scored"]})
        if reached_deal_A[count]['role'] == 'assistant':
            Food = int(reached_deal_A[count]['deal']['my_food'])
            Water = int(reached_deal_A[count]['deal']['my_water'])
            Firewood = int(reached_deal_A[count]['deal']['my_firewood'])
            
            item_quantities = {
                'Food': Food,
                'Water': Water,
                'Firewood': Firewood
            }

            fake_point = 0
            for item, qty in item_quantities.items():
                if item in High:
                    fake_point += 5 * qty
                elif item in Medium:
                    fake_point += 4 * qty
                elif item in Low:
                    fake_point += 3 * qty

            fake_points.append(fake_point)
        else:
            Food = int(reached_deal_A[count]['deal']['your_food'])
            Water = int(reached_deal_A[count]['deal']['your_water'])
            Firewood = int(reached_deal_A[count]['deal']['your_firewood'])
            
            item_quantities = {
                'Food': Food,
                'Water': Water,
                'Firewood': Firewood
            }

            # Compute fake_points
            fake_point = 0
            for item, qty in item_quantities.items():
                if item in High:
                    fake_point += 5 * qty
                elif item in Medium:
                    fake_point += 4 * qty
                elif item in Low:
                    fake_point += 3 * qty

            fake_points.append(fake_point)
        count = count+1

        real_points_2 = [next(iter(x)) for x in real_points]
    print(f"for those correct, the sum real points are {sum(real_points_2)}")
    average = sum(real_points_2) / len(real_points_2)
    print(f"for those correct, the average real points are {average}")
    print(f"for those correct, the sum fake points are {sum(fake_points)}")
    average_2 = sum(fake_points) / len(fake_points)
    print(f"for those correct, the average fake points are {average_2}")
if __name__ == "__main__":
    main()
