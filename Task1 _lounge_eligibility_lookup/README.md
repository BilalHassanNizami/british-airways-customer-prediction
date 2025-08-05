# British Airways – Task 1: Modeling Lounge Eligibility at Heathrow Terminal 3

## Overview

British Airways aims to anticipate future lounge demand at Heathrow Terminal 3 as part of its long-term planning. This project models lounge eligibility using flexible, scalable assumptions that can be applied to unknown or future flight schedules. The deliverables include a **lookup table** estimating lounge eligibility and a **written justification** explaining the logic and grouping methodology.

## 🧠 Objective

- Estimate what percentage of passengers across different flight categories will be eligible for each lounge tier.
- Build a reusable lookup table based on meaningful flight groupings (e.g., route type, time of day).
- Provide a written rationale that ensures the model is scalable, interpretable, and aligned with real-world planning needs.

## Lounge Tiers Modeled

| Tier | Lounge Type       | Description                                     |
|------|--------------------|-------------------------------------------------|
| 1    | Concorde Room      | Hypothetical future Tier 1 space (no current lounge) |
| 2    | First Lounge       | For first class and high-tier status travelers |
| 3    | Club Lounge        | For business class and mid-tier frequent flyers |

> Note: Tier 1 (Concorde Room) does not currently exist in Terminal 3, but is modeled hypothetically to help BA assess if a future investment might be needed.

## Methodology

1. **Flight Grouping Criteria**
   - Flights were categorized by:
     - **Time of Day** (AM, PM, Evening)
     - **Route Type** (Short-haul, Mid-haul, Long-haul)
   - Example:
     - Flight 456 to Madrid at 06:15 → Short-haul AM
     - Flight 664 to Larnaca at 07:00 → Mid-haul AM

2. **Eligibility Assumptions**
   - For each group, logical assumptions were applied based on typical customer mix and travel class:
     - Short-haul AM: 5% Tier 1, 20% Tier 2, 60% Tier 3
     - Mid-haul AM: 7% Tier 1, 25% Tier 2, 65% Tier 3
     - *(More groupings in full table)*

3. **Reusable Lookup Table**
   - Outputs estimated % of passengers eligible for each tier based on category.
   - Allows BA to apply same logic to new flight schedules without detailed passenger-level data.

4. **Justification Sheet**
   - Documents:
     - How flights were grouped
     - Why those groupings were chosen
     - Key assumptions behind tier estimates
     - How the model remains scalable for future scenarios

## Files Included

- `lounge_eligibility_and_justification.xlsx`
  - **Sheet 1 – Lookup Table:**  
    Lounge eligibility estimates by flight category
  - **Sheet 2 – Justification:**  
    Responses to four strategic questions explaining the modeling logic

## Example from the Model

| Grouping   | Destination | Tier 1 % | Tier 2 % | Tier 3 % |
|------------|-------------|----------|----------|----------|
| Short-Haul | Madrid      | 5%       | 20%      | 60%      |
| Long-Haul  | Larnaca     | 7%       | 25%      | 65%      |

## Key Takeaways

- A scalable and flexible model for forecasting lounge demand.
- Lookup table supports rapid assessment of future flying schedules.
- Enables BA to make informed investment decisions in lounge capacity.

## Author

Bilal Hassan Nizami  
*Task completed as part of the British Airways Data Analyst/Data Scientist Virtual Internship*
