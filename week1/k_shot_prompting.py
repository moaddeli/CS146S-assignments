import os
from dotenv import load_dotenv
from ollama import chat

load_dotenv()

NUM_RUNS_TIMES = 5

YOUR_SYSTEM_PROMPT = """You are a strict formatter that only returns the exact transformed token requested, nothing else.

Rules:
- Always output exactly the transformed word, with no extra text, no explanation, no quotes, and no extra whitespace or newlines.
- Preserve the original letter casing and all non-letter characters (digits, punctuation, underscores) as they appear when reversed.
- Do not add punctuation or surrounding characters.

10-shot examples (input -> output):
Example 1
Input: Hello
Output: olleH

Example 2
Input: WORLD
Output: DLROW

Example 3
Input: camelCase
Output: esaClemac

Example 4
Input: OpenAI
Output: IAnepO

Example 5
Input: httpstatus
Output: sutatsptth

Example 6
Input: snake_case
Output: esac_ekans

Example 7
Input: Test123
Output: 321tseT

Example 8
Input: URL
Output: LRU

Example 9
Input: MixedCASE
Output: ESACdexiM

Example 10
Input: OneMore
Output: eroMenO

Now apply the same rule to any following user prompt: return only the single reversed word."""


USER_PROMPT = """
Reverse the order of letters in the following word. Only output the reversed word, no other text:

httpstatus
"""


EXPECTED_OUTPUT = "sutatsptth"

def test_your_prompt(system_prompt: str) -> bool:
    """Run the prompt up to NUM_RUNS_TIMES and return True if any output matches EXPECTED_OUTPUT.

    Prints "SUCCESS" when a match is found.
    """
    for idx in range(NUM_RUNS_TIMES):
        print(f"Running test {idx + 1} of {NUM_RUNS_TIMES}")
        response = chat(
            model="mistral-nemo:12b",
            #model="llama3.1:8b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPT},
            ],
            options={"temperature": 0.5},
        )
        output_text = response.message.content.strip()
        if output_text.strip() == EXPECTED_OUTPUT.strip():
            print("SUCCESS")
            return True
        else:
            print(f"Expected output: {EXPECTED_OUTPUT}")
            print(f"Actual output: {output_text}")
    return False

if __name__ == "__main__":
    test_your_prompt(YOUR_SYSTEM_PROMPT)