#!/usr/bin/python3

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .question_model import QAgent

import random
import json
import re


class QuestioningAgent:
    r"""Agent responsible for generating and validating questions"""

    # =====================================================================
    # INIT
    # =====================================================================

    def __init__(self, verbose: bool = False, **kwargs):
        self.verbose = verbose
        self.agent = QAgent(**kwargs)

    # =====================================================================
    # ICL BUILDER
    # =====================================================================

    def build_inc_samples(self, inc_samples: List[Dict[str, str]], topic: str) -> str:
        if not inc_samples:
            return ""

        fmt = (
            "EXAMPLE:\n"
            "{{\n"
            '  "topic": "{}",\n'
            '  "question": "{}",\n'
            '  "choices": ["{}", "{}", "{}", "{}"],\n'
            '  "answer": "{}",\n'
            '  "explanation": "{}"\n'
            "}}\n\n"
        )

        output = ""
        for sample in inc_samples:
            output += fmt.format(
                topic,
                sample.get("question", ""),
                *sample.get("choices", [""] * 4),
                sample.get("answer", ""),
                sample.get("explanation", ""),
            )

        return output.strip()

    # =====================================================================
    # PROMPT BUILDER
    # =====================================================================

    def build_prompt(
        self,
        topic: str,
        wadvsys: bool = True,
        wicl: bool = True,
        inc_samples: List[Dict[str, str]] | None = None,
    ) -> Tuple[str, str]:

        if wadvsys:
            sys_prompt = """
You are an expert-level examiner designing extremely difficult MCQs.

You MUST output STRICTLY VALID JSON.

CRITICAL RULES:
1. Output ONE JSON object only.
2. No markdown or commentary.
3. No physical newline characters inside JSON string values.
   Use backslash followed by n if required.
4. No trailing commas.
5. Exactly four choices labeled A), B), C), D).
6. Only one correct answer.
7. Explanation under 100 words.
8. Do not reveal chain-of-thought reasoning.

SYLLOGISMS FORMAT:
{
  "topic": "Logical Reasoning/Syllogisms",
  "question": "Use single-line formatting only.",
  "choices": ["A) ...","B) ...","C) ...","D) ..."],
  "answer": "A/B/C/D",
  "explanation": "..."
}

OTHER TOPICS FORMAT:
{
  "topic": "...",
  "question": "...",
  "choices": ["A) ...","B) ...","C) ...","D) ..."],
  "answer": "A/B/C/D",
  "explanation": "..."
}
"""
        else:
            sys_prompt = "Generate an extremely difficult MCQ in strict JSON format."

        correct_option = random.choice(["A", "B", "C", "D"])
        distractors = ", ".join([x for x in ["A", "B", "C", "D"] if x != correct_option])

        icl_block = self.build_inc_samples(inc_samples, topic) if (wicl and inc_samples) else ""

        user_prompt = (
            f"Generate an EXTREMELY DIFFICULT MCQ on topic: {topic}.\n\n"
            f"Ensure option {correct_option} is the ONLY correct answer.\n"
            f"Options {distractors} must be plausible distractors.\n\n"
            f"{icl_block}\n"
            "Respond ONLY with valid JSON."
        )

        return user_prompt, sys_prompt

    # =====================================================================
    # GENERATION
    # =====================================================================

    def generate_question(
        self,
        topic: Tuple[str, str] | List[Tuple[str, str]],
        wadvsys: bool,
        wicl: bool,
        inc_samples: Dict[str, List[Dict[str, str]]] | None,
        **gen_kwargs,
    ) -> Tuple[List[str], int | None, float | None]:

        if isinstance(topic, list):
            prompts = []
            for t in topic:
                p, sp = self.build_prompt(
                    f"{t[0]}/{t[1]}",
                    wadvsys,
                    wicl,
                    inc_samples[t[1]] if inc_samples else None,
                )
                prompts.append(p)
            system_prompt = sp
        else:
            p, system_prompt = self.build_prompt(
                f"{topic[0]}/{topic[1]}",
                wadvsys,
                wicl,
                inc_samples[topic[1]] if inc_samples else None,
            )
            prompts = [p]

        resp, tl, gt = self.agent.generate_response(prompts, system_prompt, **gen_kwargs)

        if isinstance(resp, str):
            return [resp], tl, gt
        if isinstance(resp, list):
            return resp, tl, gt

        return [], tl, gt

    # =====================================================================
    # BATCH GENERATION
    # =====================================================================

    def generate_batches(
        self,
        num_questions: int,
        topics: Dict[str, List[str]],
        batch_size: int = 5,
        wadvsys: bool = True,
        wicl: bool = True,
        inc_samples: Dict[str, List[Dict[str, str]]] | None = None,
        **kwargs,
    ):

        extended_topics = self.populate_topics(topics, num_questions)

        questions = []
        tls = []
        gts = []

        total_batches = (len(extended_topics) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="STEPS: ")

        for i in range(0, len(extended_topics), batch_size):
            batch = extended_topics[i : i + batch_size]

            q, tl, gt = self.generate_question(
                batch, wadvsys, wicl, inc_samples, **kwargs
            )

            questions.extend(q)
            tls.append(tl)
            gts.append(gt)
            pbar.update(1)

        pbar.close()
        return questions, tls, gts

    # =====================================================================
    # SANITATION
    # =====================================================================

    def _sanitize_json(self, text: str) -> str:
        text = re.sub(r"```json\s*|```", "", text).strip()

        string_pattern = r'("(?:\\.|[^"\\])*")'

        def sanitize_match(m):
            literal = m.group(0)
            inner = literal[1:-1]

            inner = inner.replace("\r\n", "\\n").replace("\n", "\\n")
            inner = re.sub(r'(?<!\\)"', r'\\"', inner)

            return '"' + inner + '"'

        return re.sub(string_pattern, sanitize_match, text, flags=re.DOTALL)

    # =====================================================================
    # VALIDATION
    # =====================================================================

    def count_tokens_q(self, text: str) -> int:
        return len(self.agent.tokenizer.encode(text, add_special_tokens=False))

    def filter_questions(self, questions):

        def basic_checks(q_dict):

            if not all(k in q_dict for k in ["topic", "choices", "answer"]):
                return False

            if not (isinstance(q_dict["choices"], list) and len(q_dict["choices"]) == 4):
                return False

            # Normalize answer
            ans = str(q_dict["answer"]).strip().upper().replace(")", "")
            if ans not in ["A", "B", "C", "D"]:
                return False
            q_dict["answer"] = ans

            if "question" not in q_dict:
                return False

            content = (
                [q_dict["topic"], q_dict["question"]]
                + q_dict["choices"]
                + [q_dict["answer"]]
            )

            q_tokens = sum(self.count_tokens_q(str(x)) for x in content)
            e_tokens = self.count_tokens_q(q_dict.get("explanation", ""))

            return q_tokens <= 220 and (q_tokens + e_tokens) <= 1024

        valid = []

        for i, q in enumerate(questions):
            try:
                clean = self._sanitize_json(q) if isinstance(q, str) else q
                parsed = json.loads(clean) if isinstance(clean, str) else clean

                if isinstance(parsed, dict) and basic_checks(parsed):
                    valid.append(parsed)
                elif self.verbose:
                    print(f"Index {i}: JSON parsed but failed validation.")

            except Exception as e:
                if self.verbose:
                    print(f"Skipping index {i}: {e}")

        return valid

    # =====================================================================
    # UTILITIES
    # =====================================================================

    def save_questions(self, questions: Any, file_path: str | Path):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(questions, f, indent=4)

    def populate_topics(self, topics, num_questions):
        all_subtopics = [(t, st) for t, subs in topics.items() for st in subs]
        return random.choices(all_subtopics, k=num_questions)

    @staticmethod
    def load_icl_samples(file_path: str | Path):
        with open(file_path, "r") as f:
            return json.load(f)


# Example usage
if __name__ == "__main__":
    import argparse
    import yaml

    # ++++++++++++++++++++++++++
    # Run: python -m agents.question_agent --num_questions 20 --output_file outputs/questions.json --batch_size 5 --verbose
    # ++++++++++++++++++++++++++

    argparser = argparse.ArgumentParser(
        description="Generate questions using the QuestioningAgent."
    )
    argparser.add_argument(
        "--num_questions",
        type=int,
        default=10,
        help="Total number of questions to generate.",
    )
    argparser.add_argument(
        "--output_file",
        type=str,
        default="outputs/questions.json",
        help="Output file name to save the generated questions.",
    )
    argparser.add_argument(
        "--batch_size", type=int, default=5, help="Batch size for generating questions."
    )
    argparser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output for debugging."
    )
    args = argparser.parse_args()

    inc_samples = QuestioningAgent.load_icl_samples("assets/topics_example.json")

    # Load topics.json file.
    with open("assets/topics.json") as f:
        topics = json.load(f)

    # Pass verbosity into the agent
    agent = QuestioningAgent(verbose=args.verbose)
    # gen_kwargs = {"tgps_show": True, "max_new_tokens": 1024, "temperature": 0.1, "top_p": 0.9, "do_sample": True}
    gen_kwargs = {"tgps_show": True}
    with open("qgen.yaml", "r") as f:
        gen_kwargs.update(yaml.safe_load(f))

    question, tls, gts = agent.generate_batches(
        num_questions=args.num_questions,
        topics=topics,
        batch_size=args.batch_size,
        wadvsys=True,
        wicl=True,
        inc_samples=inc_samples,
        **gen_kwargs,
    )
    print(f"Generated {len(question)} questions!")
    if args.verbose:
        for q in question:
            print(q, flush=True)
        print("\n" + "=" * 50 + "\n\n")
        if gen_kwargs.get("tgps_show", False):
            print("Time taken per batch generation:", gts)
            print("Tokens generated per batch:", tls)
            if sum([g for g in gts if g]):
                total_time = sum([g for g in gts if g])
                total_tokens = sum([t for t in tls if t])
                print(
                    f"Total Time Taken: {total_time:.3f} seconds; Total Tokens: {total_tokens}; TGPS: {total_tokens/total_time if total_time>0 else 0:.3f} tokens/sec\n\n"
                )
        print("\n" + "+" * 50 + "\n")

    # check if question is JSON format
    ques = []
    for q in question:
        try:
            json.loads(q)
            ques.append(q)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON format in question: {q}\nError: {e}")
            # attempt simple extraction via the agent (best-effort fallback)
            prompt = (
                "Extract ONLY the topic, question, choices, answer, and explanation while discarding the rest.\n"
                "Also please remove JSON code block text with backticks like ```json and ```.\n\n"
                "CRITICAL: Replace all physical newlines inside string values with the character sequence '\\n'.\n"
                "String:\n"
                "{}\n\n"
                "Given Format (Just for your knowledge):\n"
                "{{\n"
                '  "topic": "...",\n'
                '  "question": "...",\n'
                '  "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
                '  "answer": "Only the option letter (A, B, C, or D)",\n'
                '  "explanation": "..."\n'
                "}}"
            )
            q_fixed = agent.agent.generate_response(
                prompt.format(q),
                "You are an expert JSON extractor.",
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False,
            )
            # if extractor returns tuple-like, normalize safely
            if isinstance(q_fixed, tuple):
                q_fixed = q_fixed[0] if q_fixed else ""
            ques.append(q_fixed)

    # Save the questions for later analysis
    agent.save_questions(ques, args.output_file)
    filtered_file_name = args.output_file.replace("questions.json", "filtered_questions.json")
    agent.save_questions(agent.filter_questions(ques), filtered_file_name)
    print(f"Saved to {args.output_file}!")
