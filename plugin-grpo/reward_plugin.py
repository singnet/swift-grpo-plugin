import re
from typing import List

from swift.rewards import ORM, AsyncORM, orms, rm_plugins
from latex2sympy2_extended import NormalizationConfig
from math_verify import parse, verify, LatexExtractionConfig


class AccuracyReward(ORM):
    """
    Verification of the accuracy of the answer (mathematical verification via math_verify)
    """

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        rewards = []

        for content, gold in zip(completions, solution):
            # Parsim the reference response
            gold_parsed = parse(
                gold,
                extraction_mode="first_match",
            )

            if not gold_parsed:
                # If the golden answer could not be parsed → skip the example / give 0
                rewards.append(0.0)
                continue

            # We will parse the response of the model with a stricter configuration
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",           # looking for boxed in the first place
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            try:
                correct = verify(gold_parsed, answer_parsed)
                reward = 1.0 if correct else 0.0
            except Exception as e:
                # print(f"Verification failed: {e}")
                reward = 0.0

            rewards.append(reward)

        return rewards


class TagCountReward(ORM):
    """
    Reward for the correct number and location of <think> and <answer> tags
    Gives 0.25 points for each correct tag/closing tag.
    """

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        def count_tags(text: str) -> float:
            score = 0.0
            if text.count("<think>\n") == 1:
                score += 0.25
            if text.count("\n</think>\n") == 1:
                score += 0.25
            if text.count("\n<answer>\n") == 1:
                score += 0.25
            if text.count("\n</answer>") == 1 or text.count("</answer>") == 1:
                score += 0.25
            return score

        return [count_tags(content) for content in completions]


class ReasoningStepsReward(ORM):
    """
    Reward for having explicit reasoning steps
    """

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        # A pattern for searching for signs of step-by-step reasoning
        pattern = r"(Step \d+:|^\d+\.\s| \d+\.\s|\n- |\n\* |First[,:\s]|Second[,:\s]|Next[,:\s]|Finally[,:\s])"

        rewards = []
        for content in completions:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            count = len(matches)
            # Normalization: 3 steps → 1.0, less → proportional
            reward = min(1.0, count / 3.0)
            rewards.append(reward)

        return rewards


orms['accuracy_openr1_reward'] = AccuracyReward
orms['tag_count_openr1_reward'] = TagCountReward
orms['reasoning_steps_openr1_reward'] = ReasoningStepsReward
