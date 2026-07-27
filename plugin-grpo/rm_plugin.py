from copy import deepcopy

import torch

from swift.llm import Template
from swift.plugin import rm_plugins
from swift.utils import to_device


class TestRewardModelPlugin:
    """
    Reward Model plugin for GRPO.

    It is assumed that self.model is
    automodel_fequenceclassification with num_labels=1.

    Reward = logits[:, 0].
    """

    def __init__(self, model, template):
        self.model = model
        self.template: Template = template

    @torch.inference_mode()
    def __call__(self, inputs, **kwargs):
        batched_inputs = [
            self.template.encode(deepcopy(example))
            for example in inputs
        ]

        reward_inputs = self.template.data_collator(batched_inputs)
        reward_inputs = to_device(reward_inputs, self.model.device)

        outputs = self.model(**reward_inputs)

        return outputs.logits[:, 0].float()


class SkyworkRewardV2Plugin:
    """
    Reward Model plugin for Skywork/Skywork-Reward-V2-Qwen3-1.7B.

    This plugin assumes that the reward model is loaded as a
    HuggingFace `AutoModelForSequenceClassification` with
    `num_labels=1`.

    The reward score is taken from the model's single output logit.

    Returns:
        torch.FloatTensor of shape (batch_size,)
    """

    def __init__(self, model, template):
        self.model = model
        self.template: Template = template

    @torch.inference_mode()
    def __call__(self, inputs, **kwargs):
        """
        Args:
            inputs:
                List of inference requests produced by GRPO.

        Returns:
            torch.FloatTensor containing one reward per sample.
        """
        # Encode every sample independently.
        encoded_inputs = [
            self.template.encode(deepcopy(example))
            for example in inputs
        ]

        # Build a batch and move it onto the reward model device.
        reward_inputs = self.template.data_collator(encoded_inputs)
        reward_inputs = to_device(reward_inputs, self.model.device)

        # Forward pass through the reward model.
        outputs = self.model(**reward_inputs)

        # Skywork-Reward-V2 produces a single scalar reward.
        rewards = outputs.logits.squeeze(-1)

        return rewards.float()


# Register plugin
rm_plugins["test_reward_model"] = TestRewardModelPlugin
rm_plugins["skywork_reward_v2"] = SkyworkRewardV2Plugin