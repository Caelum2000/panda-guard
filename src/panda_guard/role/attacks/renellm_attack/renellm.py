#!/usr/bin/env python
# coding: utf-8

from typing import Dict, List
from dataclasses import dataclass, field
from panda_guard.role.attacks import BaseAttacker, BaseAttackerConfig


from panda_guard.llms import (
    create_llm,
    BaseLLMConfig,
    LLMGenerateConfig,
)

import argparse
import random
import logging

from panda_guard.role.attacks.renellm_attack.utils.prompt_rewrite_utils import shortenSentence, misrewriteSentence, changeOrder, addChar, languageMix, styleChange
from panda_guard.role.attacks.renellm_attack.utils.scenario_nest_utils import SCENARIOS
from panda_guard.role.attacks.renellm_attack.utils.harmful_classification_utils import harmful_classification


def get_fixed_args():
    args = argparse.Namespace(
        iter_max=20,
    )
    return args


@dataclass
class ReNeLLMAttackerConfig(BaseAttackerConfig):
    """
    Configuration for the ReNeLLM Attacker.

    :param attacker_cls: Class of the attacker, default is "ReNeLLMAttacker".  
    :param attacker_name: Name of the attacker. 
    :param rewrite_llm_config: Configuration of rewrite llm. 
    :param target_llm_config: Configuration of target llm. 
    :param judge_llm_config: Configuration of judge llm. 
    """

    attacker_cls: str = field(default="ColdAttacker")
    attacker_name: str = field(default=None)

    rewrite_llm_config: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    rewrite_llm_gen_config: LLMGenerateConfig = field(default=None)

    target_llm_config: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    target_llm_gen_config: LLMGenerateConfig = field(default=None)

    judge_llm_config: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    judge_llm_gen_config: LLMGenerateConfig = field(default=None)



class ReNeLLMAttacker(BaseAttacker):
    """
    ReNeLLM Attacker Implementation that substitutes the user message with a pre-formulated attack prompt.
    Reference: Peng Ding and Jun Kuang and Dan Ma and Xuezhi Cao and Yunsen Xian and Jiajun Chen and Shujian Huang, 2024, A Wolf in Sheep’s Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily, NAACL, 2024
    """

    def __init__(self, config: ReNeLLMAttackerConfig):
        super().__init__(config)

        self.config = config

        self.rewrite_llm = create_llm(config=config.rewrite_llm_config)
        self.target_llm = create_llm(config=config.target_llm_config)
        self.judge_llm = create_llm(config=config.judge_llm_config)

        self.args = get_fixed_args()

    def attack(self, messages: List[Dict[str, str]], **kwargs) -> List[Dict[str, str]]:
        """
        :param messages: List of messages in the conversation.  
        :param kwargs: Additional parameters for the attack, must include "request_reformulated". 
        :return: Prompts containing harmful attacks on the target, is of the form “role: user, content: xx”. 
        """

        question = messages[0]["content"]

        operations = [shortenSentence, misrewriteSentence, changeOrder, addChar, languageMix, styleChange]
        scenarios = SCENARIOS

        loop_count = 0

        harm_behavior = question
        temp_harm_behavior = question

        nested_prompt = ""

        # When jailbreak_label equals 1, or loop_count reaches the maximum number of attempts, self.args.iter_max, the loop ends
        while True:
            # 1. prompt rewriting
            while True:
                # randomly select the number of rewriting operations.
                n = random.randint(1, 6)
                # arrange the execution order randomly
                operation_indexes = random.sample(range(6), n)
                logging.info(f"The number of rewriting functions is: {n}")
                logging.info(f"The order of the rewriting operations is: {operation_indexes}\n")

                temp_rewrite_results = [['original prompt', temp_harm_behavior]]
                for index in operation_indexes:
                    logging.info(f"Excute function {index}: {operations[index].__name__}")
                    harm_behavior = operations[index](self.rewrite_llm, self.config.rewrite_llm_gen_config, harm_behavior)
                    logging.info(f"Rewritten prompt: {harm_behavior}\n")
                    temp_rewrite_results.append([operations[index].__name__, harm_behavior])

                # check if the prompt meets the requirements; if it does, break out of the loop.
                logging.info("******* Check if the prompt has been modified to be harmless *******")
                label = harmful_classification(self.judge_llm, self.config.judge_llm_gen_config, harm_behavior)
                logging.info(f"\nThe label given by {self.config.judge_llm_config.model_name} is: {label}\n")
                # Some LLMs will output 0/1, followed by an explanation, here we just take the first position of str
                try:
                    value = int(label[0])
                except ValueError:
                    continue

                if int(label[0]) == 1:
                    logging.info(f"******* Prompt Rewriting Done! *******\n")
                    break
                else:
                    logging.info(f"******* Prompt Rewriting Failed! *******\n")
                    harm_behavior = temp_harm_behavior
                    continue

            # 2. scenario nesting
            logging.info(f"******* Start Prompt Scenario Nesting! *******")
            logging.info(f"harm_behavior:\n{harm_behavior}\n")
            # randomly select a scenario
            scenario = random.choice(scenarios)
            nested_prompt = scenario.replace("<>", harm_behavior)
            logging.info(f"nested_prompt:\n{nested_prompt}\n")

            logging.info(f"******* Start Prompt Jailbreaking on {self.config.target_llm_config.model_name}! *******")

            test_nested_prompt = [{"role": "user", "content": nested_prompt}]
            attack_output = self.target_llm.generate(test_nested_prompt, self.config.target_llm_gen_config)

            attack_output = attack_output[-1]['content']

            logging.info(f"The output of the attacked model {self.config.target_llm_config.model_name} is:\n\n{attack_output}\n")

            loop_count += 1

            if "apologize" not in attack_output and "sorry" not in attack_output.lower():

                logging.info(
                    f"******* Determine if the attacked model's response to the nested prompt is harmful *******")
                jailbreak_label = harmful_classification(self.judge_llm, self.config.judge_llm_gen_config, attack_output)

            else:
                if loop_count < self.args.iter_max:
                    logging.info(f"\nJailbreaking Prompt Failed!\n")
                    harm_behavior = temp_harm_behavior
                    continue
                else:
                    logging.info(
                        f"\n******* Exceeded the maximum number of iterations {self.args.iter_max}, adopt the current round results and end the loop.*******\n")
                    break
            # Some LLMs will output 0/1, followed by an explanation, here we just take the first position of str
            try:
                value = int(jailbreak_label[0])
            except ValueError:
                if loop_count < self.args.iter_max:
                    logging.info(f"\nJailbreaking Prompt Failed!\n")
                    harm_behavior = temp_harm_behavior
                    continue
                else:
                    logging.info(
                        f"\n******* Exceeded the maximum number of iterations {self.args.iter_max}, adopt the current round results and end the loop.*******\n")
                    break

            if int(jailbreak_label[0]) == 1:
                logging.info(f"\n******* Jailbreaking Prompt Successful! *******\n")
                break
            else:
                if loop_count < self.args.iter_max:
                    logging.info(f"\nJailbreaking Prompt Failed!\n")
                    harm_behavior = temp_harm_behavior
                    continue
                else:
                    logging.info(
                        f"\n******* Exceeded the maximum number of iterations {self.args.iter_max}, adopt the current round results and end the loop.*******\n")
                    break

        messages = [{"role": "user", "content": nested_prompt}]

        return messages
