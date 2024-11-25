from base import Agent
from execution_pipeline import main
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    BitsAndBytesConfig,
)
from utils import RAG
import torch

import ipdb


def get_bnb_config():
    """function for model quantization with bf16 setting"""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_compute_dtype = torch.bfloat16,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type = "n4f",
    )

    return quantization_config



class ClassificationAgent(Agent):
    """
    An agent that classifies text into one of the labels in the given label set.
    """
    def __init__(self, config: dict) -> None:
        """setting agent elements like `device`, `tokenizer`, `model`, `quantization_config`, 
            `rag_config(utils)`, `prompt_format`, `buffer`"""
        # device
        self.device = config.get("device")
        
        model_name = config.get("model_name")
        quantization_type = config.get("quantization_type")

        # tokenizer and model
        if config.get("tokenizer_name") is not None:
            tokenizer_name = config.get("tokenizer_name")    
        else:
            tokenizer_name = model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if quantization_type == "bf16":
            self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_type = get_bnb_config(), torch_dtype = torch.bfloat16)
        elif quantization_type == "fp16":
            # self.model = None
            pass
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # rag_config
        self.rag_config = {
            # I think we need to finetune embedding_model when we select the model to implement.
            "embedding_model": config.get("embedding_model") \
                if config.get("embedding_model") is not None else "BAAI/bge-base-en-v1.5",
            "seed": config.get("seed") if config.get("seed") is not None else 42,
            "top_k": config.get("top_k") if config.get("top_k") is not None else 5,
            "order": config.get("order") if config.get("order") is not None else "similar_at_top",
        }



    def __call__(
        self,
        label2desc: dict[str, str],
        text: str
    ) -> str:
        ipdb.set_trace()
        """
        Classify the text into one of the labels.

        Args:
            label2desc (dict[str, str]): A dictionary mapping each label to its description.
            text (str): The text to classify.

        Returns:
            str: The label (should be a key in label2desc) that the text is classified into.

        For example:
        label2desc = {
            "apple": "A fruit that is typically red, green, or yellow.",
            "banana": "A long curved fruit that grows in clusters and has soft pulpy flesh and yellow skin when ripe.",
            "cherry": "A small, round stone fruit that is typically bright or dark red.",
        }
        text = "The fruit is red and about the size of a tennis ball."
        label = "apple" (should be a key in label2desc, i.e., ["apple", "banana", "cherry"])
        """
        # TODO
        raise NotImplementedError

    def update(self, correctness: bool) -> bool:
        """
        Update your LLM agent based on the correctness of its own prediction at the current time step.

        Args:
            correctness (bool): Whether the prediction is correct.

        Returns:
            bool: Whether the prediction is correct.
        """
        # TODO
        raise NotImplementedError

class SQLGenerationAgent(Agent):
    """
    An agent that generates SQL code based on the given table schema and the user query.
    """
    def __init__(self, config: dict) -> None:
        """
        Initialize your LLM here
        """
        # TODO
        raise NotImplementedError

    def __call__(
        self,
        table_schema: str,
        user_query: str
    ) -> str:
        """
        Generate SQL code based on the given table schema and the user query.

        Args:
            table_schema (str): The table schema.
            user_query (str): The user query.

        Returns:
            str: The SQL code that the LLM generates.
        """
        # TODO: Note that your output should be a valid SQL code only.
        raise NotImplementedError

    def update(self, correctness: bool) -> bool:
        """
        Update your LLM agent based on the correctness of its own SQL    code at the current time step.
        """
        # TODO
        raise NotImplementedError
        
if __name__ == "__main__":
    from argparse import ArgumentParser
    from execution_pipeline import main

    parser = ArgumentParser()
    parser.add_argument('--bench_name', type=str, required=True)
    args = parser.parse_args()

    if args.bench_name.startswith("classification"):
        agent_name = ClassificationAgent
    elif args.bench_name.startswith("sql_generation"):
        agent_name = SQLGenerationAgent
    else:
        raise ValueError(f"Invalid benchmark name: {args.bench_name}")

    bench_cfg = {
        'bench_name': args.bench_name
    }
    config = {
        # TODO: specify your configs for the agent here
    }
    agent = agent_name(config)
    main(agent, bench_cfg)
