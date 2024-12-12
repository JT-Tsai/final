from base import Agent
from execution_pipeline import main
from colorama import Fore, Style
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    BitsAndBytesConfig,
)
from utils import RAG, strip_all_lines
import torch
import re
import random
import copy

# import ipdb


def get_bnb_config(bits_8 = False, bits_4 = False):
    """8bits Q"""
    if bits_8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit = True,
            llm_int8_has_fp16_weight=False
            # llm_int8_has_fp16_weight=True
        )
    """4bits Q"""
    if bits_4: 
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype="float16",
        )

    return quantization_config


class ClassificationAgent(Agent):
    """
    An agent that classifies text into one of the labels in the given label set.
    """
    def __init__(self, config: dict) -> None:
        """setting agent elements like `device`, `tokenizer`, `model`, `quantization_config`, 
            `rag_config(utils)`, `prompt_format`, `buffer`"""
        
        super().__init__(config)
        
        # device
        self.device = config.get("device")
        self.device_map = self.device

        if torch.cuda.device_count() > 1:
            self.device_map = "auto"
        
        model_name = config.get("model_name")
        type = config.get("weight_type")

        # tokenizer and model
        if config.get("tokenizer_name") is not None:
            tokenizer_name = config.get("tokenizer_name")    
        else:
            tokenizer_name = model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print(f"load tokenizer which name is {tokenizer_name}")


        print(f"load model which name is {model_name}")
        

        if config.get("use_8bits"):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                quantization_config = get_bnb_config(bits_8 = True),
                torch_dtype = torch.float16,
                # torch_dtype = torch.bfloat16 if type == "bf16" else (torch.float16 if type == "fp16" else torch.float32),
                device_map = self.device_map
            )

            # ipdb.set_trace()
            print(f"load model using quantization")
        elif config.get("use_4bits"):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config = get_bnb_config(bits_4 = True),
                torch_dtype = torch.float16,
                device_map = self.device_map
            )

            print(f"load model using quantization")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_type = torch.float16,
                # torch_dtype = torch.bfloat16 if type == "bf16" else (torch.float16 if type == "fp16" else torch.float32),
                device_map = self.device_map,
            )
            # ipdb.set_trace()

        self.model.eval()
        

        # rag_config
        T_rag_config = {
            # I think we need to finetune embedding_model when we select the model to implement.
            "embedding_model": config.get("embedding_model") \
                if config.get("embedding_model") is not None else "BAAI/bge-base-en-v1.5",
            "seed": config.get("seed") if config.get("seed") is not None else 42,
            "top_k": config.get("top_Tk") if config.get("top_Tk") is not None else 5,
            "order": config.get("order") if config.get("order") is not None else "similar_at_top",
        }
        F_rag_config = {
            # I think we need to finetune embedding_model when we select the model to implement.
            "embedding_model": config.get("embedding_model") \
                if config.get("embedding_model") is not None else "BAAI/bge-base-en-v1.5",
            "seed": config.get("seed") if config.get("seed") is not None else 42,
            "top_k": config.get("top_Fk") if config.get("top_Fk") is not None else 5,
            "order": config.get("order") if config.get("order") is not None else "similar_at_top",
        }
        self.correct_rag = RAG(T_rag_config)
        self.wrong_rag = RAG(F_rag_config)
 
        # gen max token
        self.max_token = config.get("max_token") if config.get("max_token") is not None else 32
        self.top_p = config.get("top_p")

        # save the streaming inputs and outputs for iterative improvement
        self.inputs = list()
        self.model_outputs = list()

        
    def generate_response(self, message: list) -> str:
        text_chat = self.tokenizer.apply_chat_template(
            message,
            tokenize=False, # no covert to embedding in this step
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_chat], return_tensors="pt").to(self.device)
        
        # optimize VRAM
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens = self.max_token,
                do_sample = True,
                top_p = self.top_p,
                temperature = 1.0,
            )

        # ignore the input partial, keep the output partial to the result
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    def get_prompt(self, text, option_text, shots = None):
        if shots is None:
            prompt = f"""\
                You are a professional medical doctor. Diagnose the patient based on the provided profile.

                Patient Profile:
                {text}

                Possible Diagnoses (select one, in the format ID: <number>, <diagnosis>):
                {option_text}

                Provide your diagnosis in this exact format: ID: <number>, <diagnosis>. Do not include any additional information.""".strip()
        else:
            prompt = f"""\
                You are a professional medical doctor. Diagnose the patient based on the provided profile.

                Possible Diagnoses (select one, in the format ID: <number>, <diagnosis>):
                {option_text}

                Reference Cases for Your Review:
                {shots}

                Patient Profile:
                {text}

                Provide your diagnosis in this exact format: ID: <number>, <diagnosis>. Do not include any additional information.""".strip()

        return strip_all_lines(prompt)
    
    def get_shot_template(self, question, answer) -> str:
        prompt = f"""profile content: {question}, Answer: {answer}"""
        return strip_all_lines(prompt)

    def extract_label(self, pred_text: str, label2desc: dict[str, str]) -> str:
        prediction = re.findall(pattern = r"ID: (\d+)", string = pred_text)
        candidate = [k.replace("ID: ", "") for k in prediction]
        
        result = None

        if len(candidate) == 1:
            id = candidate[0]
            if int(id) in label2desc:
                result = id
            else:
                print(Fore.RED + f"Prediction {pred_text} not found in the label set. Randomly select one." + Style.RESET_ALL)
                result = random.choice(list(label2desc.keys()))
        else:
            if len(candidate) > 1:
                print(Fore.YELLOW + f"Extracted numbers {candidate} is not exactly one. Select the first one." + Style.RESET_ALL)
                result = candidate[0]
            else:
                print(Fore.RED + f"Prediction {pred_text} has no extracted numbers. Randomly select one." + Style.RESET_ALL)
                result = random.choice(list(label2desc.keys()))

        return str(result)


    def __call__(
        self,
        label2desc: dict[str, str],
        text: str
    ) -> str:
        option_text = '\n'.join([f"ID: {str(k)}, {v}" for k, v in label2desc.items()])
        correct_shots = self.correct_rag.retrieve(query = text, top_k = self.correct_rag.top_k) if (self.correct_rag.insert_acc > 7) else []
        wrong_shots = self.wrong_rag.retrieve(query = text, top_k = self.wrong_rag.top_k) if (self.wrong_rag.insert_acc > 7) else []
        # ipdb.set_trace()
        if len(wrong_shots):
            option = copy.deepcopy(label2desc)
            for i in range(len(wrong_shots)):
                ans_id = re.findall(pattern = r"ID: (\d+)", string = wrong_shots[i])
                ans_id = [k.replace("ID: ", "") for k in ans_id]
                ans_id = int(ans_id[0])
                if ans_id in option:
                    del option[ans_id]
            
            option_text = '\n'.join([f"ID: {str(k)}, {v}" for k, v in option.items()])

        if len(correct_shots):
            prompt = self.get_prompt(text, option_text, correct_shots)
        else:
            prompt = self.get_prompt(text, option_text)

        # ipdb.set_trace()

        messages = [
            {"role": "user", "content": prompt}
        ]
        response = self.generate_response(messages)
        # ipdb.set_trace()
        prediction = self.extract_label(response, label2desc)

        self.update_log_info(log_data={
            "num_input_tokens": len(self.tokenizer.encode(prompt)),
            "num_output_tokens": len(self.tokenizer.encode(response)),
            "correct_shots": str(len(correct_shots)),
            "wrong_shots": str(len(wrong_shots)),
            "input_pred": prompt,
            "output_pred": response,
        })

        self.inputs.append(text)
        self.model_outputs.append(f"ID: {str(prediction)}, {label2desc[int(prediction)]}")
        return prediction

    def update(self, correctness: bool) -> bool:
        if correctness:
            if self.correct_rag.insert_acc < 500:
                question = self.inputs[-1]
                answer = self.model_outputs[-1]
                chunk = self.get_shot_template(question, answer)
                self.correct_rag.insert(key = question, value = chunk)
            return True
        else:
            if self.wrong_rag.insert_acc < 700:
                question = self.inputs[-1]
                answer = self.model_outputs[-1]
                chunk = self.get_shot_template(question, answer)
                self.wrong_rag.insert(key = question, value = chunk)
            return False

class SQLGenerationAgent(Agent):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        
        # device
        self.device = config.get("device")
        self.device_map = self.device

        if torch.cuda.device_count() > 1:
            self.device_map = "auto"
        
        model_name = config.get("model_name")

        # tokenizer and model
        if config.get("tokenizer_name") is not None:
            tokenizer_name = config.get("tokenizer_name")    
        else:
            tokenizer_name = model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print(f"load tokenizer : {tokenizer_name}")
        print(f"load model : {model_name}")
        

        if config.get("use_8bits"):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                quantization_config = get_bnb_config(bits_8 = True),
                torch_dtype = torch.float16,
                # torch_dtype = torch.bfloat16 if type == "bf16" else (torch.float16 if type == "fp16" else torch.float32),
                device_map = self.device_map
            )

            # ipdb.set_trace()
            print(f"load model using quantization (8bits)")
        elif config.get("use_4bits"):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config = get_bnb_config(bits_4 = True),
                torch_dtype = torch.float16,
                device_map = self.device_map
            )

            print(f"load model using quantization (4bits)")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_type = torch.float16,
                # torch_dtype = torch.bfloat16 if type == "bf16" else (torch.float16 if type == "fp16" else torch.float32),
                device_map = self.device_map,
            )
            # ipdb.set_trace()

        self.model.eval()
        

        # rag_config
        T_rag_config = {
            # I think we need to finetune embedding_model when we select the model to implement.
            "embedding_model": config.get("embedding_model") \
                if config.get("embedding_model") is not None else "BAAI/bge-base-en-v1.5",
            "seed": config.get("seed") if config.get("seed") is not None else 42,
            "top_k": config.get("top_Tk") if config.get("top_Tk") is not None else 5,
            "order": config.get("order") if config.get("order") is not None else "similar_at_top",
        }
        F_rag_config = {
            "embedding_model": config.get("embedding_model") \
                if config.get("embedding_model") is not None else "BAAI/bge-base-en-v1.5",
            "seed": config.get("seed") if config.get("seed") is not None else 42,
            "top_k": config.get("top_Fk") if config.get("top_Fk") is not None else 5,
            "order": config.get("order") if config.get("order") is not None else "similar_at_top",
        }
        self.correct_rag = RAG(T_rag_config)
        self.wrong_rag = RAG(F_rag_config)
 
        # gen max token
        self.max_token = config.get("max_token") if config.get("max_token") is not None else 512
        self.top_p = config.get("top_p")

        # save the streaming inputs and outputs for iterative improvement
        self.inputs = list()
        self.model_outputs = list()


    def generate_response(self, message: list) -> str:
        text_chat = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_chat], return_tensors="pt").to(self.device)
        
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_token,
                do_sample=True,
                top_p=self.top_p,
                temperature=1.0,
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    def get_prompt(self, schema: str, query: str, T_shots = None, F_shots = None) -> str:
        if shots is None:
            prompt = f"""\
                You are an expert SQL developer. Write SQL code to answer the given query based on the provided table schema.

                Table Schema:
                {schema}

                User Query:
                {query}

                Write a valid SQL query that answers the user's question. Only provide the SQL code without any explanations.""".strip()
        else:
            prompt = f"""\
                You are an expert SQL developer. Write SQL code to answer the given query based on the provided table schema.

                Table Schema:
                {schema}

                Similar Examples for Reference:
                {shots}

                User Query:
                {query}

                Write a valid SQL query that answers the user's question. Only provide the SQL code without any explanations.""".strip()
            
        return strip_all_lines(prompt)
    
    def get_shot_template(self, query: str, sql: str) -> str:
        prompt = f"""User Query: {query}\nSQL: {sql}"""
        return strip_all_lines(prompt)

    def clean_sql(self, sql: str) -> str:
        """Clean and standardize SQL output"""
        # Remove comments
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        # Remove multiple spaces/newlines
        sql = ' '.join(sql.split())
        # Remove any non-SQL content
        sql = re.sub(r'```sql|```', '', sql)
        return sql.strip()

    def __call__(self, table_schema: str, user_query: str) -> str:
        correct_shots = self.correct_rag.retrieve(query=user_query, top_k=self.correct_rag.top_k) if (self.correct_rag.insert_acc > 0) else []
        wrong_shots = self.correct_rag.retrieve(query=user_query, top_k=self.wrong_rag.top_k) if (self.wrong_rag.insert_acc > 0) else []

        if len(correct_shots) and len(wrong_shots):
            prompt = self.get_prompt(table_schema, user_query, correct_shots, wrong_shots)
        elif len(correct_shots):
            prompt = self.get_prompt(table_schema, user_query, T_shots=correct_shots)
        elif len(wrong_shots):
            prompt = self.get_prompt(table_schema, user_query, F_shots = wrong_shots)
        else:
            prompt = self.get_prompt(table_schema, user_query)

        messages = [{"role": "user", "content": prompt}]
        response = self.generate_response(messages)
        sql_code = self.clean_sql(response)
        
        self.update_log_info(log_data={
            "num_input_tokens": len(self.tokenizer.encode(prompt)),
            "num_output_tokens": len(self.tokenizer.encode(response)),
            "num_shots": str(len(correct_shots) + len(wrong_shots)),
            "input_pred": prompt,
            "output_pred": sql_code,
        })
        
        self.inputs.append((table_schema, user_query))
        self.model_outputs.append(sql_code)
        return sql_code

    def update(self, correctness: bool) -> bool:
        if correctness:
            schema, query = self.inputs[-1]
            sql = self.model_outputs[-1]
            chunk = self.get_shot_template(schema, query, sql)
            self.rag.insert(key=query, value=chunk)
            return True
        return False
        
if __name__ == "__main__":
    from argparse import ArgumentParser
    from execution_pipeline import main

    parser = ArgumentParser()
    parser.add_argument('--bench_name', type=str, required=True)
    parser.add_argument('--embedding_model', type=str, default = "BAAI/bge-base-en-v1.5")
    parser.add_argument('--model_name', type=str, default = "Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument('--device', type=str, default = "cuda")
    parser.add_argument('--use_8bits', action = "store_true")
    parser.add_argument('--use_4bits', action = "store_true")
    parser.add_argument('--weight_type', type = str, default = None)
    parser.add_argument('--output_path', type=str, default = None)
    parser.add_argument('--use_wandb', action = "store_true")
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--top_Tk', type = int, default = 5)
    parser.add_argument('--top_Fk', type = int, default = 5)
    parser.add_argument('--top_p', type = float, default = 0.75)
    parser.add_argument('--description', type = str, default = None)


    args = parser.parse_args()

    if args.bench_name.startswith("classification"):
        agent_name = ClassificationAgent
    elif args.bench_name.startswith("sql_generation"):
        agent_name = SQLGenerationAgent
    else:
        raise ValueError(f"Invalid benchmark name: {args.bench_name}")

    bench_cfg = {
        'bench_name': args.bench_name,
        'output_path': args.output_path
    }
    config = {
        "exp_name":f'self_streamicl_{args.bench_name}_{args.model_name}_{args.description}',
        "bench_name": args.bench_name,
        "embedding_model": args.embedding_model,
        "model_name": args.model_name,
        "device": args.device,
        "use_8bits": args.use_8bits,
        "use_4bits": args.use_4bits,
        "weight_type": args.weight_type,
        "seed": args.seed,
        "top_Tk": args.top_Tk,
        "top_Fk": args.top_Fk,
        "top_p": args.top_p,
    }
    agent = agent_name(config)
    main(agent, bench_cfg, use_wandb=args.use_wandb, wandb_name=config["exp_name"], wandb_config=config)