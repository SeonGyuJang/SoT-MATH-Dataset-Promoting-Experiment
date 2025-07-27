# 후보군 : Take full advantage of your internal reasoning space
import os
import json
import time
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from tqdm import tqdm
import dotenv
from collections import Counter
import random
import glob

# 환경 설정
dotenv.load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY가 .env 파일에 설정되지 않았습니다.")
os.environ["GOOGLE_API_KEY"] = api_key

@dataclass
class ExperimentResult:
    """실험 결과를 저장하는 데이터 클래스"""
    question: str
    correct_answer: str
    zero_shot_response: str
    zero_shot_answer: str
    zero_shot_correct: bool
    few_shot_response: str
    few_shot_answer: str
    few_shot_correct: bool
    few_shot_cot_response: str
    few_shot_cot_answer: str
    few_shot_cot_correct: bool
    zero_shot_cot_response: str
    zero_shot_cot_answer: str
    zero_shot_cot_correct: bool
    zero_shot_sot_response: str
    zero_shot_sot_answer: str
    zero_shot_sot_correct: bool
    few_shot_sot_response: str
    few_shot_sot_answer: str
    few_shot_sot_correct: bool
    problem_level: Optional[str] = None  # MATH 데이터셋의 난이도
    problem_type: Optional[str] = None   # MATH 데이터셋의 문제 유형

class MathExperiment:
    """MATH 데이터셋에서 Zero-shot, Few-shot, Few-shot-CoT, Zero-shot-CoT, Zero-shot-SoT, Few-shot-SoT 성능 비교"""
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        # LLM 초기화
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,  # 일관성을 위한 Greedy 디코딩
            max_output_tokens=1024
        )
        
        # 프롬프트 템플릿 설정 (영어)
        self.zero_shot_template = PromptTemplate.from_template(
            "Problem: {question}\nAnswer:"
        )
        
        self.few_shot_template = PromptTemplate.from_template(
            """Here are some example math problems with their answers:
Example 1: {example1_question}\nAnswer: {example1_answer}
Example 2: {example2_question}\nAnswer: {example2_answer}
Example 3: {example3_question}\nAnswer: {example3_answer}

Now solve this problem:
Problem: {question}
Answer:"""
        )
        
        self.few_shot_cot_template = PromptTemplate.from_template(
            """Here are some example math problems with their answers:
Example 1: {example1_question}\nAnswer: {example1_answer}
Example 2: {example2_question}\nAnswer: {example2_answer}
Example 3: {example3_question}\nAnswer: {example3_answer}

Now solve this problem:
Problem: {question}
Let's think step by step."""
        )
        
        self.zero_shot_cot_template = PromptTemplate.from_template(
            "Problem: {question}\nLet's think step by step."
        )
        
        self.zero_shot_sot_template = PromptTemplate.from_template(
            "Problem: {question}\nThink step‑by‑step internally and, once certain, reply with ANSWER: <your answer> and nothing else."
        )
        
        self.few_shot_sot_template = PromptTemplate.from_template(
            """Here are some example math problems with their answers:
Example 1: {example1_question}\nAnswer: {example1_answer}
Example 2: {example2_question}\nAnswer: {example2_answer}
Example 3: {example3_question}\nAnswer: {example3_answer}

Now solve this problem:
Problem: {question}
Think step‑by‑step internally and, once certain, reply with ANSWER: <your answer> and nothing else."""
        )
        
        self.extract_answer_template = PromptTemplate.from_template(
            "{response}\n\nTherefore, the answer (arabic numerals) is:"
        )
        
        self.results: List[ExperimentResult] = []
        self.examples = []  # Few-shot, Few-shot-CoT, Few-shot-SoT 예제를 저장
    
    def load_examples(self, data_dir: str, dataset_type: str, num_examples: int = 3):
        """Few-shot, Few-shot-CoT, Few-shot-SoT을 위한 예제 로드"""
        if dataset_type == "math":
            # train 폴더에서 예제 로드
            train_dir = os.path.join(data_dir, "train")
            json_files = glob.glob(os.path.join(train_dir, "**", "*.json"), recursive=True)
            if not json_files:
                raise ValueError(f"{train_dir}에서 JSON 파일을 찾을 수 없습니다.")
            samples = random.sample(json_files, min(num_examples, len(json_files)))
            self.examples = []
            for file in samples:
                with open(file, 'r', encoding='utf-8') as f:
                    sample = json.load(f)
                self.examples.append({
                    'question': sample['problem'],
                    'answer': self.extract_answer_from_math(sample['solution'])
                })
        else:  # gsm8k
            raise ValueError("GSM8K 데이터셋은 로컬 MATH 데이터셋으로 대체되었습니다.")
    
    def extract_answer_from_math(self, text: str) -> str:
        """MATH 데이터셋 형식의 답변 추출"""
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        boxed_matches = re.findall(boxed_pattern, text)
        if boxed_matches:
            answer = boxed_matches[-1]
            answer = re.sub(r'\\[a-zA-Z]+', '', answer)
            answer = answer.replace('$', '').strip()
            return answer
        
        answer_pattern = r'answer\s*(?:is|:)\s*\$?\s*([^\s,\.]+)'
        match = re.search(answer_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        number_pattern = r'(?:(?:\-?\d+(?:\.\d+)?)|(?:\-?\d+/\d+)|(?:[a-zA-Z]+))'
        numbers = re.findall(number_pattern, text)
        if numbers:
            return numbers[-1]
        
        return ""
    
    def extract_answer_from_gsm8k(self, text: str) -> str:
        """GSM8K 형식의 답변 추출 (로컬 MATH에서는 사용되지 않음)"""
        pattern1 = r'####\s*(\-?\d+(?:,\d{3})*(?:\.\d+)?)'
        match = re.search(pattern1, text)
        if match:
            return match.group(1).replace(',', '')
        
        pattern2 = r'answer\s*(?:is|:)\s*\$?\s*(\-?\d+(?:,\d{3})*(?:\.\d+)?)'
        match = re.search(pattern2, text, re.IGNORECASE)
        if match:
            return match.group(1).replace(',', '')
        
        numbers = re.findall(r'\-?\d+(?:,\d{3})*(?:\.\d+)?', text)
        if numbers:
            return numbers[-1].replace(',', '')
        
        return ""
    
    def normalize_answer(self, answer: str) -> str:
        """답변 정규화"""
        answer = answer.strip()
        answer = answer.replace(',', '')
        answer = answer.replace('$', '')
        
        if '/' in answer:
            try:
                parts = answer.split('/')
                if len(parts) == 2:
                    num = float(parts[0])
                    den = float(parts[1])
                    result = num / den
                    if result.is_integer():
                        return str(int(result))
                    return str(result)
            except:
                pass
        
        try:
            num = float(answer)
            if num.is_integer():
                return str(int(num))
            return str(num)
        except:
            return answer
    
    def compare_answers(self, pred: str, gold: str) -> bool:
        """두 답변이 동일한지 비교"""
        pred_norm = self.normalize_answer(pred)
        gold_norm = self.normalize_answer(gold)
        return pred_norm == gold_norm
    
    def run_single_example(self, question: str, correct_answer: str, 
                          dataset_type: str = "math") -> ExperimentResult:
        """단일 문제에 대해 여섯 가지 방법 비교"""
        
        # Zero-shot
        zero_shot_prompt = self.zero_shot_template.format(question=question)
        zero_shot_response = self.llm.invoke(zero_shot_prompt).content
        zero_shot_answer = self.extract_answer_from_math(zero_shot_response) if dataset_type == "math" else self.extract_answer_from_gsm8k(zero_shot_response)
        zero_shot_correct = self.compare_answers(zero_shot_answer, correct_answer)
        
        # Few-shot
        few_shot_prompt = self.few_shot_template.format(
            question=question,
            example1_question=self.examples[0]['question'],
            example1_answer=self.examples[0]['answer'],
            example2_question=self.examples[1]['question'],
            example2_answer=self.examples[1]['answer'],
            example3_question=self.examples[2]['question'],
            example3_answer=self.examples[2]['answer']
        )
        few_shot_response = self.llm.invoke(few_shot_prompt).content
        few_shot_answer = self.extract_answer_from_math(few_shot_response) if dataset_type == "math" else self.extract_answer_from_gsm8k(few_shot_response)
        few_shot_correct = self.compare_answers(few_shot_answer, correct_answer)
        
        # Few-shot-CoT
        few_shot_cot_prompt = self.few_shot_cot_template.format(
            question=question,
            example1_question=self.examples[0]['question'],
            example1_answer=self.examples[0]['answer'],
            example2_question=self.examples[1]['question'],
            example2_answer=self.examples[1]['answer'],
            example3_question=self.examples[2]['question'],
            example3_answer=self.examples[2]['answer']
        )
        few_shot_cot_response = self.llm.invoke(few_shot_cot_prompt).content
        if "therefore" not in few_shot_cot_response.lower() and "answer" not in few_shot_cot_response.lower():
            extract_prompt = self.extract_answer_template.format(response=few_shot_cot_response)
            few_shot_cot_final = self.llm.invoke(extract_prompt).content
            few_shot_cot_response += "\n" + few_shot_cot_final
        few_shot_cot_answer = self.extract_answer_from_math(few_shot_cot_response) if dataset_type == "math" else self.extract_answer_from_gsm8k(few_shot_cot_response)
        few_shot_cot_correct = self.compare_answers(few_shot_cot_answer, correct_answer)
        
        # Zero-shot-CoT
        zero_shot_cot_prompt = self.zero_shot_cot_template.format(question=question)
        zero_shot_cot_response = self.llm.invoke(zero_shot_cot_prompt).content
        if "therefore" not in zero_shot_cot_response.lower() and "answer" not in zero_shot_cot_response.lower():
            extract_prompt = self.extract_answer_template.format(response=zero_shot_cot_response)
            zero_shot_cot_final = self.llm.invoke(extract_prompt).content
            zero_shot_cot_response += "\n" + zero_shot_cot_final
        zero_shot_cot_answer = self.extract_answer_from_math(zero_shot_cot_response) if dataset_type == "math" else self.extract_answer_from_gsm8k(zero_shot_cot_response)
        zero_shot_cot_correct = self.compare_answers(zero_shot_cot_answer, correct_answer)
        
        # Zero-shot-SoT
        zero_shot_sot_prompt = self.zero_shot_sot_template.format(question=question)
        zero_shot_sot_response = self.llm.invoke(zero_shot_sot_prompt).content
        zero_shot_sot_answer = self.extract_answer_from_math(zero_shot_sot_response) if dataset_type == "math" else self.extract_answer_from_gsm8k(zero_shot_sot_response)
        zero_shot_sot_correct = self.compare_answers(zero_shot_sot_answer, correct_answer)
        
        # Few-shot-SoT
        few_shot_sot_prompt = self.few_shot_sot_template.format(
            question=question,
            example1_question=self.examples[0]['question'],
            example1_answer=self.examples[0]['answer'],
            example2_question=self.examples[1]['question'],
            example2_answer=self.examples[1]['answer'],
            example3_question=self.examples[2]['question'],
            example3_answer=self.examples[2]['answer']
        )
        few_shot_sot_response = self.llm.invoke(few_shot_sot_prompt).content
        few_shot_sot_answer = self.extract_answer_from_math(few_shot_sot_response) if dataset_type == "math" else self.extract_answer_from_gsm8k(few_shot_sot_response)
        few_shot_sot_correct = self.compare_answers(few_shot_sot_answer, correct_answer)
        
        return ExperimentResult(
            question=question,
            correct_answer=correct_answer,
            zero_shot_response=zero_shot_response,
            zero_shot_answer=zero_shot_answer,
            zero_shot_correct=zero_shot_correct,
            few_shot_response=few_shot_response,
            few_shot_answer=few_shot_answer,
            few_shot_correct=few_shot_correct,
            few_shot_cot_response=few_shot_cot_response,
            few_shot_cot_answer=few_shot_cot_answer,
            few_shot_cot_correct=few_shot_cot_correct,
            zero_shot_cot_response=zero_shot_cot_response,
            zero_shot_cot_answer=zero_shot_cot_answer,
            zero_shot_cot_correct=zero_shot_cot_correct,
            zero_shot_sot_response=zero_shot_sot_response,
            zero_shot_sot_answer=zero_shot_sot_answer,
            zero_shot_sot_correct=zero_shot_sot_correct,
            few_shot_sot_response=few_shot_sot_response,
            few_shot_sot_answer=few_shot_sot_answer,
            few_shot_sot_correct=few_shot_sot_correct
        )
    
    def run_math_experiment(self, data_set: str = "C:/Users/dsng3/Desktop/MATH", num_samples: int = 50, 
                           difficulty_filter: Optional[int] = None,
                           save_results: bool = True):
        """MATH 데이터셋에서 실험 수행"""
        
        print(f"MATH 데이터셋 로딩 중... 경로: {data_set}")
        test_dir = os.path.join(data_set, "test")
        if not os.path.exists(test_dir):
            raise ValueError(f"테스트 데이터셋 경로 {test_dir}가 존재하지 않습니다.")
        
        # 모든 JSON 파일 로드
        test_json_files = glob.glob(os.path.join(test_dir, "**", "*.json"), recursive=True)
        if not test_json_files:
            raise ValueError(f"{test_dir}에서 JSON 파일을 찾을 수 없습니다.")
        
        test_data = []
        for file in test_json_files:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                test_data.append(data)
        
        # 예제 로드
        self.load_examples(data_set, dataset_type="math")
        
        # 난이도 필터링
        if difficulty_filter:
            print(f"난이도 {difficulty_filter} 문제만 필터링...")
            filtered_data = [d for d in test_data if d['level'] == f'Level {difficulty_filter}']
            if len(filtered_data) < num_samples:
                print(f"경고: 난이도 {difficulty_filter} 문제가 {len(filtered_data)}개밖에 없습니다.")
                test_samples = filtered_data
            else:
                test_samples = random.sample(filtered_data, num_samples)
        else:
            test_samples = random.sample(test_data, min(num_samples, len(test_data)))
                
        print(f"총 {len(test_samples)}개 문제로 실험 시작...")
        print(f"문제 난이도 분포: {Counter([s['level'] for s in test_samples])}")
        print(f"문제 유형 분포: {Counter([s['type'] for s in test_samples])}")
        
        for sample in tqdm(test_samples, desc="MATH 실험 진행"):
            question = sample['problem']
            solution = sample['solution']
            correct_answer = self.extract_answer_from_math(solution) or re.findall(r'\\boxed\{([^}]+)\}', solution)[-1]
            
            try:
                result = self.run_single_example(question, correct_answer, dataset_type="math")
                result.problem_level = sample['level']
                result.problem_type = sample['type']
                self.results.append(result)
                
                if len(self.results) % 10 == 0:
                    zero_shot_acc = sum(r.zero_shot_correct for r in self.results) / len(self.results)
                    few_shot_acc = sum(r.few_shot_correct for r in self.results) / len(self.results)
                    few_shot_cot_acc = sum(r.few_shot_cot_correct for r in self.results) / len(self.results)
                    zero_shot_cot_acc = sum(r.zero_shot_cot_correct for r in self.results) / len(self.results)
                    zero_shot_sot_acc = sum(r.zero_shot_sot_correct for r in self.results) / len(self.results)
                    few_shot_sot_acc = sum(r.few_shot_sot_correct for r in self.results) / len(self.results)
                    print(f"\n현재까지 - Zero-shot: {zero_shot_acc:.1%}, Few-shot: {few_shot_acc:.1%}, "
                          f"Few-shot-CoT: {few_shot_cot_acc:.1%}, Zero-shot-CoT: {zero_shot_cot_acc:.1%}, "
                          f"Zero-shot-SoT: {zero_shot_sot_acc:.1%}, Few-shot-SoT: {few_shot_sot_acc:.1%}")
                    
            except Exception as e:
                print(f"\n오류 발생: {e}")
                continue
                
            time.sleep(0.5)
        
        if save_results:
            self.save_results(dataset_name="math")
            
        self.print_results(dataset_name="MATH")
        self.plot_results(dataset_name="MATH")
        
    def run_gsm8k_experiment(self, num_samples: int = 100, save_results: bool = True):
        """GSM8K 데이터셋에서 실험 수행 (로컬 MATH 사용 시 제외 가능)"""
        print("GSM8K 실험은 로컬 MATH 데이터셋을 사용하므로 실행되지 않습니다.")
        return
    
    def print_results(self, dataset_name: str = "MATH"):
        """실험 결과 출력"""
        if not self.results:
            print("결과가 없습니다.")
            return
            
        zero_shot_correct = sum(r.zero_shot_correct for r in self.results)
        few_shot_correct = sum(r.few_shot_correct for r in self.results)
        few_shot_cot_correct = sum(r.few_shot_cot_correct for r in self.results)
        zero_shot_cot_correct = sum(r.zero_shot_cot_correct for r in self.results)
        zero_shot_sot_correct = sum(r.zero_shot_sot_correct for r in self.results)
        few_shot_sot_correct = sum(r.few_shot_sot_correct for r in self.results)
        total = len(self.results)
        
        print(f"\n{'='*60}")
        print(f"{dataset_name} 프롬프팅 실험 결과")
        print(f"{'='*60}")
        print(f"총 문제 수: {total}")
        print(f"\nZero-shot 정확도: {zero_shot_correct}/{total} = {zero_shot_correct/total:.1%}")
        print(f"Few-shot 정확도: {few_shot_correct}/{total} = {few_shot_correct/total:.1%}")
        print(f"Few-shot-CoT 정확도: {few_shot_cot_correct}/{total} = {few_shot_cot_correct/total:.1%}")
        print(f"Zero-shot-CoT 정확도: {zero_shot_cot_correct}/{total} = {zero_shot_cot_correct/total:.1%}")
        print(f"Zero-shot-SoT 정확도: {zero_shot_sot_correct}/{total} = {zero_shot_sot_correct/total:.1%}")
        print(f"Few-shot-SoT 정확도: {few_shot_sot_correct}/{total} = {few_shot_sot_correct/total:.1%}")
        
        if dataset_name == "MATH" and any(r.problem_level for r in self.results):
            print(f"\n난이도별 성능:")
            levels = sorted(set(r.problem_level for r in self.results if r.problem_level))
            for level in levels:
                level_results = [r for r in self.results if r.problem_level == level]
                if level_results:
                    zs_acc = sum(r.zero_shot_correct for r in level_results) / len(level_results)
                    fs_acc = sum(r.few_shot_correct for r in level_results) / len(level_results)
                    fsc_acc = sum(r.few_shot_cot_correct for r in level_results) / len(level_results)
                    zsc_acc = sum(r.zero_shot_cot_correct for r in level_results) / len(level_results)
                    zss_acc = sum(r.zero_shot_sot_correct for r in level_results) / len(level_results)
                    fss_acc = sum(r.few_shot_sot_correct for r in level_results) / len(level_results)
                    print(f"{level}: Zero-shot {zs_acc:.1%}, Few-shot {fs_acc:.1%}, "
                          f"Few-shot-CoT {fsc_acc:.1%}, Zero-shot-CoT {zsc_acc:.1%}, "
                          f"Zero-shot-SoT {zss_acc:.1%}, Few-shot-SoT {fss_acc:.1%} (n={len(level_results)})")
        
        print(f"{'='*60}")
        
        print("\n예시 결과:")
        for i, result in enumerate(self.results[:3]):
            print(f"\n문제 {i+1}: {result.question[:100]}...")
            print(f"정답: {result.correct_answer}")
            print(f"Zero-shot: {result.zero_shot_answer} ({'✓' if result.zero_shot_correct else '✗'})")
            print(f"Few-shot: {result.few_shot_answer} ({'✓' if result.few_shot_correct else '✗'})")
            print(f"Few-shot-CoT: {result.few_shot_cot_answer} ({'✓' if result.few_shot_cot_correct else '✗'})")
            print(f"Zero-shot-CoT: {result.zero_shot_cot_answer} ({'✓' if result.zero_shot_cot_correct else '✗'})")
            print(f"Zero-shot-SoT: {result.zero_shot_sot_answer} ({'✓' if result.zero_shot_sot_correct else '✗'})")
            print(f"Few-shot-SoT: {result.few_shot_sot_answer} ({'✓' if result.few_shot_sot_correct else '✗'})")
            
    def plot_results(self, dataset_name: str = "MATH"):
        """결과 시각화"""
        if not self.results:
            return
            
        zero_shot_acc = sum(r.zero_shot_correct for r in self.results) / len(self.results) * 100
        few_shot_acc = sum(r.few_shot_correct for r in self.results) / len(self.results) * 100
        few_shot_cot_acc = sum(r.few_shot_cot_correct for r in self.results) / len(self.results) * 100
        zero_shot_cot_acc = sum(r.zero_shot_cot_correct for r in self.results) / len(self.results) * 100
        zero_shot_sot_acc = sum(r.zero_shot_sot_correct for r in self.results) / len(self.results) * 100
        few_shot_sot_acc = sum(r.few_shot_sot_correct for r in self.results) / len(self.results) * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 정확도 비교
        methods = ['Zero-shot', 'Few-shot', 'Few-shot-CoT', 'Zero-shot-CoT', 'Zero-shot-SoT', 'Few-shot-SoT']
        accuracies = [zero_shot_acc, few_shot_acc, few_shot_cot_acc, zero_shot_cot_acc, zero_shot_sot_acc, few_shot_sot_acc]
        colors = ['#FF6B6B', '#F39C12', '#27AE60', '#4ECDC4', '#8E44AD', '#3498DB']
        
        bars = ax1.bar(methods, accuracies, color=colors)
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title(f'{dataset_name} Performance Comparison')
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        # 개선 분석
        categories = ['All Wrong', 'Zero-shot Only', 'Few-shot Only', 'Few-shot-CoT Only', 
                      'Zero-shot-CoT Only', 'Zero-shot-SoT Only', 'Few-shot-SoT Only', 'All Correct']
        values = [
            sum(1 for r in self.results if not r.zero_shot_correct and not r.few_shot_correct and 
                not r.few_shot_cot_correct and not r.zero_shot_cot_correct and 
                not r.zero_shot_sot_correct and not r.few_shot_sot_correct),
            sum(1 for r in self.results if r.zero_shot_correct and not r.few_shot_correct and 
                not r.few_shot_cot_correct and not r.zero_shot_cot_correct and 
                not r.zero_shot_sot_correct and not r.few_shot_sot_correct),
            sum(1 for r in self.results if not r.zero_shot_correct and r.few_shot_correct and 
                not r.few_shot_cot_correct and not r.zero_shot_cot_correct and 
                not r.zero_shot_sot_correct and not r.few_shot_sot_correct),
            sum(1 for r in self.results if not r.zero_shot_correct and not r.few_shot_correct and 
                r.few_shot_cot_correct and not r.zero_shot_cot_correct and 
                not r.zero_shot_sot_correct and not r.few_shot_sot_correct),
            sum(1 for r in self.results if not r.zero_shot_correct and not r.few_shot_correct and 
                not r.few_shot_cot_correct and r.zero_shot_cot_correct and 
                not r.zero_shot_sot_correct and not r.few_shot_sot_correct),
            sum(1 for r in self.results if not r.zero_shot_correct and not r.few_shot_correct and 
                not r.few_shot_cot_correct and not r.zero_shot_cot_correct and 
                r.zero_shot_sot_correct and not r.few_shot_sot_correct),
            sum(1 for r in self.results if not r.zero_shot_correct and not r.few_shot_correct and 
                not r.few_shot_cot_correct and not r.zero_shot_cot_correct and 
                not r.zero_shot_sot_correct and r.few_shot_sot_correct),
            sum(1 for r in self.results if r.zero_shot_correct and r.few_shot_correct and 
                r.few_shot_cot_correct and r.zero_shot_cot_correct and 
                r.zero_shot_sot_correct and r.few_shot_sot_correct)
        ]
        
        ax2.bar(categories, values, color=['#E74C3C', '#FF6B6B', '#F39C12', '#27AE60', '#4ECDC4', '#8E44AD', '#3498DB', '#2ECC71'])
        ax2.set_ylabel('Number of Problems')
        ax2.set_title('Detailed Performance Analysis')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{dataset_name.lower()}_prompting_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # MATH 데이터셋인 경우 난이도별 분석
        if dataset_name == "MATH" and any(r.problem_level for r in self.results):
            levels = sorted(set(r.problem_level for r in self.results if r.problem_level))
            level_zs_accs = []
            level_fs_accs = []
            level_fsc_accs = []
            level_zsc_accs = []
            level_zss_accs = []
            level_fss_accs = []
            
            for level in levels:
                level_results = [r for r in self.results if r.problem_level == level]
                if level_results:
                    zs_acc = sum(r.zero_shot_correct for r in level_results) / len(level_results) * 100
                    fs_acc = sum(r.few_shot_correct for r in level_results) / len(level_results) * 100
                    fsc_acc = sum(r.few_shot_cot_correct for r in level_results) / len(level_results) * 100
                    zsc_acc = sum(r.zero_shot_cot_correct for r in level_results) / len(level_results) * 100
                    zss_acc = sum(r.zero_shot_sot_correct for r in level_results) / len(level_results) * 100
                    fss_acc = sum(r.few_shot_sot_correct for r in level_results) / len(level_results) * 100
                    level_zs_accs.append(zs_acc)
                    level_fs_accs.append(fs_acc)
                    level_fsc_accs.append(fsc_acc)
                    level_zsc_accs.append(zsc_acc)
                    level_zss_accs.append(zss_acc)
                    level_fss_accs.append(fss_acc)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(levels))
            width = 0.15
            
            ax.bar(x - width*2.5, level_zs_accs, width, label='Zero-shot', color='#FF6B6B')
            ax.bar(x - width*1.5, level_fs_accs, width, label='Few-shot', color='#F39C12')
            ax.bar(x - width*0.5, level_fsc_accs, width, label='Few-shot-CoT', color='#27AE60')
            ax.bar(x + width*0.5, level_zsc_accs, width, label='Zero-shot-CoT', color='#4ECDC4')
            ax.bar(x + width*1.5, level_zss_accs, width, label='Zero-shot-SoT', color='#8E44AD')
            ax.bar(x + width*2.5, level_fss_accs, width, label='Few-shot-SoT', color='#3498DB')
            
            ax.set_xlabel('Difficulty Level')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Performance by Difficulty')
            ax.set_xticks(x)
            ax.set_xticklabels(levels)
            ax.legend()
            ax.set_ylim(0, 100)
            
            plt.tight_layout()
            plt.savefig(f'{dataset_name.lower()}_difficulty_results.png', dpi=300, bbox_inches='tight')
            plt.show()
        
    def save_results(self, filename: Optional[str] = None, dataset_name: str = "math"):
        """결과를 JSON 파일로 저장"""
        if filename is None:
            filename = f'{dataset_name}_prompting_results.json'
            
        results_dict = {
            'dataset': dataset_name,
            'summary': {
                'total_problems': len(self.results),
                'zero_shot_accuracy': sum(r.zero_shot_correct for r in self.results) / len(self.results),
                'few_shot_accuracy': sum(r.few_shot_correct for r in self.results) / len(self.results),
                'few_shot_cot_accuracy': sum(r.few_shot_cot_correct for r in self.results) / len(self.results),
                'zero_shot_cot_accuracy': sum(r.zero_shot_cot_correct for r in self.results) / len(self.results),
                'zero_shot_sot_accuracy': sum(r.zero_shot_sot_correct for r in self.results) / len(self.results),
                'few_shot_sot_accuracy': sum(r.few_shot_sot_correct for r in self.results) / len(self.results)
            },
            'detailed_results': [
                {
                    'question': r.question,
                    'correct_answer': r.correct_answer,
                    'problem_level': r.problem_level,
                    'problem_type': r.problem_type,
                    'zero_shot': {
                        'response': r.zero_shot_response,
                        'extracted_answer': r.zero_shot_answer,
                        'correct': r.zero_shot_correct
                    },
                    'few_shot': {
                        'response': r.few_shot_response,
                        'extracted_answer': r.few_shot_answer,
                        'correct': r.few_shot_correct
                    },
                    'few_shot_cot': {
                        'response': r.few_shot_cot_response,
                        'extracted_answer': r.few_shot_cot_answer,
                        'correct': r.few_shot_cot_correct
                    },
                    'zero_shot_cot': {
                        'response': r.zero_shot_cot_response,
                        'extracted_answer': r.zero_shot_cot_answer,
                        'correct': r.zero_shot_cot_correct
                    },
                    'zero_shot_sot': {
                        'response': r.zero_shot_sot_response,
                        'extracted_answer': r.zero_shot_sot_answer,
                        'correct': r.zero_shot_sot_correct
                    },
                    'few_shot_sot': {
                        'response': r.few_shot_sot_response,
                        'extracted_answer': r.few_shot_sot_answer,
                        'correct': r.few_shot_sot_correct
                    }
                }
                for r in self.results
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        print(f"\n결과가 {filename}에 저장되었습니다.")

def main():
    """메인 실행 함수"""
    print("프롬프팅 실험을 시작합니다...")
    print("이 실험은 Zero-shot, Few-shot, Few-shot-CoT, Zero-shot-CoT, Zero-shot-SoT, Few-shot-SoT의 수학 문제 해결 성능을 비교합니다.\n")
    
    data_set = "C:/Users/dsng3/Desktop/MATH/MATH"
    if not os.path.exists(data_set):
        raise ValueError(f"지정한 경로 {data_set}가 존재하지 않습니다.")
    
    print("\n=== MATH 데이터셋 실험 ===")
    difficulty = input("특정 난이도만 테스트하시겠습니까? (1-5, Enter는 전체): ").strip()
    difficulty_filter = int(difficulty) if difficulty.isdigit() and 1 <= int(difficulty) <= 5 else None
    
    experiment = MathExperiment(model_name="gemini-2.0-flash")
    experiment.run_math_experiment(
        data_set=data_set,
        num_samples=50,
        difficulty_filter=difficulty_filter,
        save_results=True
    )
    
    print("\n실험이 완료되었습니다!")

if __name__ == "__main__":
    main()