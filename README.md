# MATH 데이터셋 프롬프팅 실험
- 2025.07.28 작성(연구 아이디어)

이 저장소는 MATH 데이터셋의 수학 문제를 풀기 위해 다양한 프롬프팅 전략의 성능을 비교하는 Python 스크립트(`main.py`)를 포함합니다. 실험은 **Zero-shot**, **Few-shot**, **Few-shot-CoT**, **Zero-shot-CoT**, **Zero-shot-SoT**, **Few-shot-SoT** 등 6가지 프롬프팅 방법을 평가합니다. 주요 목표는 새로운 **Silent-of-Thought (SoT)** 프롬프팅 전략("Think silently and only output the answer")이 단계별 추론을 명시적으로 요구하는 기존 **Chain-of-Thought (CoT)** 프롬프팅보다 더 나은 성능을 보이는지 확인하는 것입니다.

## 동기

**Chain-of-Thought (CoT)** 프롬프팅은 대형 언어 모델(LLM)이 단계별로 명시적인 추론을 수행하도록 유도하여 산술 및 논리적 추론과 같은 복잡한 작업에서 성능을 향상시킵니다. 그러나 상세한 추론 단계를 생성하는 과정은 모델이 사용자에게 이해 가능한 출력을 만드는 데 집중하게 되어 비효율적이거나 오류를 유발할 수 있습니다. **Silent-of-Thought (SoT)**는 모델이 내부적으로 추론을 수행하되 이를 외부로 드러내지 않도록 지시함으로써 계산 자원을 더 효율적으로 사용하고, 상세한 설명으로 인한 오류를 줄이며, 더 높은 정확도를 달성할 가능성을 테스트합니다.

이 실험은 MATH 데이터셋에서 SoT와 CoT를 전통적인 Zero-shot 및 Few-shot 프롬프팅과 비교하여 이 가설을 검증합니다.

## 실험 개요

### 프롬프팅 전략
- **Zero-shot**: 예제나 추론 지시 없이 직접 답변 요구
  - 프롬프트: `Problem: {question}\nAnswer:`
- **Few-shot**: 세 가지 예제 문제와 답변 제공
  - 프롬프트: 예제 후 `Problem: {question}\nAnswer:`
- **Few-shot-CoT**: 세 가지 예제와 단계별 사고 지시
  - 프롬프트: 예제 후 `Let's think step by step.`
- **Zero-shot-CoT**: 예제 없이 단계별 사고 지시
  - 프롬프트: `Problem: {question}\nLet's think step by step.`
- **Zero-shot-SoT**: 예제 없이 조용히 사고하고 답변만 출력
  - 프롬프트: `Problem: {question}\nThink silently and only output the answer.`
- **Few-shot-SoT**: 세 가지 예제와 조용히 사고 지시
  - 프롬프트: 예제 후 `Think silently and only output the answer.`

### 데이터셋
- **MATH 데이터셋**: 대수학, 기하학, 미적분학 등 다양한 도메인의 수학 문제(난이도 1~5)
- **경로**: `../MATH` (또는 사용자 지정 경로, 예: `C:/Users/dsng3/Desktop/MATH`)
- **구조**:
  - `train/`: Few-shot 예제 샘플링
  - `test/`: 평가용
- **문제 형식**: JSON 파일(문제 텍스트, 솔루션, 난이도, 유형 포함)

### 모델
- **LLM**: Google Gemini-2.0-Flash (`model_name`으로 설정 가능)
- **설정**: Temperature = 0 (Greedy 디코딩), 최대 출력 토큰 = 1024

### 평가 지표
- **정확도**: 각 프롬프팅 방법으로 올바르게 푼 문제의 비율
- **난이도별 분석**: 난이도 레벨(1~5)별 정확도
- **상세 분석**: 특정 방법만 문제를 푼 경우 분류(예: "All Correct", "Zero-shot-SoT Only")

## 설정

### 요구 사항
- **종속성**:
  ```bash
  pip install langchain langchain-google-genai matplotlib numpy tqdm python-dotenv
  ```
- **Google API 키**:
  - 프로젝트 루트에 `.env` 파일 생성:
    ```plaintext
    GOOGLE_API_KEY=your_google_api_key_here
    ```
- **MATH 데이터셋**:
  - `MATH.zip`을 다운로드하여 `../MATH` 또는 사용자 지정 경로(예: `C:/Users/dsng3/Desktop/MATH`)에 압축 해제
  - `train/` 및 `test/` 하위 디렉토리에 JSON 파일 포함

### 파일 구조
```
project_root/
├── main.py                    # 메인 실험 스크립트
├── .env                      # API 키 환경 파일
├── math_prompting_results.json  # 출력 결과
├── math_prompting_results.png   # 정확도 플롯
├── math_difficulty_results.png  # 난이도별 플롯
├── ../MATH                   # MATH 데이터셋
│   ├── train/
│   └── test/
```

## 사용 방법

1. **저장소 클론**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **환경 설정**:
   - MATH 데이터셋을 `../MATH` 또는 지정 경로(예: `C:/Users/dsng3/Desktop/MATH`)에 배치
   - `.env` 파일에 Google API 키 추가

3. **실험 실행**:
   ```bash
   python main.py
   ```
   - **프롬프트**: 특정 난이도(1-5) 입력 또는 Enter로 전체 테스트
   - **출력**:
     - 콘솔: 진행 상황, 최종 정확도, 난이도별 분석, 예시 결과
     - 파일: `math_prompting_results.json` (상세 결과)
     - 플롯: `math_prompting_results.png` (정확도 비교), `math_difficulty_results.png` (난이도별 성능)

4. **설정 변경**:
   - 샘플 수: `run_math_experiment`의 `num_samples` 수정 (기본값: 50)
   - 모델: `MathExperiment`의 `model_name` 변경 (기본값: "gemini-2.0-flash")
   - Few-shot 예제 수: `load_examples`의 `num_examples` 조정 (기본값: 3)

## 예상 결과

- **정확도 비교**: Zero-shot, Few-shot, Few-shot-CoT, Zero-shot-CoT, Zero-shot-SoT, Few-shot-SoT의 정확도를 비교하는 막대 그래프
- **상세 분석**: 특정 방법만 문제를 푼 경우를 보여주는 막대 그래프(예: "Zero-shot-SoT Only")
- **난이도별 분석**: 각 난이도 레벨별 정확도 막대 그래프
- **JSON 출력**: 질문, 답변, 응답, 정확 여부를 포함한 상세 결과

이 실험은 SoT가 외부 설명 대신 내부 추론에 집중함으로써 CoT보다 더 높은 정확도를 달성하는지 확인합니다.

## SoT 가설

SoT는 CoT의 한계에서 영감을 얻었습니다:
- **CoT 한계**: 명시적 추론 단계 생성은 사용자 이해에 초점을 맞춰 최적의 문제 해결을 방해할 수 있으며, 복잡한 작업에서 오류나 비효율성을 유발할 수 있습니다.
- **SoT 장점**: "Think silently" 지시로 내부 다단계 추론을 장려하면서 설명 생성 부담을 줄여, 계산 자원을 추론에 집중시켜 정확도를 높이고 오류를 줄일 가능성이 있습니다.

이 실험은 MATH 데이터셋에서 SoT와 CoT의 성능을 비교하여 이 가설을 검증합니다.
