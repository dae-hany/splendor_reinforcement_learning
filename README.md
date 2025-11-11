# Splendor Solo Play - Reinforcement Learning Project

Splendor 보드게임의 솔로 플레이 버전을 위한 강화학습 프로젝트입니다. DQN (Deep Q-Network) 알고리즘을 사용하여 에이전트를 학습시킵니다.

---

## 📁 프로젝트 구조

### 🎯 핵심 파일 (필수)

| 파일명 | 설명 | 용도 |
|--------|------|------|
| **`splendor_solo_env.py`** | Splendor 솔로 게임 환경 | Gymnasium 환경 정의, 게임 로직 구현 |
| **`train_dqn_splendor.py`** | DQN 학습 스크립트 | 모델 정의 및 학습 실행 |
| **`evaluate_model.py`** | 모델 평가 스크립트 | 학습된 모델의 성능 평가 |
| **`cards.csv`** | 카드 데이터 | 게임에 사용되는 모든 카드 정보 |
| **`noble_tiles.csv`** | 귀족 타일 데이터 | 게임에 사용되는 귀족 타일 정보 |
| **`splendor_soloplay_rulebook.txt`** | 게임 규칙 | Splendor 솔로 플레이 규칙 |

### 🛠️ 유틸리티 파일 (선택)

| 파일명 | 설명 | 삭제 가능 여부 |
|--------|------|----------------|
| `evaluate_random.py` | 무작위 정책 평가 (베이스라인) | ⚠️ 권장 유지 |
| `analyze_training_logs.py` | TensorBoard 로그 분석 | ⚠️ 권장 유지 |
| `test_env.py` | 환경 테스트 스크립트 | ✅ 삭제 가능 |
| `test_dqn_setup.py` | DQN 설정 테스트 | ✅ 삭제 가능 |
| `test_token_limit.py` | 토큰 제한 테스트 | ✅ 삭제 가능 |
| `verify_fixes.py` | 버그 수정 검증 | ✅ 삭제 가능 |
| `debug_info_dict.py` | 디버그 스크립트 | ✅ 삭제 가능 |
| `quick_test.py` | 빠른 테스트 | ✅ 삭제 가능 |
| `run_minimal_test.py` | 최소 테스트 | ✅ 삭제 가능 |

### 📄 문서 파일

| 파일명 | 설명 | 삭제 가능 여부 |
|--------|------|----------------|
| **`README.md`** (이 파일) | 프로젝트 메인 문서 | 🔴 필수 |
| `MODEL_EVALUATION_REPORT.md` | 모델 평가 결과 | ⚠️ 참고용 유지 권장 |
| `DQN_TRAINING_GUIDE.md` | DQN 학습 가이드 | ✅ 삭제 가능 (README에 통합) |
| `IMPLEMENTATION_SUMMARY.md` | 구현 요약 | ✅ 삭제 가능 |
| `QUICK_REFERENCE.md` | 빠른 참조 | ✅ 삭제 가능 |
| `TOKEN_LIMIT_FIX.md` | 토큰 제한 수정 문서 | ✅ 삭제 가능 |

### 📂 디렉토리

- **`runs/`**: 학습 결과 및 TensorBoard 로그 저장
  - 각 학습 실행마다 별도 폴더 생성
  - `.pth` 파일: 저장된 모델 가중치
  - `events.out.tfevents.*`: TensorBoard 로그
- **`__pycache__/`**: Python 캐시 (자동 생성)

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 필요한 패키지 설치
pip install gymnasium numpy pandas torch tensorboard
```

### 2. 환경 테스트

```python
import gymnasium as gym
import splendor_solo_env

# 환경 등록
gym.register(
    id="SplendorSolo-v0",
    entry_point="splendor_solo_env:SplendorSoloEnv",
)

# 환경 생성 및 테스트
env = gym.make("SplendorSolo-v0")
obs, info = env.reset()
print("관찰 공간:", env.observation_space)
print("행동 공간:", env.action_space)
```

### 3. 모델 학습

```bash
# 기본 학습 (500,000 타임스텝)
python train_dqn_splendor.py

# 학습 시간 조정
python train_dqn_splendor.py --total-timesteps 1000000

# GPU 사용 비활성화
python train_dqn_splendor.py --cuda False

# 주요 하이퍼파라미터 조정
python train_dqn_splendor.py \
    --learning-rate 0.00025 \
    --buffer-size 100000 \
    --batch-size 128 \
    --gamma 0.99
```

### 4. 학습 모니터링

```bash
# TensorBoard 실행
tensorboard --logdir runs/

# 브라우저에서 http://localhost:6006 접속
```

### 5. 모델 평가

```bash
# 학습된 모델 평가 (100 에피소드)
python evaluate_model.py --model-path "runs/[RUN_NAME]/SplendorSolo-v0.pth" --n-eval-episodes 100

# 무작위 베이스라인과 비교
python evaluate_random.py

# 학습 로그 분석
python analyze_training_logs.py "runs/[RUN_NAME]"
```

---

## 🎮 게임 환경 상세

### Splendor Solo Play 규칙

- **목표**: 17턴 이내에 15점 이상 획득
- **행동 공간**: 39가지 행동
  - 0-9: 3개의 서로 다른 보석 가져오기
  - 10-14: 2개의 같은 보석 가져오기
  - 15-23: 공개된 카드 구매
  - 24-26: 보유 중인 카드 구매
  - 27-35: 공개된 카드 예약
  - 36-38: 덱 상단 카드 예약

### 관찰 공간 (Dict)

| 요소 | 크기 | 설명 |
|------|------|------|
| `bank_tokens` | (6,) | 은행의 토큰 [흑, 백, 적, 청, 녹, 금] |
| `player_tokens` | (6,) | 플레이어의 토큰 |
| `player_bonuses` | (5,) | 플레이어의 보너스 [흑, 백, 적, 청, 녹] |
| `player_points` | (1,) | 플레이어의 점수 |
| `player_reserved` | (21,) | 보유 카드 3장 (각 7차원) |
| `face_up_cards` | (63,) | 공개 카드 9장 (각 7차원) |
| `noble_tiles` | (12,) | 귀족 타일 2개 (각 6차원) |
| `game_clock` | (1,) | 남은 턴 수 |

**총 차원**: 115

---

## 🧠 DQN 모델 구조

### Q-Network 아키텍처

```
입력 (115차원) 
    ↓
Linear(115 → 256) + ReLU
    ↓
Linear(256 → 256) + ReLU
    ↓
Linear(256 → 39)
    ↓
출력 (39차원 Q-values)
```

### 주요 기능

1. **Action Masking**: 불법 행동의 Q-value를 -inf로 설정
2. **Experience Replay**: 100,000개의 전환 저장
3. **Target Network**: 안정적인 학습을 위한 타겟 네트워크
4. **Epsilon-Greedy Exploration**: 선형 감소 (1.0 → 0.05)

### 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `total_timesteps` | 500,000 | 총 학습 스텝 |
| `learning_rate` | 0.00025 | 학습률 |
| `buffer_size` | 100,000 | 리플레이 버퍼 크기 |
| `batch_size` | 128 | 배치 크기 |
| `gamma` | 0.99 | 할인 계수 |
| `tau` | 1.0 | 타겟 네트워크 업데이트 비율 |
| `target_network_frequency` | 500 | 타겟 업데이트 주기 |
| `exploration_fraction` | 0.5 | Epsilon 감소 기간 비율 |

---

## 📊 평가 지표

### 성능 메트릭

- **평균 리턴**: 에피소드당 누적 보상
- **평균 점수**: 게임 종료 시 획득한 점수
- **승리율**: 15점 이상 달성 비율
- **에피소드 길이**: 게임이 지속된 턴 수

### 현재 성능 (최신 평가)

```
학습된 모델 (runs/.../SplendorSolo-v0.pth):
  - 평균 점수: 1.06 ± 1.72
  - 승리율: 0.0%
  - 평균 리턴: 1.06

무작위 정책 (베이스라인):
  - 평균 점수: 1.45 ± 1.43
  - 승리율: 0.0%
  - 평균 리턴: 1.45

⚠️ 현재 모델은 제대로 학습되지 않았습니다 (5 에피소드만 완료)
```

---

## 🔧 문제 해결

### 학습이 진행되지 않는 경우

1. **학습 시간 증가**
   ```bash
   python train_dqn_splendor.py --total-timesteps 1000000
   ```

2. **TensorBoard로 모니터링**
   ```bash
   tensorboard --logdir runs/
   ```
   - `charts/episodic_return`: 증가하는지 확인
   - `losses/td_loss`: 안정적으로 감소하는지 확인

3. **하이퍼파라미터 튜닝**
   - Learning rate 조정: `--learning-rate 0.0001`
   - Batch size 증가: `--batch-size 256`

### 메모리 부족

```bash
# 버퍼 크기 감소
python train_dqn_splendor.py --buffer-size 50000
```

### GPU 사용 문제

```bash
# CPU로 실행
python train_dqn_splendor.py --cuda False
```

---

## 📈 개선 방안

### 즉시 적용 가능

1. ✅ **더 긴 학습**: `--total-timesteps 2000000` 이상
2. ✅ **보상 함수 개선**: 중간 보상 추가 (카드 구매, 보너스 획득)
3. ✅ **커리큘럼 러닝**: 쉬운 목표부터 시작 (5점 → 10점 → 15점)

### 장기적 개선

1. 🔄 **알고리즘 변경**: PPO, A3C, Rainbow DQN
2. 🔄 **네트워크 구조**: Attention, Transformer 기반 모델
3. 🔄 **상태 표현**: 더 효율적인 인코딩 방식

---

## 📝 파일 정리 권장사항

### 최소 구성으로 정리하기

**꼭 유지해야 할 파일:**
```
splendor_reinforcement_learning/
├── splendor_solo_env.py          # 환경 정의
├── train_dqn_splendor.py          # 모델 학습
├── evaluate_model.py              # 모델 평가
├── evaluate_random.py             # 베이스라인 평가
├── analyze_training_logs.py       # 로그 분석
├── cards.csv                      # 카드 데이터
├── noble_tiles.csv                # 귀족 데이터
├── splendor_soloplay_rulebook.txt # 게임 규칙
├── README.md                      # 이 문서
└── MODEL_EVALUATION_REPORT.md     # 평가 결과
```

**삭제 가능한 파일:**
```
❌ test_env.py
❌ test_dqn_setup.py
❌ test_token_limit.py
❌ verify_fixes.py
❌ debug_info_dict.py
❌ quick_test.py
❌ run_minimal_test.py
❌ DQN_TRAINING_GUIDE.md
❌ IMPLEMENTATION_SUMMARY.md
❌ QUICK_REFERENCE.md
❌ TOKEN_LIMIT_FIX.md
```

---

## 📞 참고 자료

- **Gymnasium**: https://gymnasium.farama.org/
- **CleanRL**: https://github.com/vwxyzjn/cleanrl
- **DQN Paper**: "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)
- **Splendor 게임**: https://boardgamegeek.com/boardgame/148228/splendor

---

## 📄 라이센스

이 프로젝트는 교육 목적으로 작성되었습니다.

---

**마지막 업데이트**: 2025년 11월 11일
