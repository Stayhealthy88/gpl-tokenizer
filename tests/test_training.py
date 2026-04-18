"""
v0.5 학습 파이프라인 End-to-End 테스트
=======================================
합성 데이터 생성 → 모델 학습 → SVG 생성 → 품질 평가.

테스트 구성:
    Suite 1: 합성 데이터셋 — 생성, 배치, 패딩
    Suite 2: GPLTransformer — 구조, forward, loss, 파라미터
    Suite 3: 학습 루프 — 미니 학습 + loss 감소 확인
    Suite 4: SVG 생성 — 무조건/조건부 생성, 유효성
    Suite 5: 평가 메트릭 — 구조/기하학/다양성 점수
    Suite 6: End-to-End 벤치마크 — 전체 파이프라인 통합
"""

import sys
import os
import time

sys.path.insert(0, '..')

import torch
from torch.utils.data import DataLoader

from gpl_tokenizer.tokenizer.vocabulary import GPLVocabulary, SpecialToken
from gpl_tokenizer.tokenizer.arcs import ARCS
from gpl_tokenizer.training.synthetic_dataset import (
    SyntheticSVGDataset, SyntheticSVGGenerator, SVGCollator,
)
from gpl_tokenizer.training.gpl_transformer import (
    GPLTransformer, GPLTransformerConfig,
)
from gpl_tokenizer.training.trainer import GPLTrainer, TrainingConfig
from gpl_tokenizer.training.generator import GPLGenerator
from gpl_tokenizer.training.evaluator import GPLEvaluator


# ===================== 설정 =====================

VOCAB = GPLVocabulary(max_coord_level=6)
ARCS_INST = ARCS(max_level=6)

passed = 0
failed = 0


def check(condition: bool, message: str):
    global passed, failed
    if condition:
        print(f"  ✓ {message}")
        passed += 1
    else:
        print(f"  ✗ {message}")
        failed += 1


# ============================================================
# TEST SUITE 1: 합성 데이터셋
# ============================================================

print("=" * 60)
print("TEST SUITE 1: 합성 데이터셋")
print("=" * 60)

# Test 1.1: 데이터 생성기
gen = SyntheticSVGGenerator(VOCAB, ARCS_INST, seed=42)
samples = gen.generate_batch(100)
check(len(samples) == 100, f"100개 샘플 생성 (실제: {len(samples)})")

# Test 1.2: 카테고리 다양성
categories = set(s.category for s in samples)
check(len(categories) >= 3, f"카테고리 수: {len(categories)} ({', '.join(categories)})")

# Test 1.3: 토큰 시퀀스 유효성
valid_seqs = sum(1 for s in samples
                 if s.token_ids[0] == SpecialToken.BOS
                 and s.token_ids[-1] == SpecialToken.EOS)
check(valid_seqs == 100, f"BOS/EOS 완전성: {valid_seqs}/100")

# Test 1.4: 토큰 ID 범위
all_ids = [tid for s in samples for tid in s.token_ids]
max_id = max(all_ids)
check(max_id < VOCAB.vocab_size, f"최대 토큰 ID({max_id}) < vocab_size({VOCAB.vocab_size})")

# Test 1.5: Dataset 클래스
dataset = SyntheticSVGDataset(VOCAB, ARCS_INST, n_samples=200, seed=42)
check(len(dataset) > 0, f"Dataset 크기: {len(dataset)}")

item = dataset[0]
check("token_ids" in item, "Dataset 항목에 token_ids 존재")
check(isinstance(item["token_ids"], torch.Tensor), "token_ids는 Tensor")

# Test 1.6: Collator + DataLoader
collator = SVGCollator(VOCAB)
loader = DataLoader(dataset, batch_size=16, collate_fn=collator)
batch = next(iter(loader))
check("input_ids" in batch, "배치에 input_ids 존재")
check("target_ids" in batch, "배치에 target_ids 존재")
check("attention_mask" in batch, "배치에 attention_mask 존재")
check(batch["input_ids"].shape[0] == 16, f"배치 크기: {batch['input_ids'].shape[0]}")
check(batch["input_ids"].shape == batch["target_ids"].shape, "input/target 형태 일치")


# ============================================================
# TEST SUITE 2: GPLTransformer 모델
# ============================================================

print()
print("=" * 60)
print("TEST SUITE 2: GPLTransformer 모델")
print("=" * 60)

# Test 2.1: 모델 생성
config = GPLTransformerConfig(
    d_model=128, d_type=16, d_coord=32,
    n_heads=4, n_layers=4, d_ff=512,
    max_seq_len=64, dropout=0.1,
)
model = GPLTransformer(VOCAB, config)
check(model is not None, "GPLTransformer 생성 성공")

# Test 2.2: 파라미터 수
params = model.count_parameters()
check(1_000_000 <= params["total"] <= 5_000_000,
      f"Micro 모델 범위: {params['total']:,} params")
print(f"    - Embedding: {params['embedding']:,}")
print(f"    - Decoder:   {params['decoder']:,}")

# Test 2.3: Forward pass
test_input = batch["input_ids"][:4, :32]
logits = model(test_input)
check(logits.shape == (4, 32, VOCAB.vocab_size),
      f"logits 형태: {logits.shape}")

# Test 2.4: Loss 계산
result = model.compute_loss(
    input_ids=batch["input_ids"][:4, :32],
    target_ids=batch["target_ids"][:4, :32],
    attention_mask=batch["attention_mask"][:4, :32],
)
check(result["loss"].item() > 0, f"loss > 0: {result['loss'].item():.4f}")
check(0 <= result["accuracy"].item() <= 1,
      f"accuracy ∈ [0,1]: {result['accuracy'].item():.4f}")

# Test 2.5: Weight tying
check(model.lm_head.weight is model.embedding.token_embedding.weight,
      "LM head - embedding weight tying 확인")

# Test 2.6: Gradient 흐름
result["loss"].backward()
has_grad = any(p.grad is not None and p.grad.norm() > 0
               for p in model.parameters() if p.requires_grad)
check(has_grad, "역전파 gradient 흐름 확인")
model.zero_grad()


# ============================================================
# TEST SUITE 3: 학습 루프 (Mini Training)
# ============================================================

print()
print("=" * 60)
print("TEST SUITE 3: 학습 루프 (Mini Training)")
print("=" * 60)

# 소규모 학습 데이터
mini_dataset = SyntheticSVGDataset(VOCAB, ARCS_INST, n_samples=200, max_seq_len=64, seed=123)
mini_collator = SVGCollator(VOCAB, max_seq_len=64)
train_loader = DataLoader(mini_dataset, batch_size=32, collate_fn=mini_collator, shuffle=True)

# 검증 데이터
val_dataset = SyntheticSVGDataset(VOCAB, ARCS_INST, n_samples=50, max_seq_len=64, seed=456)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=mini_collator)

# 모델 재생성 (깨끗한 상태)
model = GPLTransformer(VOCAB, config)
train_config = TrainingConfig(
    epochs=10,
    batch_size=32,
    learning_rate=1e-3,
    checkpoint_dir="/tmp/gpl_test_ckpt",
    save_every=5,
    eval_every=2,
    patience=20,  # 테스트에서는 early stopping 비활성화
)
trainer = GPLTrainer(model, train_config)

# Test 3.1: 학습 실행
t0 = time.time()
history = trainer.train(train_loader, val_loader, verbose=False)
elapsed = time.time() - t0
check(len(history["train_loss"]) == 10, f"10 에폭 학습 완료 ({elapsed:.1f}초)")

# Test 3.2: Loss 감소
first_loss = history["train_loss"][0]
last_loss = history["train_loss"][-1]
check(last_loss < first_loss,
      f"loss 감소: {first_loss:.4f} → {last_loss:.4f}")

# Test 3.3: 검증 loss 존재
check(len(history["val_loss"]) > 0, f"검증 loss 기록: {len(history['val_loss'])}개")

# Test 3.4: 정확도 개선
first_acc = history["train_acc"][0]
last_acc = history["train_acc"][-1]
check(last_acc > first_acc,
      f"정확도 개선: {first_acc:.3f} → {last_acc:.3f}")

# Test 3.5: 체크포인트 저장
ckpt_path = os.path.join(train_config.checkpoint_dir, "final.pt")
check(os.path.exists(ckpt_path), "체크포인트 저장 확인")

# Test 3.6: 체크포인트 로드
model2 = GPLTransformer(VOCAB, config)
trainer2 = GPLTrainer(model2, train_config)
trainer2.load_checkpoint("final.pt")
check(trainer2.state.epoch == 10, "체크포인트 로드: epoch 복원")


# ============================================================
# TEST SUITE 4: SVG 생성
# ============================================================

print()
print("=" * 60)
print("TEST SUITE 4: SVG 생성")
print("=" * 60)

generator = GPLGenerator(model, VOCAB, ARCS_INST)

# Test 4.1: 무조건 생성
result = generator.generate_unconditional(max_len=32, temperature=0.8)
check(len(result.token_ids) >= 2, f"생성 토큰 수: {result.n_tokens}")
check(result.token_ids[0] == SpecialToken.BOS, "BOS로 시작")

# Test 4.2: 도형 지정 생성
circle_result = generator.generate_shape("circle", max_len=16)
check(len(circle_result.token_ids) >= 2, f"원 생성: {circle_result.n_tokens} 토큰")

# Test 4.3: 배치 생성
batch_results = generator.generate_batch(n=10, mode="unconditional", max_len=32)
check(len(batch_results) == 10, f"10개 일괄 생성")

# Test 4.4: SVG 문자열 생성
has_svg = sum(1 for r in batch_results if '<svg' in r.svg_full)
check(has_svg == 10, f"SVG 래핑 완료: {has_svg}/10")

# Test 4.5: 부분 시퀀스 완성
partial = [SpecialToken.BOS, 20]  # [BOS][CIRCLE]
completion = generator.generate_completion(partial, max_len=16)
check(completion.token_ids[0] == SpecialToken.BOS, "완성: BOS 보존")
check(completion.token_ids[1] == 20, "완성: CIRCLE 보존")


# ============================================================
# TEST SUITE 5: 평가 메트릭
# ============================================================

print()
print("=" * 60)
print("TEST SUITE 5: 평가 메트릭")
print("=" * 60)

evaluator = GPLEvaluator()

# 생성 결과 평가
eval_samples = generator.generate_batch(n=30, mode="unconditional", max_len=32)
metrics = evaluator.evaluate(eval_samples)

check(metrics["n_samples"] == 30, f"평가 샘플 수: {metrics['n_samples']}")
check(0 <= metrics["valid_svg_rate"] <= 1,
      f"Valid SVG Rate: {metrics['valid_svg_rate']:.1%}")
check(0 <= metrics["structural_score"] <= 1,
      f"Structural Score: {metrics['structural_score']:.3f}")
check(0 <= metrics["geometric_score"] <= 1,
      f"Geometric Score: {metrics['geometric_score']:.3f}")
check(0 <= metrics["diversity_score"] <= 1,
      f"Diversity Score: {metrics['diversity_score']:.3f}")
check(metrics["avg_token_length"] > 0,
      f"평균 토큰 길이: {metrics['avg_token_length']:.1f}")

# 카테고리 분포
cats = metrics.get("categories", {})
check(len(cats) >= 1, f"감지된 카테고리: {len(cats)}개")

# 평가 보고서 출력
evaluator.print_report(metrics)


# ============================================================
# TEST SUITE 6: End-to-End 벤치마크
# ============================================================

print()
print("=" * 60)
print("TEST SUITE 6: End-to-End 벤치마크")
print("=" * 60)

print(f"\n  {'Component':<30} {'Detail':>25}")
print("  " + "-" * 57)
print(f"  {'Vocab size':<30} {VOCAB.vocab_size:>25,}")
print(f"  {'Model params':<30} {params['total']:>25,}")
print(f"  {'Training samples':<30} {len(mini_dataset):>25,}")
print(f"  {'Training epochs':<30} {10:>25}")
print(f"  {'Training time':<30} {elapsed:>24.1f}s")
print(f"  {'Final train loss':<30} {last_loss:>25.4f}")
print(f"  {'Final train acc':<30} {last_acc:>25.3f}")
print(f"  {'Generated samples':<30} {30:>25}")
print(f"  {'Valid SVG rate':<30} {metrics['valid_svg_rate']:>24.1%}")
print(f"  {'Structural score':<30} {metrics['structural_score']:>25.3f}")
print(f"  {'Geometric score':<30} {metrics['geometric_score']:>25.3f}")

# 전체 시스템 체크
check(params["total"] < 5_000_000, "Micro 모델 크기 준수 (< 5M)")
check(last_loss < first_loss, "학습 수렴 확인")
check(metrics["structural_score"] > 0, "구조 점수 > 0")


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
total = passed + failed
print(f"\n  Total: {total} | Passed: {passed} | Failed: {failed}")

if failed == 0:
    print(f"\n  ★ ALL {total} TESTS PASSED ★")
else:
    print(f"\n  ✗ {failed} TESTS FAILED")

# 임시 파일 정리
import shutil
if os.path.exists("/tmp/gpl_test_ckpt"):
    shutil.rmtree("/tmp/gpl_test_ckpt")

sys.exit(0 if failed == 0 else 1)
