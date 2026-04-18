"""
v0.4 GPLEmbedding + HMN 초기화 테스트
======================================
PyTorch 임베딩 레이어의 구조, 초기화, forward pass, 기하학적 속성 검증.

테스트 구성:
    Suite 1: GPLEmbedding 기본 — 생성, forward pass, 출력 형태
    Suite 2: HMN 초기화 — PAD 영벡터, 좌표 공간 유사성, 연속성 순서
    Suite 3: 좌표 구조 인코딩 — 가까운 좌표 vs 먼 좌표
    Suite 4: 통합 벤치마크 — 파라미터 수, HMN vs Random 비교
"""

import sys
import math

sys.path.insert(0, '..')

import torch
import torch.nn.functional as F

from gpl_tokenizer.tokenizer.vocabulary import GPLVocabulary, SpecialToken, CommandToken
from gpl_tokenizer.embedding.gpl_embedding import GPLEmbedding, _token_type_id, N_TOKEN_TYPES
from gpl_tokenizer.embedding.hmn_init import HMNInitializer


# ===================== 설정 =====================

VOCAB = GPLVocabulary(max_coord_level=6)
D_MODEL = 128   # 테스트용 작은 차원
D_TYPE = 16
D_COORD = 32

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
# TEST SUITE 1: GPLEmbedding 기본
# ============================================================

print("=" * 60)
print("TEST SUITE 1: GPLEmbedding 기본")
print("=" * 60)

# Test 1.1: 생성
emb = GPLEmbedding(VOCAB, d_model=D_MODEL, d_type=D_TYPE,
                    d_coord=D_COORD, use_hmn_init=True)
check(emb is not None, "GPLEmbedding 생성 성공")

# Test 1.2: 파라미터 존재
total_params = sum(p.numel() for p in emb.parameters())
check(total_params > 0, f"총 파라미터: {total_params:,}")

# Test 1.3: Forward pass — 단일 시퀀스
token_ids = torch.tensor([[1, 10, 150, 200, 31, 42, 17, 2]])  # BOS, MOVE, coords, G0, κ2, CLOSE, EOS
output = emb(token_ids)
check(output.shape == (1, 8, D_MODEL), f"출력 형태: {output.shape} (기대: (1, 8, {D_MODEL}))")

# Test 1.4: Forward pass — 배치
batch = torch.tensor([
    [1, 10, 150, 17, 2, 0, 0, 0],
    [1, 20, 200, 300, 17, 2, 0, 0],
])
batch_out = emb(batch)
check(batch_out.shape == (2, 8, D_MODEL), f"배치 출력: {batch_out.shape}")

# Test 1.5: PAD 토큰 임베딩이 영벡터에 가까움
pad_ids = torch.tensor([[0, 0, 0]])
pad_out = emb.token_embedding(pad_ids)
pad_norm = pad_out.norm().item()
check(pad_norm < 0.01, f"PAD 임베딩 노름 ≈ 0 (실제: {pad_norm:.6f})")

# Test 1.6: 토큰 유형 매핑 정확성
check(_token_type_id(0) == 0, "PAD → type 0 (special)")
check(_token_type_id(10) == 1, "MOVE → type 1 (command)")
check(_token_type_id(20) == 2, "CIRCLE → type 2 (composite)")
check(_token_type_id(31) == 3, "G0 → type 3 (continuity)")
check(_token_type_id(42) == 4, "κ2 → type 4 (curvature)")
check(_token_type_id(60) == 5, "ALIGN → type 5 (spatial)")
check(_token_type_id(150) == 6, "coord → type 6 (coord)")


# ============================================================
# TEST SUITE 2: HMN 초기화 속성
# ============================================================

print()
print("=" * 60)
print("TEST SUITE 2: HMN 초기화 속성")
print("=" * 60)

w = emb.token_embedding.weight.detach()

# Test 2.1: 좌표 토큰 — 가까운 좌표가 유사한 임베딩
# Level 4에서 인접한 좌표 2개 vs 먼 좌표 2개
near_id1 = VOCAB.coord_to_id(4, 8, 8)
near_id2 = VOCAB.coord_to_id(4, 9, 8)
far_id = VOCAB.coord_to_id(4, 0, 0)

if near_id1 and near_id2 and far_id:
    sim_near = F.cosine_similarity(w[near_id1].unsqueeze(0), w[near_id2].unsqueeze(0)).item()
    sim_far = F.cosine_similarity(w[near_id1].unsqueeze(0), w[far_id].unsqueeze(0)).item()
    check(sim_near > sim_far,
          f"인접 좌표 유사도({sim_near:.3f}) > 원거리 유사도({sim_far:.3f})")

# Test 2.2: 같은 레벨 좌표들의 노름이 유사
level3_ids = [VOCAB.coord_to_id(3, qx, qy) for qx in range(8) for qy in range(8)]
level3_ids = [tid for tid in level3_ids if tid is not None]
level3_norms = [w[tid].norm().item() for tid in level3_ids]
if level3_norms:
    norm_std = torch.tensor(level3_norms).std().item()
    norm_mean = torch.tensor(level3_norms).mean().item()
    check(norm_std < norm_mean * 0.5,
          f"Level 3 노름: mean={norm_mean:.3f}, std={norm_std:.3f} (std < mean×0.5)")

# Test 2.3: 높은 레벨(세밀) 좌표가 낮은 레벨(거친) 좌표보다 작은 노름
level1_ids = [VOCAB.coord_to_id(1, qx, qy) for qx in range(2) for qy in range(2)]
level5_ids = [VOCAB.coord_to_id(5, qx, qy) for qx in range(4) for qy in range(4)]
level1_ids = [tid for tid in level1_ids if tid is not None]
level5_ids = [tid for tid in level5_ids if tid is not None]

if level1_ids and level5_ids:
    l1_mean_norm = torch.tensor([w[tid].norm().item() for tid in level1_ids]).mean().item()
    l5_mean_norm = torch.tensor([w[tid].norm().item() for tid in level5_ids]).mean().item()
    check(l1_mean_norm > l5_mean_norm,
          f"Level 1 노름({l1_mean_norm:.3f}) > Level 5 노름({l5_mean_norm:.3f})")

# Test 2.4: 연속성 토큰 순서 보존 — G0~G2가 순서대로 이동
# DISC(30), G0(31), G1(32), G2(33)
cont_vecs = [w[30], w[31], w[32], w[33]]
# G0과 G1의 거리 < DISC와 G2의 거리 (중간은 가깝고 극단은 멀다)
dist_g0_g1 = (cont_vecs[1] - cont_vecs[2]).norm().item()
dist_disc_g2 = (cont_vecs[0] - cont_vecs[3]).norm().item()
check(dist_g0_g1 < dist_disc_g2,
      f"G0↔G1 거리({dist_g0_g1:.3f}) < DISC↔G2 거리({dist_disc_g2:.3f})")

# Test 2.5: 유사 명령어 클러스터링 — LINE/HLINE/VLINE 거리 < LINE/CUBIC 거리
dist_line_hline = (w[11] - w[12]).norm().item()  # LINE - HLINE
dist_line_cubic = (w[11] - w[14]).norm().item()   # LINE - CUBIC
check(dist_line_hline < dist_line_cubic,
      f"LINE↔HLINE({dist_line_hline:.3f}) < LINE↔CUBIC({dist_line_cubic:.3f})")

# Test 2.6: 유사 도형 클러스터링 — CIRCLE/ELLIPSE 거리 < CIRCLE/RECT 거리
dist_circ_elli = (w[20] - w[21]).norm().item()
dist_circ_rect = (w[20] - w[22]).norm().item()
check(dist_circ_elli < dist_circ_rect,
      f"CIRCLE↔ELLIPSE({dist_circ_elli:.3f}) < CIRCLE↔RECT({dist_circ_rect:.3f})")


# ============================================================
# TEST SUITE 3: 좌표 구조 인코딩
# ============================================================

print()
print("=" * 60)
print("TEST SUITE 3: 좌표 구조 인코딩 (CoordStructureEncoder)")
print("=" * 60)

# 좌표 토큰 vs 비좌표 토큰의 구조 인코딩 차이
coord_ids = torch.tensor([[150, 200, 300]])   # 좌표 토큰
non_coord_ids = torch.tensor([[1, 10, 31]])    # 비좌표 토큰

coord_enc = emb.coord_encoder(coord_ids)
non_coord_enc = emb.coord_encoder(non_coord_ids)

coord_norm = coord_enc.norm(dim=-1).mean().item()
non_coord_norm = non_coord_enc.norm(dim=-1).mean().item()

check(coord_norm > non_coord_norm,
      f"좌표 인코딩 노름({coord_norm:.3f}) > 비좌표 인코딩 노름({non_coord_norm:.3f})")

# 좌표 구조 인코딩이 공간적 정보를 담는지 확인
# 같은 레벨에서 인접 좌표의 인코딩 vs 먼 좌표의 인코딩
near_coord_ids = torch.tensor([[near_id1, near_id2]])
far_coord_ids = torch.tensor([[near_id1, far_id]])

near_enc = emb.coord_encoder(near_coord_ids)
far_enc = emb.coord_encoder(far_coord_ids)

# 인접 좌표 쌍의 구조 인코딩 유사도
near_sim = F.cosine_similarity(near_enc[0, 0:1], near_enc[0, 1:2]).item()
far_sim = F.cosine_similarity(far_enc[0, 0:1], far_enc[0, 1:2]).item()
check(near_sim > far_sim,
      f"인접 좌표 구조 유사도({near_sim:.3f}) > 원거리({far_sim:.3f})")


# ============================================================
# TEST SUITE 4: 통합 벤치마크
# ============================================================

print()
print("=" * 60)
print("TEST SUITE 4: 통합 벤치마크")
print("=" * 60)

stats = emb.get_embedding_stats()
print(f"\n  {'Metric':<30} {'Value':>15}")
print("  " + "-" * 47)
print(f"  {'Vocab size':<30} {stats['vocab_size']:>15,}")
print(f"  {'d_model':<30} {stats['d_model']:>15}")
print(f"  {'Total parameters':<30} {stats['total_params']:>15,}")
print(f"  {'Trainable parameters':<30} {stats['trainable_params']:>15,}")
print(f"  {'Emb mean norm':<30} {stats['emb_mean_norm']:>15.4f}")
print(f"  {'Emb std norm':<30} {stats['emb_std_norm']:>15.4f}")
print(f"  {'Coord emb mean norm':<30} {stats['coord_emb_mean_norm']:>15.4f}")
print(f"  {'Special emb mean norm':<30} {stats['special_emb_mean_norm']:>15.4f}")

check(stats['total_params'] > 500000, f"총 파라미터 > 500K")
check(stats['trainable_params'] == stats['total_params'],
      "모든 파라미터 학습 가능")

# HMN vs Random 초기화 비교 — 좌표 공간 유사성 보존 테스트
print("\n  [HMN vs Random 초기화 비교]")

# HMN 임베딩
hmn_emb = GPLEmbedding(VOCAB, d_model=D_MODEL, d_type=D_TYPE,
                        d_coord=D_COORD, use_hmn_init=True)
# Random 임베딩
rand_emb = GPLEmbedding(VOCAB, d_model=D_MODEL, d_type=D_TYPE,
                         d_coord=D_COORD, use_hmn_init=False)

# 인접 좌표 쌍 100개의 유사도 평균
hmn_sims = []
rand_sims = []

for qx in range(0, 14):
    for qy in range(0, 14):
        id1 = VOCAB.coord_to_id(4, qx, qy)
        id2 = VOCAB.coord_to_id(4, qx + 1, qy)
        if id1 and id2:
            h_w = hmn_emb.token_embedding.weight.detach()
            r_w = rand_emb.token_embedding.weight.detach()
            hmn_sim = F.cosine_similarity(h_w[id1:id1+1], h_w[id2:id2+1]).item()
            rand_sim = F.cosine_similarity(r_w[id1:id1+1], r_w[id2:id2+1]).item()
            hmn_sims.append(hmn_sim)
            rand_sims.append(rand_sim)

hmn_avg = sum(hmn_sims) / len(hmn_sims) if hmn_sims else 0
rand_avg = sum(rand_sims) / len(rand_sims) if rand_sims else 0

print(f"  HMN 인접 좌표 평균 유사도:    {hmn_avg:.4f}")
print(f"  Random 인접 좌표 평균 유사도:  {rand_avg:.4f}")
check(hmn_avg > rand_avg,
      f"HMN 유사도({hmn_avg:.4f}) > Random 유사도({rand_avg:.4f})")

# Gradient 흐름 확인
print("\n  [Gradient 흐름 확인]")
test_ids = torch.tensor([[1, 10, 150, 200, 17, 2]])
output = emb(test_ids)
loss = output.sum()
loss.backward()

has_grad = emb.token_embedding.weight.grad is not None
check(has_grad, "역전파 gradient 존재")

if has_grad:
    grad_norm = emb.token_embedding.weight.grad.norm().item()
    check(grad_norm > 0, f"gradient 노름 > 0 (실제: {grad_norm:.6f})")


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

sys.exit(0 if failed == 0 else 1)
