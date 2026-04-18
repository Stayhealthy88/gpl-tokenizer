"""
GPL 학습 루프
=============
GPLTransformer의 학습/검증/체크포인트를 관리하는 Trainer.

특징:
    - Cosine annealing LR scheduler
    - Gradient clipping
    - 학습/검증 loss 추적
    - 체크포인트 저장/로드
    - Early stopping (선택)
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .gpl_transformer import GPLTransformer


@dataclass
class TrainingConfig:
    """학습 설정."""
    # 기본
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01

    # 스케줄러
    warmup_steps: int = 100
    min_lr: float = 1e-5

    # 안정성
    grad_clip: float = 1.0
    label_smoothing: float = 0.0

    # 체크포인트
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5          # N 에폭마다 저장
    eval_every: int = 1          # N 에폭마다 검증

    # Early stopping
    patience: int = 5
    min_delta: float = 0.001

    # 로깅
    log_every: int = 50          # N 스텝마다 로그


@dataclass
class TrainingState:
    """학습 상태 추적."""
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float('inf')
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_accs: List[float] = field(default_factory=list)
    val_accs: List[float] = field(default_factory=list)
    patience_counter: int = 0


class GPLTrainer:
    """
    GPLTransformer 학습기.

    사용법:
        trainer = GPLTrainer(model, config)
        history = trainer.train(train_loader, val_loader)
    """

    def __init__(self, model: GPLTransformer, config: TrainingConfig):
        self.model = model
        self.config = config
        self.state = TrainingState()

        # 옵티마이저
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        # LR 스케줄러: linear warmup + cosine decay
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.min_lr,
        )

        # 체크포인트 디렉토리
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              verbose: bool = True,
              ) -> Dict[str, List[float]]:
        """
        전체 학습 루프 실행.

        Args:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더 (선택)
            verbose: 로그 출력 여부
        Returns:
            {"train_loss": [...], "val_loss": [...], "train_acc": [...], "val_acc": [...]}
        """
        if verbose:
            params = self.model.count_parameters()
            print(f"\n{'='*60}")
            print(f"GPL Training Pipeline v0.5")
            print(f"{'='*60}")
            print(f"  Model params:     {params['total']:,}")
            print(f"  Epochs:           {self.config.epochs}")
            print(f"  Batch size:       {self.config.batch_size}")
            print(f"  Learning rate:    {self.config.learning_rate}")
            print(f"  Train batches:    {len(train_loader)}")
            if val_loader:
                print(f"  Val batches:      {len(val_loader)}")
            print(f"{'='*60}\n")

        for epoch in range(self.config.epochs):
            self.state.epoch = epoch + 1

            # 학습
            train_loss, train_acc = self._train_epoch(train_loader, verbose)
            self.state.train_losses.append(train_loss)
            self.state.train_accs.append(train_acc)

            # 검증
            val_loss, val_acc = 0.0, 0.0
            if val_loader and (epoch + 1) % self.config.eval_every == 0:
                val_loss, val_acc = self._eval_epoch(val_loader)
                self.state.val_losses.append(val_loss)
                self.state.val_accs.append(val_acc)

                # Early stopping
                if val_loss < self.state.best_val_loss - self.config.min_delta:
                    self.state.best_val_loss = val_loss
                    self.state.patience_counter = 0
                    self.save_checkpoint("best.pt")
                else:
                    self.state.patience_counter += 1

            # LR 스케줄러
            self.scheduler.step()

            # 로깅
            if verbose:
                lr = self.optimizer.param_groups[0]['lr']
                msg = f"  Epoch {epoch+1:3d}/{self.config.epochs}"
                msg += f" | train_loss={train_loss:.4f} acc={train_acc:.3f}"
                if val_loss > 0:
                    msg += f" | val_loss={val_loss:.4f} acc={val_acc:.3f}"
                msg += f" | lr={lr:.2e}"
                print(msg)

            # 체크포인트
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch+1}.pt")

            # Early stopping 체크
            if self.state.patience_counter >= self.config.patience:
                if verbose:
                    print(f"\n  Early stopping at epoch {epoch+1}")
                break

        # 최종 체크포인트
        self.save_checkpoint("final.pt")

        return {
            "train_loss": self.state.train_losses,
            "val_loss": self.state.val_losses,
            "train_acc": self.state.train_accs,
            "val_acc": self.state.val_accs,
        }

    def _train_epoch(self, loader: DataLoader, verbose: bool) -> Tuple[float, float]:
        """1 에폭 학습."""
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(loader):
            self.state.global_step += 1

            # Forward
            result = self.model.compute_loss(
                input_ids=batch["input_ids"],
                target_ids=batch["target_ids"],
                attention_mask=batch["attention_mask"],
            )

            loss = result["loss"]
            acc = result["accuracy"]

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )

            self.optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()
            n_batches += 1

        return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """1 에폭 검증."""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for batch in loader:
            result = self.model.compute_loss(
                input_ids=batch["input_ids"],
                target_ids=batch["target_ids"],
                attention_mask=batch["attention_mask"],
            )
            total_loss += result["loss"].item()
            total_acc += result["accuracy"].item()
            n_batches += 1

        return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)

    def save_checkpoint(self, filename: str):
        """체크포인트 저장."""
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "training_state": {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "best_val_loss": self.state.best_val_loss,
                "train_losses": self.state.train_losses,
                "val_losses": self.state.val_losses,
            },
            "config": self.config,
        }, path)

    def load_checkpoint(self, filename: str):
        """체크포인트 로드."""
        path = os.path.join(self.config.checkpoint_dir, filename)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        state = ckpt["training_state"]
        self.state.epoch = state["epoch"]
        self.state.global_step = state["global_step"]
        self.state.best_val_loss = state["best_val_loss"]
        self.state.train_losses = state["train_losses"]
        self.state.val_losses = state["val_losses"]
