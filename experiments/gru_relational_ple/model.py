"""GRU + relational graph attention + progressive layered extraction model."""

from __future__ import annotations

import torch
from torch import nn

from experiments.relgt.data import LABELS
from experiments.relgt.model import RelGTInspired


class Expert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class PLELayer(nn.Module):
    """One progressive extraction layer with shared and task-specific experts."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_tasks: int,
        num_shared_experts: int,
        num_task_experts: int,
        dropout: float,
        return_shared: bool,
    ) -> None:
        super().__init__()
        self.num_shared_experts = num_shared_experts
        self.num_task_experts = num_task_experts
        self.return_shared = return_shared
        self.shared_experts = nn.ModuleList(
            Expert(input_dim, hidden_dim, dropout)
            for _ in range(num_shared_experts)
        )
        self.task_experts = nn.ModuleList(
            nn.ModuleList(
                Expert(input_dim, hidden_dim, dropout)
                for _ in range(num_task_experts)
            )
            for _ in range(num_tasks)
        )
        self.task_gates = nn.ModuleList(
            nn.Linear(input_dim, num_shared_experts + num_task_experts)
            for _ in range(num_tasks)
        )
        self.shared_gate = (
            nn.Linear(
                input_dim, num_shared_experts + num_tasks * num_task_experts
            )
            if return_shared
            else None
        )

    def forward(
        self,
        shared_input: torch.Tensor,
        task_inputs: list[torch.Tensor],
    ) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
        shared_outputs = [expert(shared_input) for expert in self.shared_experts]
        task_outputs = [
            [expert(task_input) for expert in experts]
            for task_input, experts in zip(task_inputs, self.task_experts)
        ]

        task_representations = []
        for task_index, task_input in enumerate(task_inputs):
            candidates = torch.stack(
                [*shared_outputs, *task_outputs[task_index]], dim=1
            )
            weights = torch.softmax(
                self.task_gates[task_index](task_input), dim=-1
            ).unsqueeze(-1)
            task_representations.append((candidates * weights).sum(dim=1))

        shared_representation = None
        if self.shared_gate is not None:
            all_experts = [
                *shared_outputs,
                *(output for outputs in task_outputs for output in outputs),
            ]
            candidates = torch.stack(all_experts, dim=1)
            weights = torch.softmax(
                self.shared_gate(shared_input), dim=-1
            ).unsqueeze(-1)
            shared_representation = (candidates * weights).sum(dim=1)
        return shared_representation, task_representations


class GRURelationalAttentionPLE(nn.Module):
    """Fuse 90-day action sequences and relational graph context for six tasks."""

    def __init__(
        self,
        num_entities: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_graph_layers: int = 2,
        num_gru_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.graph_encoder = RelGTInspired(
            num_entities=num_entities,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_local_layers=num_graph_layers,
            dropout=dropout,
        )
        self.sequence_projection = nn.Linear(len(LABELS), hidden_dim)
        self.sequence_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            dropout=dropout if num_gru_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.first_extraction = PLELayer(
            hidden_dim,
            hidden_dim,
            num_tasks=len(LABELS),
            num_shared_experts=2,
            num_task_experts=2,
            dropout=dropout,
            return_shared=True,
        )
        self.final_extraction = PLELayer(
            hidden_dim,
            hidden_dim,
            num_tasks=len(LABELS),
            num_shared_experts=2,
            num_task_experts=2,
            dropout=dropout,
            return_shared=False,
        )
        self.task_towers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in LABELS
        )

    def forward(
        self,
        entity_ids: torch.Tensor,
        node_type_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        hop_ids: torch.Tensor,
        node_features: torch.Tensor,
        relative_days: torch.Tensor,
        daily_activity: torch.Tensor,
    ) -> torch.Tensor:
        if daily_activity.ndim != 3 or daily_activity.shape[-1] != len(LABELS):
            raise ValueError(
                "daily_activity must have shape (batch, days, number of labels)."
            )

        graph_representation = self.graph_encoder.encode(
            entity_ids,
            node_type_ids,
            relation_ids,
            hop_ids,
            node_features,
            relative_days,
        )
        sequence = torch.nn.functional.gelu(
            self.sequence_projection(daily_activity)
        )
        _, hidden = self.sequence_gru(sequence)
        sequence_representation = hidden[-1]
        fused = self.fusion(
            torch.cat([graph_representation, sequence_representation], dim=-1)
        )

        shared, tasks = self.first_extraction(fused, [fused] * len(LABELS))
        if shared is None:
            raise RuntimeError("The first PLE layer must return a shared stream.")
        _, tasks = self.final_extraction(shared, tasks)
        return torch.cat(
            [
                tower(task_representation)
                for tower, task_representation in zip(self.task_towers, tasks)
            ],
            dim=1,
        )
