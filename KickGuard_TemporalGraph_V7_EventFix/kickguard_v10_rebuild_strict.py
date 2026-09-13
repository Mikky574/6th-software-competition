#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strict-LOSO entry point for KickGuard V10 Baseline-Rebuild.

This wrapper fixes one subtle validation issue in the base V10 file:
router feature selection must be fitted inside each LOSO fold.  The final test
fit may still use all ten wells, but the held-out validation well must not
influence router feature selection or router centroids.

Run this file, not kickguard_v10_rebuild.py.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

import kickguard_v10_rebuild as v10


def strict_loso(X, y, groups, router_features_unused, router_med_unused, args):
    oof_global = np.zeros(len(y), dtype=float)
    oof_moe = np.zeros(len(y), dtype=float)
    rows = []

    for held in np.unique(groups):
        tr = groups != held
        va = groups == held
        available = [int(g) for g in np.unique(groups[tr])]

        # IMPORTANT: router feature selection is fitted on the 9 training wells only.
        fold_router_features, fold_router_med = v10.choose_router_features(
            X[tr], groups[tr], y[tr], max_features=args.router_features
        )

        model = v10.fit_global_and_experts(
            X[tr], y[tr], groups[tr], available,
            seed=8000 + int(held),
            target_boost=args.expert_boost,
            background=args.background_weight,
        )
        pg, pe = v10.predict_global_and_experts(model, X[va])

        cents, scales = v10.router_centroids(
            X[tr], groups[tr], fold_router_features, fold_router_med, available
        )
        W, expert_ids, _ = v10.route_weights(
            X[va], fold_router_features, fold_router_med, cents, scales,
            temperature=args.router_temperature,
            top_experts=args.top_experts,
        )

        pm = np.zeros(int(va.sum()), dtype=float)
        for j, g in enumerate(expert_ids):
            pm += W[:, j] * pe[g]

        oof_global[va] = pg
        oof_moe[va] = pm

        try:
            auc_g = float(roc_auc_score(y[va], pg))
            auc_m = float(roc_auc_score(y[va], pm))
        except Exception:
            auc_g = auc_m = float("nan")

        kdiag = max(int(y[va].sum()), round(int(va.sum()) * 0.38))
        mg = v10.competition_metric(y[va], v10.topk_pred(pg, kdiag))
        mm = v10.competition_metric(y[va], v10.topk_pred(pm, kdiag))

        print(
            f"[V10 STRICT LOSO] WELL_{int(held)+1:06d} "
            f"globalAUC={auc_g:.5f} moeAUC={auc_m:.5f} "
            f"globalScore={mg[0]:.3f} moeScore={mm[0]:.3f} "
            f"routerFeatures={len(fold_router_features)}"
        )
        rows.append({
            "well": int(held)+1,
            "global_auc": auc_g,
            "moe_auc": auc_m,
            "global_score": mg[0],
            "moe_score": mm[0],
            "n": int(va.sum()),
            "pos": int(y[va].sum()),
            "router_features": int(len(fold_router_features)),
        })

    mix = v10.optimize_global_moe_weight(y, oof_global, oof_moe)
    final = (
        mix["global_weight"] * v10.rank01(oof_global)
        + (1.0 - mix["global_weight"]) * v10.rank01(oof_moe)
    )
    final = v10.rank01(final)

    try:
        auc_g = float(roc_auc_score(y, oof_global))
        auc_m = float(roc_auc_score(y, oof_moe))
        auc_f = float(roc_auc_score(y, final))
    except Exception:
        auc_g = auc_m = auc_f = float("nan")

    print("=" * 72)
    print(f"[V10 STRICT OOF] global AUC={auc_g:.6f}")
    print(f"[V10 STRICT OOF] MoE    AUC={auc_m:.6f}")
    print(
        f"[V10 STRICT OOF] blend  AUC={auc_f:.6f}, "
        f"global_weight={mix['global_weight']:.2f}"
    )
    print("=" * 72)

    return oof_global, oof_moe, final, rows, {
        "global_auc": auc_g,
        "moe_auc": auc_m,
        "blend_auc": auc_f,
        "global_weight": mix["global_weight"],
        "strict_loso": True,
    }


def main():
    # Replace the base implementation before invoking its normal run pipeline.
    v10.loso = strict_loso
    args = v10.parser().parse_args()
    v10.run(args)


if __name__ == "__main__":
    main()
