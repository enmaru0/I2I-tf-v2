def get_optimizer_iterations(optimizer):
    """LossScaleOptimizerを考慮して実際の更新step変数を返す。"""
    seen = set()
    while hasattr(optimizer, "inner_optimizer") and id(optimizer) not in seen:
        seen.add(id(optimizer))
        optimizer = optimizer.inner_optimizer
    return optimizer.iterations
