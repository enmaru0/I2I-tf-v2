from pathlib import Path

from omegaconf import OmegaConf


def load_config_with_extends(config_path: str | Path):
    """``extends``を持つ差分configを再帰的に読み込む。"""
    return _load_config_with_extends(Path(config_path).resolve(), seen=set())


def _load_config_with_extends(config_path: Path, seen: set[Path]):
    if config_path in seen:
        chain = " -> ".join(map(str, [*seen, config_path]))
        raise ValueError(f"config extendsが循環しています: {chain}")
    if not config_path.exists():
        raise FileNotFoundError(f"configが見つかりません: {config_path}")

    seen = {*seen, config_path}
    config = OmegaConf.load(config_path)
    parent = config.get("extends", None)
    if parent is None:
        return config

    del config["extends"]
    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    parent_config = _load_config_with_extends(parent_path.resolve(), seen)
    return OmegaConf.merge(parent_config, config)
