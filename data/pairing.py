"""Source/target画像のファイル対応付け。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def _pair_matching_config(cfg):
    matching = cfg.data.get("pair_matching", {})
    return {
        "method": str(matching.get("method", "exact")).lower(),
        "delimiter": str(matching.get("delimiter", "_")),
        "token_index": int(matching.get("token_index", 0)),
        "case_sensitive": bool(matching.get("case_sensitive", True)),
    }


def _normalise_key(key: str, case_sensitive: bool) -> str:
    key = key.strip()
    if not key:
        raise ValueError("pair matching keyが空です")
    return key if case_sensitive else key.casefold()


def extract_pair_key(path: str | Path, cfg) -> str:
    """設定に従ってfilenameからpair keyを抽出する。"""
    path = Path(path)
    matching = _pair_matching_config(cfg)
    method = matching["method"]
    if method == "exact":
        return _normalise_key(path.name, matching["case_sensitive"])
    if method != "stem_token":
        raise ValueError(
            f"data.pair_matching.methodはexact/stem_tokenで指定してください: {method}"
        )

    delimiter = matching["delimiter"]
    if not delimiter:
        raise ValueError("stem_token matchingではdelimiterを空にできません")
    tokens = path.stem.split(delimiter)
    token_index = matching["token_index"]
    try:
        key = tokens[token_index]
    except IndexError as error:
        raise ValueError(
            f"pair token_index={token_index}がfilenameに存在しません: {path.name}"
        ) from error
    return _normalise_key(key, matching["case_sensitive"])


@lru_cache(maxsize=256)
def _target_index(
    target_dir: str, method: str, delimiter: str, token_index: int, case_sensitive: bool
):
    """同一directoryのtarget indexをprocess内でcacheする。"""
    directory = Path(target_dir)
    index: dict[str, list[Path]] = {}
    for candidate in sorted(directory.glob("*.hdr")):
        if ".mask" in candidate.name:
            continue
        if method == "exact":
            key = candidate.name
        else:
            tokens = candidate.stem.split(delimiter)
            if not (-len(tokens) <= token_index < len(tokens)):
                continue
            key = tokens[token_index]
        key = _normalise_key(key, case_sensitive)
        index.setdefault(key, []).append(candidate)
    return {key: tuple(paths) for key, paths in index.items()}


def resolve_target_hdr_path(src_hdr_path: str | Path, cfg) -> Path:
    """sourceのheader pathから対応するtarget header pathを解決する。

    ``paired``は従来のsuffix規則を使う。``paired_dir``では、デフォルトの完全
    同名、またはstemをdelimiter分割したtokenによる対応付けを利用できる。
    """
    src_hdr_path = Path(src_hdr_path)
    if cfg.data.mode != "paired_dir":
        target_suffix = cfg.data.target_suffix
        if not target_suffix.startswith("."):
            raise ValueError("target_suffix must start with .")
        return src_hdr_path.with_suffix(target_suffix + ".hdr")

    source_root = Path(cfg.data_dir)
    target_root = Path(cfg.data.target_data_dir)
    try:
        relative = src_hdr_path.relative_to(source_root)
    except ValueError as error:
        raise ValueError(
            f"sourceがdata_dir配下にありません: {src_hdr_path} / {source_root}"
        ) from error

    matching = _pair_matching_config(cfg)
    if matching["method"] == "exact":
        return target_root / relative
    if matching["method"] != "stem_token":
        # extract_pair_keyと同じvalidation errorを返す。
        extract_pair_key(src_hdr_path, cfg)

    key = extract_pair_key(src_hdr_path, cfg)
    target_dir = target_root / relative.parent
    index = _target_index(
        str(target_dir.resolve(strict=False)),
        matching["method"],
        matching["delimiter"],
        matching["token_index"],
        matching["case_sensitive"],
    )
    candidates = index.get(key, ())
    if not candidates:
        raise FileNotFoundError(
            "patient IDが一致するtargetが見つかりません: "
            f"key={key}, source={src_hdr_path}, target_dir={target_dir}"
        )
    if len(candidates) > 1:
        candidate_text = ", ".join(map(str, candidates))
        raise ValueError(
            "patient IDが一致するtargetが複数あり、pairを一意に決められません: "
            f"key={key}, source={src_hdr_path}, candidates=[{candidate_text}]"
        )
    return candidates[0]
