from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NotRequired, Optional, TypedDict, cast

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

HEADER_RE = re.compile(r"^&([A-Za-z0-9_]+)=(.*)$")
INOTE_KEY_RE = re.compile(r"^inote_(\d+)$")
LV_KEY_RE = re.compile(r"^lv_(\d+)$")
MEASURE_RE = re.compile(r"\{[^}]+\}")
BRACKET_RE = re.compile(r"\[[^\]]*\]")


@dataclass
class ChartSample:
    source_file: Path
    song_id: str
    song_title: str
    difficulty: int
    level_constant: float
    features: Dict[str, float]


class ModelBundle(TypedDict):
    model: RandomForestRegressor
    vectorizer: DictVectorizer
    metrics: Dict[str, float]
    metadata: NotRequired[Dict[str, Any]]


def safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def read_input_path(prompt: str, default: Optional[Path] = None) -> Path:
    if default is None:
        raw = input(f"{prompt}: ").strip()
    else:
        raw = input(f"{prompt}（默认: {default}）: ").strip()
    if not raw and default is not None:
        return default
    cleaned = raw.strip().strip('"').strip("'")
    return Path(cleaned)


def read_input_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt}（默认: {default}）: ").strip()
    if not raw:
        return default
    return int(raw)


def read_input_float(prompt: str, default: float) -> float:
    raw = input(f"{prompt}（默认: {default}）: ").strip()
    if not raw:
        return default
    return float(raw)


def parse_maidata_file(path: Path) -> Tuple[Dict[str, str], Dict[int, str]]:
    headers: Dict[str, str] = {}
    inotes: Dict[int, List[str]] = {}
    current_diff: Optional[int] = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n\r")
            match = HEADER_RE.match(line)
            if match:
                key, value = match.group(1), match.group(2)
                headers[key] = value

                inote_match = INOTE_KEY_RE.match(key)
                if inote_match:
                    current_diff = int(inote_match.group(1))
                    inotes[current_diff] = []
                else:
                    current_diff = None
                continue

            if current_diff is not None:
                inotes[current_diff].append(line)

    merged_inotes = {k: "\n".join(v).strip() for k, v in inotes.items() if "\n".join(v).strip()}
    return headers, merged_inotes


def tokenize_chart(chart_text: str) -> List[str]:
    tokens: List[str] = []
    for line in chart_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "E":
            continue
        cleaned = MEASURE_RE.sub("", stripped)
        cleaned = cleaned.strip()
        if not cleaned:
            continue
        if cleaned.startswith("(") and ")" in cleaned:
            cleaned = cleaned.split(")", 1)[1]
        for part in cleaned.split(","):
            token = part.strip()
            if token:
                tokens.append(token)
    return tokens


def count_touch_token(token: str) -> int:
    return 1 if token and token[0] in "ABCDE" else 0


def count_digits_outside_brackets(token: str) -> int:
    compact = BRACKET_RE.sub("", token)
    return len(re.findall(r"[1-8]", compact))


def extract_chart_features(chart_text: str, whole_bpm: Optional[float], cabinet: str) -> Dict[str, float]:
    tokens = tokenize_chart(chart_text)
    if not tokens:
        return {}

    feature = {
        "token_count": float(len(tokens)),
        "hold_count": 0.0,
        "slide_count": 0.0,
        "break_count": 0.0,
        "ex_count": 0.0,
        "touch_count": 0.0,
        "tap_like_count": 0.0,
        "digit_count": 0.0,
        "max_token_len": 0.0,
        "mean_token_len": 0.0,
        "std_token_len": 0.0,
        "bpm": float(whole_bpm) if whole_bpm is not None else 0.0,
        "is_dx": 1.0 if cabinet.upper() == "DX" else 0.0,
        "is_sd": 1.0 if cabinet.upper() == "SD" else 0.0,
    }

    lengths: List[int] = []
    for token in tokens:
        lengths.append(len(token))
        feature["hold_count"] += 1.0 if "h[" in token else 0.0
        feature["slide_count"] += 1.0 if ("-" in token or "<" in token or ">" in token) else 0.0
        feature["break_count"] += 1.0 if "b" in token else 0.0
        feature["ex_count"] += 1.0 if "x" in token else 0.0
        feature["touch_count"] += float(count_touch_token(token))

        digit_count = count_digits_outside_brackets(token)
        feature["digit_count"] += float(digit_count)
        if digit_count > 0 and "h[" not in token and ("-" not in token and "<" not in token and ">" not in token):
            feature["tap_like_count"] += 1.0

    lengths_np = np.array(lengths, dtype=float)
    feature["max_token_len"] = float(lengths_np.max())
    feature["mean_token_len"] = float(lengths_np.mean())
    feature["std_token_len"] = float(lengths_np.std())

    token_count = max(feature["token_count"], 1.0)
    feature["hold_ratio"] = feature["hold_count"] / token_count
    feature["slide_ratio"] = feature["slide_count"] / token_count
    feature["break_ratio"] = feature["break_count"] / token_count
    feature["touch_ratio"] = feature["touch_count"] / token_count
    feature["ex_ratio"] = feature["ex_count"] / token_count
    feature["digit_per_token"] = feature["digit_count"] / token_count

    return feature


def build_dataset(root: Path) -> List[ChartSample]:
    samples: List[ChartSample] = []
    for path in root.rglob("maidata.txt"):
        headers, inotes = parse_maidata_file(path)
        bpm = safe_float(headers.get("wholebpm", ""))
        cabinet = headers.get("cabinet", "")
        song_id = headers.get("shortid", "")
        song_title = headers.get("title", path.parent.name)

        for key, raw_lv in headers.items():
            lv_match = LV_KEY_RE.match(key)
            if not lv_match:
                continue
            diff = int(lv_match.group(1))
            lv_value = safe_float(raw_lv.strip()) if raw_lv is not None else None
            chart_text = inotes.get(diff)
            if lv_value is None or not chart_text:
                continue

            features = extract_chart_features(chart_text, bpm, cabinet)
            if not features:
                continue
            features["difficulty_index"] = float(diff)

            samples.append(
                ChartSample(
                    source_file=path,
                    song_id=song_id,
                    song_title=song_title,
                    difficulty=diff,
                    level_constant=lv_value,
                    features=features,
                )
            )
    return samples


def train_model(samples: List[ChartSample], test_size: float, random_state: int) -> ModelBundle:
    if len(samples) < 30:
        raise ValueError(f"样本太少，至少需要 30 条，当前 {len(samples)} 条")

    X_dict = [s.features for s in samples]
    y = np.array([s.level_constant for s in samples], dtype=float)

    idx = np.arange(len(samples))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state)

    train_dict = [X_dict[i] for i in train_idx]
    test_dict = [X_dict[i] for i in test_idx]

    vectorizer = DictVectorizer(sparse=False)
    X_train = np.asarray(vectorizer.fit_transform(train_dict), dtype=float)
    X_test = np.asarray(vectorizer.transform(test_dict), dtype=float)

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=1,
    )
    model.fit(X_train, y[train_idx])

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    metrics = {
        "train_mae": float(mean_absolute_error(y[train_idx], pred_train)),
        "test_mae": float(mean_absolute_error(y[test_idx], pred_test)),
        "train_rmse": float(math.sqrt(mean_squared_error(y[train_idx], pred_train))),
        "test_rmse": float(math.sqrt(mean_squared_error(y[test_idx], pred_test))),
        "train_r2": float(r2_score(y[train_idx], pred_train)),
        "test_r2": float(r2_score(y[test_idx], pred_test)),
        "n_samples": int(len(samples)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }

    return {
        "model": model,
        "vectorizer": vectorizer,
        "metrics": metrics,
    }


def save_bundle(path: Path, bundle: ModelBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(bundle, f)


def load_bundle(path: Path) -> ModelBundle:
    with path.open("rb") as f:
        obj = pickle.load(f)
    return cast(ModelBundle, obj)


def predict_one(model_bundle: ModelBundle, maidata_path: Path, difficulty: int) -> float:
    headers, inotes = parse_maidata_file(maidata_path)
    chart_text = inotes.get(difficulty)
    if not chart_text:
        raise ValueError(f"{maidata_path} 中不存在 inote_{difficulty} 或为空")

    bpm = safe_float(headers.get("wholebpm", ""))
    cabinet = headers.get("cabinet", "")

    features = extract_chart_features(chart_text, bpm, cabinet)
    if not features:
        raise ValueError("谱面特征提取失败，无法预测")
    features["difficulty_index"] = float(difficulty)

    vectorizer = model_bundle["vectorizer"]
    model = model_bundle["model"]
    transformed = cast(Any, vectorizer).transform([features])
    X = np.asarray(transformed, dtype=float)
    pred = model.predict(X)[0]
    return float(pred)


def cmd_train(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()

    samples = build_dataset(root)
    bundle = train_model(samples, test_size=args.test_size, random_state=args.random_state)

    bundle["metadata"] = {
        "root": str(root),
        "model_type": "RandomForestRegressor",
        "feature_count": len(bundle["vectorizer"].feature_names_),
    }
    save_bundle(output, bundle)

    metrics = bundle["metrics"]
    print("训练完成")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"模型已保存: {output}")


def cmd_predict(args: argparse.Namespace) -> None:
    model_path = Path(args.model).resolve()
    maidata_path = Path(args.maidata).resolve()

    bundle = load_bundle(model_path)
    pred = predict_one(bundle, maidata_path, args.difficulty)
    print(f"预测定数: {pred:.3f}")


def run_interactive_mode() -> None:
    base_dir = get_base_dir()
    default_model = base_dir / "maimai_level_model.pkl"

    print("=" * 48)
    print("maimai 谱面定数预测器（双击交互模式）")
    print("1) 训练模型")
    print("2) 预测单谱面定数")
    print("=" * 48)
    choice = input("请选择功能（1/2，默认2）: ").strip() or "2"

    try:
        if choice == "1":
            root = read_input_path("训练数据根目录", base_dir)
            output = read_input_path("模型输出路径", default_model)
            test_size = read_input_float("测试集比例", 0.2)
            random_state = read_input_int("随机种子", 42)

            args = argparse.Namespace(
                root=str(root),
                output=str(output),
                test_size=test_size,
                random_state=random_state,
            )
            cmd_train(args)
        else:
            model = read_input_path("模型路径", default_model)
            maidata = read_input_path("maidata.txt 路径")
            difficulty = read_input_int("难度编号", 5)

            args = argparse.Namespace(
                model=str(model),
                maidata=str(maidata),
                difficulty=difficulty,
            )
            cmd_predict(args)
    except Exception as exc:
        print(f"执行失败: {exc}")

    input("\n按回车键退出...")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="maimai 谱面定数预测器")
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="使用目录下 maidata.txt 训练模型")
    train_p.add_argument("--root", default=str(get_base_dir()), help="训练数据根目录")
    train_p.add_argument("--output", default=str(get_base_dir() / "maimai_level_model.pkl"), help="模型输出路径")
    train_p.add_argument("--test-size", type=float, default=0.2, help="测试集占比")
    train_p.add_argument("--random-state", type=int, default=42)
    train_p.set_defaults(func=cmd_train)

    pred_p = sub.add_parser("predict", help="预测单个谱面的定数")
    pred_p.add_argument("--model", default=str(get_base_dir() / "maimai_level_model.pkl"), help="模型文件路径")
    pred_p.add_argument("--maidata", required=True, help="待预测的 maidata.txt 路径")
    pred_p.add_argument("--difficulty", type=int, required=True, help="难度序号（2~5 常见）")
    pred_p.set_defaults(func=cmd_predict)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        run_interactive_mode()
        return
    args.func(args)


if __name__ == "__main__":
    main()
