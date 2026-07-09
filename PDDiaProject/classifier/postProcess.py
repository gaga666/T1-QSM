import os
import ast
import csv
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve
)


def build_transition_matrix(n_pd=5, n_hc=5):
    A_pd_pd = 1.0 - 1.0 / n_pd
    A_pd_hc = 1.0 - A_pd_pd

    A_hc_hc = 1.0 - 1.0 / n_hc
    A_hc_pd = 1.0 - A_hc_hc

    A = np.array([
        [A_hc_hc, A_hc_pd],
        [A_pd_hc, A_pd_pd]
    ])

    return A


def viterbi(pd_probs, n_pd=5, n_hc=5):
    pd_probs = np.asarray(pd_probs, dtype=float)

    eps = 1e-8
    pd_probs = np.clip(pd_probs, eps, 1.0 - eps)

    T = pd_probs.shape[0]
    n_state = 2

    A = build_transition_matrix(n_pd=n_pd, n_hc=n_hc)
    logA = np.log(A + eps)

    pi = np.array([0.99, 0.01])
    logpi = np.log(pi + eps)

    log_emit_hc = np.log(1.0 - pd_probs + eps)
    log_emit_pd = np.log(pd_probs + eps)

    dp = np.zeros((T, n_state))
    backptr = np.zeros((T, n_state), dtype=int)

    dp[0, 0] = logpi[0] + log_emit_hc[0]
    dp[0, 1] = logpi[1] + log_emit_pd[0]

    for t in range(1, T):
        h_h = dp[t - 1, 0] + logA[0, 0]
        p_h = dp[t - 1, 1] + logA[1, 0]

        if h_h >= p_h:
            dp[t, 0] = h_h + log_emit_hc[t]
            backptr[t, 0] = 0
        else:
            dp[t, 0] = p_h + log_emit_hc[t]
            backptr[t, 0] = 1

        h_p = dp[t - 1, 0] + logA[0, 1]
        p_p = dp[t - 1, 1] + logA[1, 1]

        if h_p >= p_p:
            dp[t, 1] = h_p + log_emit_pd[t]
            backptr[t, 1] = 0
        else:
            dp[t, 1] = p_p + log_emit_pd[t]
            backptr[t, 1] = 1

    states = np.zeros(T, dtype=int)

    # 修正：这里要看最后一个时间点dp最大的是HC还是PD
    states[T - 1] = np.argmax(dp[T - 1])

    for t in range(T - 2, -1, -1):
        states[t] = backptr[t + 1, states[t + 1]]

    max_run = 0
    cur_run = 0

    for s in states:
        if s == 1:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0

    is_patient_pd = int(max_run >= n_pd)

    # 用于ROC的连续分数
    # 越大说明越像PD
    patient_score = max_run / T

    return states, is_patient_pd, patient_score, max_run


def parse_slice_probs(x):
    if isinstance(x, str):
        return np.array(ast.literal_eval(x), dtype=float)
    else:
        return np.array(x, dtype=float)


def compute_metrics(labels, scores, preds):
    auc = roc_auc_score(labels, scores)
    acc = accuracy_score(labels, preds)

    sen = recall_score(
        labels,
        preds,
        zero_division=0
    )

    pre = precision_score(
        labels,
        preds,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        preds,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        labels,
        preds,
        labels=[0, 1]
    ).ravel()

    spe = tn / (tn + fp + 1e-8)

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores
    )

    return {
        "auc": auc,
        "acc": acc,
        "sen": sen,
        "spe": spe,
        "pre": pre,
        "f1": f1,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds
    }


def save_roc_data(save_path, fpr, tpr, thresholds):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "fpr",
            "tpr",
            "threshold"
        ])

        for a, b, c in zip(fpr, tpr, thresholds):
            writer.writerow([
                float(a),
                float(b),
                float(c)
            ])


def process_one_fold(csv_path, save_dir, fold, n_pd=5, n_hc=5):
    df = pd.read_csv(csv_path)

    if "slice_probs" not in df.columns:
        raise ValueError(
            f"{csv_path} 中没有 slice_probs 列，"
            f"无法进行维特比重规划。需要保存每个病人的128个slice概率。"
        )

    names = []
    labels = []
    scores = []
    preds = []
    max_runs = []
    states_all = []

    for _, row in df.iterrows():
        name = row["patient_name"]
        label = int(row["true_label"])

        slice_probs = parse_slice_probs(row["slice_probs"])

        states, pred, score, max_run = viterbi(
            slice_probs,
            n_pd=n_pd,
            n_hc=n_hc
        )

        names.append(name)
        labels.append(label)
        scores.append(score)
        preds.append(pred)
        max_runs.append(max_run)
        states_all.append(states.tolist())

    result = compute_metrics(
        labels=labels,
        scores=scores,
        preds=preds
    )

    # 保存重规划后的病人结果
    result_path = os.path.join(
        save_dir,
        f"fold_{fold}_test_viterbi_results.csv"
    )

    with open(result_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "patient_name",
            "true_label",
            "viterbi_score",
            "viterbi_pred_label",
            "max_pd_run",
            "viterbi_states"
        ])

        for n, y, s, p, r, states in zip(
            names,
            labels,
            scores,
            preds,
            max_runs,
            states_all
        ):
            writer.writerow([
                n,
                int(y),
                float(s),
                int(p),
                int(r),
                states
            ])

    # 保存ROC曲线数据
    roc_path = os.path.join(
        save_dir,
        f"fold_{fold}_test_viterbi_roc.csv"
    )

    save_roc_data(
        roc_path,
        result["fpr"],
        result["tpr"],
        result["thresholds"]
    )

    return result


def main():
    input_dir = "./result/classifier_cv"
    save_dir = "./result/viterbi_replan"

    os.makedirs(save_dir, exist_ok=True)

    n_folds = 5
    n_pd = 5
    n_hc = 5

    all_metrics = []

    for fold in range(1, n_folds + 1):
        csv_path = os.path.join(
            input_dir,
            f"fold_{fold}_test_results.csv"
        )

        print(f"\n处理 Fold {fold}: {csv_path}")

        result = process_one_fold(
            csv_path=csv_path,
            save_dir=save_dir,
            fold=fold,
            n_pd=n_pd,
            n_hc=n_hc
        )

        all_metrics.append([
            fold,
            result["auc"],
            result["acc"],
            result["sen"],
            result["spe"],
            result["pre"],
            result["f1"]
        ])

        print(
            f"Fold {fold} Viterbi: "
            f"AUC={result['auc']:.4f}, "
            f"ACC={result['acc']:.4f}, "
            f"SEN={result['sen']:.4f}, "
            f"SPE={result['spe']:.4f}, "
            f"PRE={result['pre']:.4f}, "
            f"F1={result['f1']:.4f}"
        )

    metrics_path = os.path.join(
        save_dir,
        "five_fold_viterbi_metrics.csv"
    )

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "fold",
            "auc",
            "acc",
            "sen",
            "spe",
            "pre",
            "f1"
        ])

        writer.writerows(all_metrics)

    metrics_np = np.array(all_metrics)

    print("\n========== 维特比重规划五折结果 ==========")
    print(f"平均 AUC: {metrics_np[:, 1].mean():.4f}")
    print(f"平均 ACC: {metrics_np[:, 2].mean():.4f}")
    print(f"平均 SEN: {metrics_np[:, 3].mean():.4f}")
    print(f"平均 SPE: {metrics_np[:, 4].mean():.4f}")
    print(f"平均 PRE: {metrics_np[:, 5].mean():.4f}")
    print(f"平均 F1 : {metrics_np[:, 6].mean():.4f}")


if __name__ == "__main__":
    main()