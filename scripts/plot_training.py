"""Publication figures from saved metrics, retaining the UF restart boundary."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def smoothed(values, window=20):
    return [sum(values[max(0, i-window+1):i+1]) / min(window, i+1) for i in range(len(values))]


def main():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    destination = ROOT / "submission/27100384/plots"
    destination.mkdir(parents=True, exist_ok=True)
    for track, title in (("sft", "Supervised fine-tuning"), ("dpo_hh", "HH-RLHF DPO"), ("dpo_uf", "UltraFeedback DPO"), ("persona", "Persona LoRA-DPO")):
        records = rows(ROOT / f"submission/27100384/evidence/training/{track}/metrics.jsonl")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.subplots_adjust(left=.11, right=.86, bottom=.31, top=.89)
        if track == "dpo_uf":
            old = rows(ROOT / "outputs/downloads/main/pa2/outputs/dpo_uf/metrics.jsonl")
            discarded = [r for r in old if r.get("split") == "train" and r["step"] > 300]
            ax.plot([r["step"] for r in discarded], smoothed([r["train_loss"] for r in discarded]), color=".65", ls="--", label="Discarded original segment")
            records = [r for r in old if r.get("step", 999) <= 300] + records
            ax.axvline(300, color="#bd6a16", ls=":", label="Weight-only recovery; Adam reset")
        train = [r for r in records if r.get("split") == "train"]
        val = [r for r in records if r.get("split") == "validation"]
        loss = "loss" if track == "persona" else "train_loss"
        vloss = "eval_loss" if track == "persona" else "loss"
        train = [r for r in train if loss in r]
        ax.plot([r["step"] for r in train], smoothed([r[loss] for r in train]), color="#246a9c", label="Train loss (20-log moving mean)")
        ax.plot([r["step"] for r in val], [r[vloss] for r in val], "o-", color="#b44b40", label="Validation loss (raw)")
        ax.set(title=title, xlabel="Attempted update step", ylabel="Response-token NLL" if track == "sft" else "DPO loss")
        ax.grid(alpha=.2)
        if track != "sft":
            axis = ax.twinx()
            key = "eval_rewards/accuracies" if track == "persona" else "reward_accuracy"
            axis.plot([r["step"] for r in val], [r[key] for r in val], "s--", color="#2d8654", label="Validation reward accuracy")
            axis.set_ylabel("Reward accuracy")
            axis.set_ylim(0, 1.04)
            handles, labels = ax.get_legend_handles_labels()
            h, l = axis.get_legend_handles_labels()
            fig.legend(handles+h, labels+l, fontsize=7, loc="lower center", ncol=2, frameon=False)
        else:
            fig.legend(*ax.get_legend_handles_labels(), fontsize=7, loc="lower center", ncol=2, frameon=False)
        fig.savefig(destination / f"{track}.png", dpi=240)
        plt.close(fig)
    print(str(destination))


if __name__ == "__main__":
    main()
