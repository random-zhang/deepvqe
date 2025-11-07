import os
import argparse
import json
import torch
import soundfile as sf

from train_aec import AECDataset, STFTConfig, spec_to_time
from stream.deepvqe_aec import DeepVQE_AEC


def build_argparser():
    p = argparse.ArgumentParser(description="Run DeepVQE-AEC inference on test split and save predictions")
    p.add_argument("--checkpoint", type=str, default="checkpoints/deepvqe_aec_epoch85.pt", help="Path to trained checkpoint (.pt)")
    p.add_argument("--manifest_csv", type=str, default="train.csv", help="CSV manifest with columns mix_filepath,farnerd_filepath,target_filepath,split")
    p.add_argument("--output_dir", type=str, default="predictions/test_epoch85", help="Directory to save predicted wav files and mapping CSV")
    p.add_argument("--segment_frames", type=int, default=63, help="Number of STFT frames used in dataset; keep default for var-length no-crop")
    p.add_argument("--cpu", action="store_true", help="Force CPU inference")
    p.add_argument("--no_crop", action=argparse.BooleanOptionalAction, default=True, help="Use full-length segments with right padding for inference")
    p.add_argument("--sample_rate", type=int, default=16000, help="Output wav sample rate")
    p.add_argument("--limit", type=int, default=10, help="Limit number of test items to predict")
    return p


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Load checkpoint (ensure PyTorch 2.6+ compatibility)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})

    # Model/STFT params: prefer values saved in checkpoint
    align_delay = int(saved_args.get("align_delay", 200))
    n_fft = int(saved_args.get("n_fft", 512))
    hop_length = int(saved_args.get("hop_length", 256))
    win_length = int(saved_args.get("win_length", 512))

    stft_cfg = STFTConfig(n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    # Build and load model
    model = DeepVQE_AEC(align_hidden=2, align_delay=align_delay).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Dataset over test split; use var-length handling via no_crop
    ds = AECDataset(
        args.manifest_csv,
        split="test",
        segment_frames=args.segment_frames,
        stft_cfg=stft_cfg,
        device="cpu",
        no_crop=args.no_crop,
    )

    if not hasattr(ds, "items") or len(ds.items) == 0:
        print("[WARN] Manifest has no split=test items. Nothing to predict.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    mapping_rows = []

    with torch.no_grad():
        total = min(len(ds.items), max(0, int(args.limit))) if args.limit is not None else len(ds.items)
        for idx in range(total):
            X_mic, X_far, X_clean, valid_T = ds[idx]
            # Add batch dimension
            X_mic = X_mic.unsqueeze(0).to(device)
            X_far = X_far.unsqueeze(0).to(device)

            # Forward
            Y = model(X_mic, X_far)  # (B,F,T,2)
            B, F, T, C = Y.shape
            # Build valid frame mask for padded part
            t_idx = torch.arange(T, device=Y.device).unsqueeze(0)
            valid_Ts = torch.tensor([int(valid_T)], device=Y.device)
            valid_mask_bt = (t_idx < valid_Ts.unsqueeze(1)).unsqueeze(1).unsqueeze(-1).float()  # (B,1,T,1)
            Y_masked = Y * valid_mask_bt

            # Convert to time domain
            y_pred_time = spec_to_time(Y_masked, n_fft=n_fft, hop_length=hop_length, win_length=win_length)  # (B, samples)
            y = y_pred_time[0].detach().cpu().numpy()

            # Crop to valid samples (frames -> samples)
            valid_samples = int(valid_T) * hop_length
            y = y[:valid_samples]

            # Derive output filename using mix filepath basename
            mix_fp, far_fp, tgt_fp = ds.items[idx]
            base = os.path.splitext(os.path.basename(mix_fp))[0]
            out_wav = os.path.join(args.output_dir, f"{base}_pred.wav")
            sf.write(out_wav, y, args.sample_rate)

            # Also place corresponding source wavs into same folder for inspection
            # Use standardized names to avoid conflicts
            import shutil
            mic_out = os.path.join(args.output_dir, f"{base}_mic.wav")
            far_out = os.path.join(args.output_dir, f"{base}_far.wav")
            clean_out = os.path.join(args.output_dir, f"{base}_clean.wav")
            try:
                if os.path.isfile(mix_fp):
                    shutil.copy2(mix_fp, mic_out)
                if os.path.isfile(far_fp):
                    shutil.copy2(far_fp, far_out)
                if os.path.isfile(tgt_fp):
                    shutil.copy2(tgt_fp, clean_out)
            except Exception as e:
                print(f"[WARN] Copy sources failed for {base}: {e}")

            # Record mapping
            mapping_rows.append({
                "mix_filepath": mix_fp,
                "farnerd_filepath": far_fp,
                "target_filepath": tgt_fp,
                "pred_filepath": out_wav,
                "mix_out": mic_out,
                "far_out": far_out,
                "clean_out": clean_out,
            })

    # Save mapping CSV
    # Save mapping CSV for easier inspection
    mapping_csv = os.path.join(args.output_dir, "predictions_mapping.csv")
    try:
        import csv
        with open(mapping_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["mix_filepath","farnerd_filepath","target_filepath","pred_filepath","mix_out","far_out","clean_out"])
            w.writeheader()
            for r in mapping_rows:
                w.writerow(r)
        print(f"Saved {len(mapping_rows)} predictions to {args.output_dir}\nMapping CSV: {mapping_csv}")
    except Exception as e:
        # Fallback to JSON if CSV fails
        mapping_json = os.path.join(args.output_dir, "predictions_mapping.json")
        with open(mapping_json, "w", encoding="utf-8") as f:
            json.dump(mapping_rows, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(mapping_rows)} predictions to {args.output_dir}\nMapping JSON: {mapping_json} (CSV failed: {e})")


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
