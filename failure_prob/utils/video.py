import logging
import math
import os
import signal

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import wandb

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from failure_prob.conf import Config
from failure_prob.data.utils import Rollout
from failure_prob.model.base import BaseModel
from concurrent.futures import ProcessPoolExecutor, as_completed

from failure_prob.utils.metrics import eval_functional_conformal

from .vis import compute_mean_std
from .routines import model_forward_dataloader


def calc_mean_fail_step_with_cp(
    cfg,
    model, 
    rollouts_by_split_name: dict,
    dataloader_by_split_name: dict,
    alpha: float = 0.2,
    calib_split_names=["val_seen"], 
    test_split_names=["val_unseen"],
):
    """
    Calculate mean fail step statistics using conformal prediction.
    
    Args:
        cfg: Configuration object
        model: Trained model
        rollouts_by_split_name: Dictionary of rollouts by split name
        dataloader_by_split_name: Dictionary of dataloaders by split name
        alpha: Conformal prediction alpha level
        calib_split_names: List of calibration split names
        test_split_names: List of test split names to evaluate
    
    Returns:
        Dictionary containing statistics for each split
    """
    #### Forward the model and compute the scores ####
    scores_by_split_name = {}
    for split, dataloader in dataloader_by_split_name.items():
        # Re-create a dataset to disable shuffling.
        dataloader = DataLoader(
            dataloader.dataset, 
            batch_size=cfg.model.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        with torch.no_grad():
            scores, valid_masks, _ = model_forward_dataloader(model, dataloader)
        scores = scores.detach().cpu().numpy()
        seq_lengths = valid_masks.sum(dim=-1).cpu().numpy()  # (B,)
        scores_by_split_name[split] = [
            scores[i, :int(seq_lengths[i])] for i in range(len(seq_lengths))
        ]
        
    df, cp_bands_by_alpha = eval_functional_conformal(
        rollouts_by_split_name, scores_by_split_name, "model",
        calib_split_names=calib_split_names, test_split_names=test_split_names
    )
    
    # Retrieve the CP band for the given alpha.
    cp_band = cp_bands_by_alpha[alpha][0]  # Shape: (T,)
    
    # Calculate and print mean fail step statistics for test rollouts
    print("\n" + "=" * 60)
    print("MEAN FAIL STEP STATISTICS (with conformal prediction)")
    print("=" * 60)
    
    all_stats = {}
    
    for split_name in test_split_names:
        if split_name != "val_unseen":
            continue
        rollouts = rollouts_by_split_name[split_name]
        split_scores = scores_by_split_name[split_name]
        
        # Calculate sequence lengths
        seq_lengths = [len(s) for s in split_scores]
        success_lengths = [len(split_scores[i]) for i, r in enumerate(rollouts) if r.episode_success == 1]
        fail_lengths = [len(split_scores[i]) for i, r in enumerate(rollouts) if r.episode_success == 0]
        
        # Calculate first CP violation step (predicted fail step)
        cp_violation_steps = []
        for i, (r, scores) in enumerate(zip(rollouts, split_scores)):
            violation_mask = scores > cp_band[:len(scores)]
            if violation_mask.any():
                first_violation = np.argmax(violation_mask)  # First True index
                cp_violation_steps.append(first_violation)
            else:
                cp_violation_steps.append(len(scores))  # No violation, use full length
        
        # Split by actual success/failure
        cp_violation_fail = [cp_violation_steps[i] for i, r in enumerate(rollouts) if r.episode_success == 0]
        cp_violation_success = [cp_violation_steps[i] for i, r in enumerate(rollouts) if r.episode_success == 1]
        
        # Calculate detection rate (how many failures were detected before end)
        failures_detected = sum(1 for i, r in enumerate(rollouts) if r.episode_success == 0 and cp_violation_steps[i] < len(split_scores[i]))
        false_alarms = sum(1 for i, r in enumerate(rollouts) if r.episode_success == 1 and cp_violation_steps[i] < len(split_scores[i]))
        
        print(f"\n{split_name}:")
        print(f"  Total rollouts: {len(rollouts)}")
        print(f"  Success: {len(success_lengths)}, Failure: {len(fail_lengths)}")
        print(f"  Mean seq length (all): {np.mean(seq_lengths):.2f} ± {np.std(seq_lengths):.2f}")
        if fail_lengths:
            print(f"  Mean seq length (failures): {np.mean(fail_lengths):.2f} ± {np.std(fail_lengths):.2f}")
        if success_lengths:
            print(f"  Mean seq length (successes): {np.mean(success_lengths):.2f} ± {np.std(success_lengths):.2f}")
        
        print(f"  --- Conformal Prediction (alpha={alpha}) ---")
        print(f"  Mean CP violation step (all): {np.mean(cp_violation_steps):.2f} ± {np.std(cp_violation_steps):.2f}")
        if cp_violation_fail:
            print(f"  Mean CP violation step (failures): {np.mean(cp_violation_fail):.2f} ± {np.std(cp_violation_fail):.2f}")
        if cp_violation_success:
            print(f"  Mean CP violation step (successes): {np.mean(cp_violation_success):.2f} ± {np.std(cp_violation_success):.2f}")
        
        if fail_lengths:
            print(f"  Failure detection rate: {failures_detected}/{len(fail_lengths)} ({100*failures_detected/len(fail_lengths):.1f}%)")
        if success_lengths:
            print(f"  False alarm rate: {false_alarms}/{len(success_lengths)} ({100*false_alarms/len(success_lengths):.1f}%)")
        
        # Plot histogram of early stopping times
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: All rollouts
        ax = axes[0]
        ax.hist(cp_violation_steps, bins=range(0, max(cp_violation_steps) + 2), edgecolor='black', alpha=0.7)
        ax.set_xlabel('Early stopping time (steps)')
        ax.set_ylabel('Number of rollouts')
        ax.set_title(f'All rollouts (n={len(cp_violation_steps)})')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Failures only
        if cp_violation_fail:
            ax = axes[1]
            ax.hist(cp_violation_fail, bins=range(0, max(cp_violation_fail) + 2), edgecolor='black', alpha=0.7, color='red')
            ax.set_xlabel('Early stopping time (steps)')
            ax.set_ylabel('Number of rollouts')
            ax.set_title(f'Failed rollouts (n={len(cp_violation_fail)})')
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Successes only
        if cp_violation_success:
            ax = axes[2]
            ax.hist(cp_violation_success, bins=range(0, max(cp_violation_success) + 2), edgecolor='black', alpha=0.7, color='green')
            ax.set_xlabel('Early stopping time (steps)')
            ax.set_ylabel('Number of rollouts')
            ax.set_title(f'Successful rollouts (n={len(cp_violation_success)})')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Log to wandb
        wandb.log({
            f'early_stopping_histogram_{split_name}_alpha{alpha}': wandb.Image(fig)
        })
        
        # Optionally save to disk
        # plt.savefig(f'early_stopping_histogram_{split_name}_alpha{alpha}.png', dpi=150, bbox_inches='tight')
        # plt.savefig(f'early_stopping_histogram_{split_name}_alpha{alpha}.pdf', bbox_inches='tight')
        print(f"  Saved histogram to early_stopping_histogram_{split_name}_alpha{alpha}.png/pdf")
        plt.close()
        
        # Store statistics
        all_stats = {
            'violation_stats/total_rollouts': len(rollouts),
            'violation_stats/n_success': len(success_lengths),
            'violation_stats/n_fail': len(fail_lengths),
            'violation_stats/mean_seq_len': np.mean(seq_lengths),
            'violation_stats/std_seq_len': np.std(seq_lengths),
            'violation_stats/mean_seq_len_fail': np.mean(fail_lengths) if fail_lengths else None,
            'violation_stats/mean_seq_len_success': np.mean(success_lengths) if success_lengths else None,
            'violation_stats/mean_cp_violation_step': np.mean(cp_violation_steps),
            'violation_stats/std_cp_violation_step': np.std(cp_violation_steps),
            'violation_stats/mean_cp_violation_fail': np.mean(cp_violation_fail) if cp_violation_fail else None,
            'violation_stats/mean_cp_violation_success': np.mean(cp_violation_success) if cp_violation_success else None,
            'violation_stats/failures_detected': failures_detected,
            'violation_stats/false_alarms': false_alarms,
            'violation_stats/detection_rate': failures_detected / len(fail_lengths) if fail_lengths else None,
            'violation_stats/false_alarm_rate': false_alarms / len(success_lengths) if success_lengths else None,
        }
    
    print("=" * 60 + "\n")
    
    return all_stats, df, cp_bands_by_alpha


def read_frames_and_frame_rate(video_path):
    """
    Reads an MP4 video file and returns its frames and frame rate.

    Parameters:
        video_path (str): The path to the MP4 video file.

    Returns:
        frames (list): A list of frames (each frame is a NumPy array), in BGR color format.
        frame_rate (float): The frame rate of the video (frames per second).
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    # Get the frame rate (frames per second)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:  # No more frames to read
            break
        frames.append(frame)
    
    cap.release()
    return frames, frame_rate


# Helper function that processes a single rollout.
def eval_save_video_single(args):
    """
    Process a single rollout:
      - Reads the video frames.
      - Plots the corresponding score progression for each frame.
      - Saves the annotated video.
    """
    (
        i,
        r,
        scores,
        task_mean_std,
        score_range,
        cfg,
        save_folder,
    ) = args

    # Read the raw frames and get the frame rate.
    raw_frames_bgr, fps = read_frames_and_frame_rate(r.mp4_path)
    
    mean_success, std_success, mean_fail, std_fail = task_mean_std
    
    if r.exec_horizon:
        exec_horizon = r.exec_horizon
    else:
        exec_horizon = cfg.dataset.exec_horizon
        
    # Somehow openpi on DROID tends to save one more frame
    if math.ceil( (len(raw_frames_bgr)-1) /float(exec_horizon)) == len(scores):
        raw_frames_bgr = raw_frames_bgr[:-1]

    # Trim scores to the correct sequence length.
    assert math.ceil(len(raw_frames_bgr)/float(exec_horizon)) == len(scores), (
        f"Mismatch: len(raw_frames_bgr)={len(raw_frames_bgr)}, len(scores)={len(scores)}, exec_horizon={exec_horizon}"
    )

    frames = []
    # For each frame in the raw video, create a figure that shows:
    # - The current RGB frame.
    # - A plot of the predicted score up to that time step.
    for j in range(len(raw_frames_bgr)):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=120)

        # Left subplot: RGB frame
        ax = axes[0]
        ax.imshow(raw_frames_bgr[j][:, :, ::-1])
        ax.axis("off")
        ax.set_title(f"RGB obs frame {j}")

        # Right subplot: score progression
        ax = axes[1]
        # Calculate how many scores to plot: one per exec_horizon steps.
        score_plot_end = j // exec_horizon + 1

        ax.plot(mean_success, color="green", label="Task success", alpha=0.7)
        ax.fill_between(range(len(mean_success)), mean_success - std_success, mean_success + std_success, color="green", alpha=0.2)
        ax.plot(mean_fail, color="red", label="Task failure", alpha=0.7)
        ax.fill_between(range(len(mean_fail)), mean_fail - std_fail, mean_fail + std_fail, color="red", alpha=0.2)
        
        ax.plot(scores[:score_plot_end], label="Current rollout", color="blue", lw=2)

        # ax.set_ylim(score_range)
        ax.set_xlim(0, len(scores))
        ax.set_title("Predicted failure score")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Score")
        
        # draw a vertical line
        ax.axvline(r.task_min_step, color='black', linestyle='--', label='earliest termination')
        
        # ax.legend()

        # Figure title with rollout information.
        fig.suptitle(
            f"{r.task_description}\nEp {r.episode_idx}, Succ {r.episode_success} Final score {scores[-1]:.2f}"
        )

        fig.tight_layout()
        # Draw the canvas and convert the figure to a numpy array.
        fig.canvas.draw()
        plot_img = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        frames.append(plot_img)

    # Save the list of frames as an mp4 video.
    # Include rollout index (i) to ensure unique filenames even if task/episode/success are the same
    save_path = os.path.join(
        save_folder,
        f"rollout{i}_task{r.task_id}_ep{r.episode_idx}_succ{r.episode_success}_finalscore{scores[-1]:.2f}.mp4",
    )
    imageio.mimsave(save_path, frames, fps=fps)


def eval_save_videos(
    dataloader: DataLoader, 
    model: BaseModel, 
    cfg: Config, 
    save_folder: str
):
    """
    Evaluate the model on the given rollouts and save annotated videos in parallel.
    """
    device = model.get_device()

    # Forward the model and get scores for all rollouts
    dataloader = DataLoader(dataloader.dataset, batch_size=cfg.model.batch_size, shuffle=False, num_workers=0)
    rollouts = dataloader.dataset.get_rollouts()
    with torch.no_grad():
        scores, valid_masks, labels = model_forward_dataloader(model, dataloader)
    rollouts_scores = (scores * valid_masks).detach().cpu().numpy()
    seq_lengths = valid_masks.sum(dim=-1).detach().cpu().numpy()

    # Determine the score range based on the model type.
    if cfg.model.name == "indep":
        score_range = (rollouts_scores.min(), rollouts_scores.max())
    elif cfg.model.name == "lstm":
        score_range = (0, 1)
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")
    
    # Convert the scores to a list of numpy arrays.
    rollouts_scores = [rollouts_scores[i][:int(seq_lengths[i])] for i in range(len(rollouts))]
    
    # Compute for each task, the mean/std of the scores
    task_ids = list(set([r.task_id for r in rollouts]))
    mean_std_by_task = {}
    for task_id in task_ids:
        task_success_scores = [rollouts_scores[i] for i in range(len(rollouts)) if rollouts[i].task_id == task_id and rollouts[i].episode_success == 1]
        task_fail_scores = [rollouts_scores[i] for i in range(len(rollouts)) if rollouts[i].task_id == task_id and rollouts[i].episode_success == 0]
        mean_success, std_success = compute_mean_std(task_success_scores)
        mean_fail, std_fail = compute_mean_std(task_fail_scores)
        mean_std_by_task[task_id] = (mean_success, std_success, mean_fail, std_fail)

    # Prepare a list of tasks. Instead of passing the entire rollout object,
    # extract only the necessary attributes.
    tasks = []
    for i, r in enumerate(rollouts):
        simple_rollout = r.get_simple_meta()
        tasks.append((
            i,
            simple_rollout,
            rollouts_scores[i],
            mean_std_by_task[r.task_id],
            score_range,
            cfg,
            save_folder
        ))
        
    max_workers = 8
    if cfg.train.debug: 
        max_workers = 1

    if cfg.train.eval_save_video_multiproc:
        # Use ProcessPoolExecutor to parallelize the processing over rollouts.
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(eval_save_video_single, task) for task in tasks]
            # Optionally, wait for all tasks to finish and handle exceptions.
            for future in tqdm(as_completed(futures), total=len(futures),
                            desc="Processing rollouts"):
                try:
                    future.result()
                except Exception as e:
                    print("Error processing rollout:", e)
    else:
        for task in tqdm(tasks, total=len(tasks), desc="Processing rollouts"):
            eval_save_video_single(task)
                

def plot_band_with_scores(
    ax: plt.Axes,
    scores: np.ndarray,
    exec_horizon: int, 
    cp_band: np.ndarray,
    ymin: float,
    ymax: float,
    xmax: float = None,
):
    x = np.arange(len(cp_band)) * exec_horizon
    # ax.plot(x, cp_band, label=f"CP band", color="green")
    ax.fill_between(x, np.zeros_like(cp_band), cp_band, color="green", alpha=0.2)

    x = np.arange(len(scores)) * exec_horizon
    ax.plot(
        x, scores, 
        label="Current rollout", 
        color="blue", 
        # marker="o",
        # markersize=1.5,
        lw=1.5,
    )

    if xmax is not None:
        ax.set_xlim(0, xmax * exec_horizon)
    else:
        ax.set_xlim(0, len(scores) * exec_horizon)
    ax.set_ylim(ymin, ymax)
    
    # ax.set_title("Predicted failure score")
    ax.set_xlabel("Time step $t$")
    ax.set_ylabel("Score $s_t$")
    # ax.legend()
    return ax


def process_single_rollout_functional(r, scores, cp_band, cfg, alpha, vmin, vmax, save_folder):
    """
    Process a single rollout: load raw frames, generate plots for each frame,
    and save the resulting video.
    """
    # Read the raw frames and frame rate.
    raw_frames_bgr, fps = read_frames_and_frame_rate(r.mp4_path)
    
    # Determine execution horizon.
    exec_horizon = r.exec_horizon if r.exec_horizon else cfg.dataset.exec_horizon
    
    # Somehow openpi on DROID tends to save one more frame
    if math.ceil( (len(raw_frames_bgr)-1) /float(exec_horizon)) == len(scores):
        raw_frames_bgr = raw_frames_bgr[:-1]

    # Trim scores to the correct sequence length.
    assert math.ceil(len(raw_frames_bgr)/float(exec_horizon)) == len(scores), (
        f"Mismatch: len(raw_frames_bgr)={len(raw_frames_bgr)}, len(scores)={len(scores)}, exec_horizon={exec_horizon}"
    )
    
    # Create a save folder for the video.
    episode_pred_fail = (scores[:len(scores)] > cp_band[:len(scores)]).any()
    save_name = f"task{r.task_id}_ep{r.episode_idx}_succ{r.episode_success}_predsucc{1-int(episode_pred_fail)}"
    frame_save_folder = os.path.join(save_folder, save_name)
    os.makedirs(frame_save_folder, exist_ok=True)
    
    # Plot a separate scores and save as pdf
    fig, ax = plt.subplots(1, 1, figsize=(4, 1.5), dpi=300)
    plot_band_with_scores(
        ax=ax,
        scores=scores,
        exec_horizon=exec_horizon,
        cp_band=cp_band,
        ymin=vmin,
        ymax=vmax,
        xmax=len(scores)
    )
    plot_save_path = os.path.join(save_folder, f"{save_name}-wide.pdf")
    fig.savefig(plot_save_path, bbox_inches="tight")
    plot_save_path = os.path.join(save_folder, f"{save_name}-wide.png")
    fig.savefig(plot_save_path, bbox_inches="tight")
    plt.close(fig)
    
    ## Re-plot a 1x1 aspect ratio figure for the score
    # Plot a separate scores and save as pdf
    fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.5), dpi=300)
    plot_band_with_scores(
        ax=ax,
        scores=scores,
        exec_horizon=exec_horizon,
        cp_band=cp_band,
        ymin=vmin,
        ymax=vmax,
        xmax=len(scores)
    )
    plot_save_path = os.path.join(save_folder, f"{save_name}.pdf")
    fig.savefig(plot_save_path, bbox_inches="tight")
    plot_save_path = os.path.join(save_folder, f"{save_name}.png")
    fig.savefig(plot_save_path, bbox_inches="tight")
    plt.close(fig)

    
    frames = []
    has_failed = False
    
    # Create a plot for each frame.
    for j in range(len(raw_frames_bgr)):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=120)
        
        # Determine how many scores to plot: one per exec_horizon steps.
        score_plot_end = j // exec_horizon + 1
        
        # Set the failure flag if the current score exceeds the CP band.
        if scores[score_plot_end - 1] > cp_band[score_plot_end - 1]:
            has_failed = True
        
        # Left subplot: Display the RGB frame with a colored border.
        ax = axes[0]
        # Convert BGR to RGB.
        obs = raw_frames_bgr[j][:, :, ::-1]

        # Save the RGB frame inference happens at the current time step. 
        if j % exec_horizon == 0:
            frame_save_path = os.path.join(frame_save_folder, f"frame_{j}.jpg")
            imageio.imwrite(frame_save_path, obs)
        
        if has_failed: # Green border if has failed.
            obs = cv2.copyMakeBorder(obs, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))
        else: # Red border if has not failed. 
            obs = cv2.copyMakeBorder(obs, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 255, 0))
        ax.imshow(obs)
        ax.axis("off")
        ax.set_title(f"RGB image")
        
        # Right subplot: Plot the CP band and current rollout scores.
        ax = axes[1]
        
        plot_band_with_scores(
            ax = ax,
            scores = scores[:score_plot_end],
            exec_horizon = exec_horizon,
            cp_band = cp_band,
            ymin = vmin,
            ymax = vmax,
            xmax = len(scores)
        )
        
        # Set a title for the entire figure including rollout meta information.
        fig.suptitle(
            f"{r.task_description}\nEp {r.episode_idx}, Succ {r.episode_success}, Frame {j}"
        )
        fig.tight_layout()
        fig.canvas.draw()
        
        # Convert the figure to a numpy array.
        plot_img = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        frames.append(plot_img)
        
    # Define the save path and save the video.
    save_path = os.path.join(
        save_folder, f"{save_name}.mp4",
    )
    imageio.mimsave(save_path, frames, fps=fps)
    

def eval_save_videos_functional_cp(
    cfg,
    model, 
    rollouts_by_split_name: dict,
    dataloader_by_split_name: dict,
    save_folder: str,
    alpha: float = 0.2,
    calib_split_names=["val_seen"], 
    test_split_names=["val_unseen"],
):
    """
    Evaluate model scores, compute the conformal prediction band, and save videos
    for the test split rollouts using multiprocessing.
    """
    #### Forward the model and compute the scores ####
    scores_by_split_name = {}
    for split, dataloader in dataloader_by_split_name.items():
        # Re-create a dataset to disable shuffling.
        dataloader = DataLoader(
            dataloader.dataset, 
            batch_size=cfg.model.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        with torch.no_grad():
            scores, valid_masks, _ = model_forward_dataloader(model, dataloader)
        scores = scores.detach().cpu().numpy()
        seq_lengths = valid_masks.sum(dim=-1).cpu().numpy()  # (B,)
        scores_by_split_name[split] = [
            scores[i, :int(seq_lengths[i])] for i in range(len(seq_lengths))
        ]
        
    df, cp_bands_by_alpha = eval_functional_conformal(
        rollouts_by_split_name, scores_by_split_name, "model",
        calib_split_names=calib_split_names, test_split_names=test_split_names
    )
    
    # Retrieve the CP band for the given alpha.
    cp_band = cp_bands_by_alpha[alpha][0]  # Shape: (T,)
    
    # Gather test rollouts and their corresponding scores.
    test_rollouts = sum([rollouts_by_split_name[k] for k in test_split_names], [])
    test_scores = sum([scores_by_split_name[k] for k in test_split_names], [])
    
    # Compute overall score range for consistent plotting.
    vmin = min(s.min() for s in test_scores)
    vmax = max(s.max() for s in test_scores)
    
    if cfg.train.eval_save_video_multiproc:
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = []
            for rollout, scores in zip(test_rollouts, test_scores):
                futures.append(
                    executor.submit(
                        process_single_rollout_functional, 
                        rollout.get_simple_meta(), scores, cp_band,
                        cfg, alpha, vmin, vmax, save_folder
                    )
                )
                
            # as each future completes, tqdm updates
            for future in tqdm(as_completed(futures), total=len(futures),
                            desc="Processing rollouts"):
                # will also re‐raise any exception from the worker
                future.result()
                
    else:
        for rollout, scores in tqdm(zip(test_rollouts, test_scores), total=len(test_rollouts), desc="Processing rollouts"):
            try:
                process_single_rollout_functional(
                    rollout.get_simple_meta(), scores, cp_band, cfg, alpha, vmin, vmax, save_folder
                )
            except Exception as e:
                logging.error(f"Error processing rollout {rollout.task_id}/{rollout.episode_idx}: {e}")
                logging.exception("Exception occurred")
                continue
            
            # TODO: This is for debugging. Remove it later.
            break
    
    return df  # Return the dataframe if needed for further analysis.