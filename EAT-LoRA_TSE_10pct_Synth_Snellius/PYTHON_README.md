# What each python file does

This document explains the python scripts in this project shortly. The scripts work together as a pipeline. First synthetic audio is built from real data. Then tse and eat lora are trained. Then knn evaluation runs on the evaluation dataset. Finally plots and latex tables are made from the csv results.

The main snellius entry point is `run_full_pipeline.slurm` in this folder. All python scripts live in the `code/` subfolder.

## audio_io.py

This small helper file is used everywhere audio files are read.

The function `load_wav_mono` opens a wav file and returns mono audio at 16 khz. It first tries the soundfile library because dcase wav files sometimes use a special format. If that fails it uses torchaudio. If the sample rate is not 16 khz it resamples the audio.

The function `load_wav_mono_1d` does the same but returns a flat one dimensional tensor. Tse training uses this shape.

## synthesize_dcase_data.py

This script creates synthetic training data from the real additional dataset. It reads normal train clips from each machine folder and writes new wav files into an output folder.

The class `SourceClip` stores one source file path plus its machine attributes.

The function `_load_source_clips` walks a machine folder and collects train wav paths together with label info from `attributes_00.csv`.

The function `_make_normal_variant` takes a real clip and makes a slightly changed normal version. It may stretch time a little, shift the signal, add soft noise and change volume. If you set a fixed snr such as 15 db the noise level follows that setting.

The function `_make_anomaly_variant` starts from a normal variant and adds extra strange sounds so the file looks like an anomaly test clip.

The function `_synthesize_machine_outputs` loops over all source clips for one machine and writes train normal test normal and test anomaly wav files for each copy.

The function `_build_output_rows` updates the attributes csv so the new synthetic files have correct labels.

The function `main` reads command line arguments, finds the input dataset root, loops over all machines and writes the full synthetic dataset tree.

## tse_snellius.py

This module holds the tse neural network and shared helpers. Tse means target signal enhancement. The idea is to make a noisy recording sound a bit cleaner before embedding.

The function `list_machines` returns all machine folder names under a data root.

The function `list_target_wavs` lists normal train wav files for one machine. These are treated as clean target sounds during tse training.

The class `TSEWaveformDataset` loads short fixed length pieces of clean audio for one machine.

The class `CrossMachineNoiseBank` collects noise clips from other machines. During training the model hears the target machine mixed with realistic background noise.

The function `mix_at_random_snr` combines a clean clip and a noise clip at a random signal to noise ratio.

The function `neg_snr_loss` is the training loss. It measures how close the network output is to the clean target. Lower loss means better reconstruction.

The class `TSEMaskNet` is the actual network. In the forward pass it converts the waveform to a spectrogram with stft. A small u net predicts a mask between zero and one on the magnitude bins. The mask is applied and the result goes back to a waveform with inverse stft. The original phase from the input is kept so the sound stays natural.

The function `enhance_waveform` runs a trained tse model on one waveform at evaluation time.

The function `load_tse_models` loads all `{machine}.pt` checkpoint files from a folder into a dictionary keyed by machine name.

## train_tse_snellius.py

This script trains one tse model per machine.

The function `train_one_machine` collects clean train wavs for one machine, mixes them with cross machine noise in each batch, runs `TSEMaskNet`, updates weights with adamw and saves `{machine}.pt` in the save folder.

The function `train_one_machine_mixed` does the same but each training step randomly picks audio from the primary data root or the secondary data root. In our pipeline the primary root is synthetic data and the secondary is raw data. The default mix uses about ten percent from synth and ninety percent from raw.

The function `main` parses arguments, finds all machines, optionally opens a loss csv file and calls the training function for each machine.

## train_wang2025_snellius.py

This script fine tunes the eat audio model with lora layers for dcase anomaly detection. Training always runs for 6000 steps.

The function `_iter_disk_train_records` lists train wav files from disk and attaches attribute labels from the csv. This works even when csv file names are longer than the short names on disk.

The function `_core_key` extracts a shared key from a file name so disk files and csv rows can be matched.

The function `get_dcase_num_classes` counts how many unique label combinations exist in the training set.

The class `DCASELabelEncoder` turns a machine name domain and attribute values into an integer class id used during training.

The function `create_master_dataframe` builds one pandas table with all training file paths and label fields.

The function `_validate_training_df` checks before training starts that every row has a valid wav file and a known label.

The class `EATDCASEDataset` is the pytorch dataset. For each index it loads a wav as mono 16 khz, optionally runs tse if configured, converts the waveform to a mel spectrogram with kaldi fbank, pads or crops to 1024 frames and normalizes the mel the same way as during evaluation.

The class `ArcFaceLoss` is the classification head. It pushes embeddings toward the correct machine class and away from other classes.

The class `EATAnomalousTrainer` wraps the eat backbone and returns an embedding vector from a mel input.

The functions `load_checkpoint` and `save_checkpoint` resume or store training state including lora weights arcface weights optimizer and step number. Loading checks that saved labels still match the current dataset.

The function `main` builds the dataset and label encoder, loads the pretrained eat model from hugging face, attaches lora, runs the training loop with optional mixing of two data roots and writes checkpoints every few thousand steps plus an optional loss csv.

## eval_wang2025_knn_snellius.py

This script runs knn anomaly detection with a trained eat lora checkpoint. It is used for development data and is imported by the dcase evaluation script.

The function `_load_eat_lora_from_checkpoint` loads the frozen eat base model plus saved lora weights from a checkpoint file.

The function `load_mel_from_wav_path` loads a wav optionally enhances it with tse and returns a mel tensor ready for the eat model.

The function `embed_mel_batch` passes a batch of mel spectrograms through eat and returns l2 normalized embedding vectors.

The function `_embed_train_bank` embeds all normal train wavs for one machine to build the reference library.

The function `_embed_test_queries` embeds all test wavs for one machine.

The function `knn_min_cosine_distance_scores` compares each test embedding to all bank embeddings using cosine similarity. The anomaly score is one minus the best match. A higher score means the test clip looks less like normal training data.

The functions `auc_pauc_all` and `auc_domain_like_evaluator` compute auc and partial auc metrics the same way as the official dcase evaluator helpers.

The function `main` loops over machines builds banks embeds tests writes scores and saves one summary csv row per machine with auc values.

## eval_wang2025_dcase_eval_snellius.py

This file lives in the `code/` folder. It runs knn on the official dcase evaluation dataset where test file names are anonymous.

The function `list_train_normal_wavs` lists normal train wav file names for the knn bank from the additional training dataset.

The function `list_eval_test_wavs` lists all test wav files for one machine from the evaluation dataset.

The function `embed_wav_paths` loads many wav paths embeds them in batches and returns a numpy matrix of vectors.

The function `threshold_from_train_bank` picks a decision threshold from the bank embeddings for submission files.

The function `write_submission_csv` writes dcase score and decision csv files under the evaluator teams folder.

The function `metrics_from_ground_truth` reads official ground truth csv files and computes auc and pauc per machine for our anomaly scores.

The function `run_official_evaluator` optionally calls the official `dcase2025_task2_evaluator.py` script to double check scores.

The function `main` loads eat and optional tse models loops over all evaluation machines builds the bank from raw additional train embeds evaluation test clips writes submission csvs writes the knn summary csv and prints overall metrics.

## export_knn_latex_table.py

This file lives in the `code/` folder. It turns a knn result csv into a latex table for a report.

The function `format_row` formats one machine row with auc columns as a latex table line.

The function `main` reads the csv orders machines writes commented latex code to a tex file and prints it to the terminal.

## plot_knn_per_machine.py

This script draws a bar chart of knn auc scores per machine.

The function `_pick_csv` chooses which result csv to use when several exist for example the highest training step.

The function `_filter_paths` separates baseline runs without tse from runs that used tse on the waveform.

The function `main` reads the chosen csv and saves a png bar chart.

## plot_knn_performance.py

This script plots overall knn performance across training checkpoints.

The function `build_performance_table` reads all matching knn csv files extracts the checkpoint step and computes a harmonic mean score per file.

The function `_hmean_from_summary` calculates the harmonic mean over auc source auc target and pauc per machine.

The function `main` builds the table saves it as csv and draws a line plot of hmean against step for baseline and tse variants.

## plot_training_loss.py

This script plots eat training loss over steps from the csv written during training.

The function `_moving_average` smooths the loss curve for easier reading.

The function `main` reads the loss csv and saves a png plot optionally with learning rate on a second axis.

## plot_synth_snr_aggregate.py

This script is used for snr sweep experiments. It scans several run folders each trained with a different synthetic noise snr setting and builds one summary table and plot comparing knn performance across snr values. It is not part of the default evaluation pipeline but helps compare how strong synthetic noise should be.

## synthesize_sorama_data.py

This is an older or alternate synth script for a different folder layout used in earlier sorama experiments. It walks wav files in a flat or category based tree and writes synthetic normal and anomaly copies. The main evaluation pipeline uses `synthesize_dcase_data.py` instead because that script follows the dcase machine train test folder structure.

## How the Files Connect

When you run the full pipeline on snellius the shell scripts call the python files in `code/` in order. `synthesize_dcase_data.py` builds synthetic additional train data. `train_tse_snellius.py` and `train_wang2025_snellius.py` train on a mix of synth and raw data. `eval_wang2025_dcase_eval_snellius.py` uses helpers from `eval_wang2025_knn_snellius.py` and optional models from `tse_snellius.py`. All of them read audio through `audio_io.py`. After evaluation `plot_knn_per_machine.py` `plot_knn_performance.py` and `plot_training_loss.py` turn csv results into figures and `export_knn_latex_table.py` can produce a latex table for documentation.
