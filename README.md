# Companion repository to "On a time-frequency blurring operator with applications in data augmentation"

This repository contains code to repeat the main experiment of the paper. Call `download_and_save_data()` to download the `SpeechCommands V2` dataset. All required Python packages should be self-explanatory. The `improve()` function works on the `accs` pickle file to add more examples of accuracies. Once training is complete, the `export_results()` prints accuracies and standard errors in a LaTeX-friendly way.
