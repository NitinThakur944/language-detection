# Fixes applied

- Retrained the detector on all 17 labels in the included dataset, rather than only six languages.
- Normalized the dataset label misspellings `Portugeese` and `Sweedish` to `Portuguese` and `Swedish`.
- Preserved Unicode letters while training, including non-Latin scripts.
- Increased character n-gram coverage and trained/evaluated with a stratified holdout split.
- Added common greeting handling and direct script identification for Hindi, Tamil, Kannada, Malayalam, Arabic, Greek, and Russian.
- Translation auto-detection now uses the translation provider's `auto` source option rather than relying on the prediction model to pick a source language.
- Translation still needs a working internet connection; third-party translation providers can occasionally be unavailable.
- Very short or ambiguous text cannot be guaranteed correct; use complete sentences for best accuracy.
