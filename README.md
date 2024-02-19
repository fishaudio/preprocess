# Fish Audio Preprocessor

[![PyPI Version](https://img.shields.io/pypi/v/fish-audio-preprocess.svg)](https://pypi.python.org/pypi/fish-audio-preprocess)

[中文文档](README.zh.md)

This repo contains some scripts for audio processing. Main features include:

- [x] Video/audio to wav
- [x] Audio vocal separation
- [x] Automatic audio slicing
- [x] Audio loudness matching
- [x] Audio data statistics (supports determining audio length)
- [x] Audio resampling
- [x] Audio transcribe (.lab)
- [ ] Audio transcribe via WhisperX

([ ] indicates not completed, [x] indicates completed)

**This code has been tested on Ubuntu 22.04 / 20.04 + Python 3.10. If you encounter problems on other versions, feedback is welcome.**

## Getting Started:

```
pip install -e . 
fap --help
```

## Reference

- [Batch Whisper](https://github.com/Blair-Johnson/batch-whisper)

