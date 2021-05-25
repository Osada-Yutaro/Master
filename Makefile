PY := python

.PHONY: detection_train

detection_train:
		$(PY) src/train/detect_train.py
