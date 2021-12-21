PY := python

.PHONY: detection_train

detection_train:
		$(PY) src/train/detect_train.py

re_ID_train:
		$(PY) src/train/re_ID_train.py
