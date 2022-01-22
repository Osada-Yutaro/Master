PY := python

.PHONY: detection_train, re_ID_train, speed

detection_train:
		$(PY) src/train/detect_train.py

re_ID_train:
		$(PY) src/train/re_ID_train.py

speed_det:
		$(PY) src/train/speed_det.py

speed_re:
		$(PY) src/train/speed_re.py
