.PHONY: all install lint format download train export demo clean

# Define some variables
MODEL_NAME := solar_bb_train
MODEL_WEIGHTS := runs/detect/$(MODEL_NAME)/weights
DATA_CONFIG := data/solar_bb.yaml

# ====================================================================================
# PROJECT SETUP & DATA
# ====================================================================================

install:
	pip install -r requirements.txt

download:
	mkdir -p data/solar_bb
	kaggle datasets download -d fxmikf/solar-panel-bounding-boxes-621 -p data/solar_bb --unzip

# ====================================================================================
# ML WORKFLOW
# ====================================================================================

train:
	yolo train data=$(DATA_CONFIG) model=yolov8n.pt epochs=30 imgsz=640 device=0 project=runs/detect name=$(MODEL_NAME)

export:
	yolo export model=$(MODEL_WEIGHTS)/best.pt format=engine device=0

demo:
	python demo.py --model $(MODEL_WEIGHTS)/best.onnx --image data/solar_bb/images/test/$(ls data/solar_bb/images/test | head -n 1)
# ====================================================================================
# QUALITY CONTROL
# ====================================================================================

lint:
	mypy . --strict
	ruff check .

format:
	black .
	ruff check . --fix

# ====================================================================================
# CLEANUP
# ====================================================================================

clean:
	rm -rf runs/
	rm -rf data/solar_bb/
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -f *.pt *.onnx *.engine *.jpg