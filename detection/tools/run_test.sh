bash dist_test.sh \
	/app/AI/yoonchul/Documents/LAB_414/object_detection/object_detection_yoon_c_c/configs/fouriervit/cascade_mask_rcnn_fouriervit_tiny_giou_coco.py \
	/app/AI/yoonchul/Documents/LAB_414/models/fouriervit/Detection/tiny/cascade_fouriervit_giou_tiny_52_4_1344_896.pth 1 \
	--out results.pkl --eval bbox 
