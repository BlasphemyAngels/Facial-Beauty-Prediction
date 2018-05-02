
[ -f "data/train.txt" ] && {
	python data.py --data_list_path=train.txt --store_path=train.tfrecord
}

[ -f "data/test.txt" ] && {
	python data.py --data_list_path=test.txt --store_path=test.tfrecord
}
