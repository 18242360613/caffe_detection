cd /home/qif/caffe_ssd/caffe/
./build/tools/caffe train \
--solver="/home/qif/PycharmProjects/caffe_detection/working_dir/meter/prototxt/SSD_512x512/solver.prototxt" \
--weights="/home/qif/PycharmProjects/caffe_detection/checkpoints/VGG_ILSVRC_16_layers_fc_reduced.caffemodel" \
--gpu 0,1 2>&1 | tee /home/qif/PycharmProjects/caffe_detection/working_dir/meter/job/SSD_512x512/VGG_meter_SSD_512x512.log
