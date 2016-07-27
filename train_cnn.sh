if [ $# -eq 2 ]
  then
  ACT=$1
  GPU=$2
else
  if [ -z "${1}" ]
    then
      GPU=0
  fi
  source act.sh
fi

echo "ACT = ${ACT}"
echo "GPU = ${GPU}"

PREFIX="cnn"
PYTHONPATH=$PWD/../caffe_dir/install/python

mkdir -p ${PREFIX}
python2 ../train.py --model 1 --prefix $PREFIX"/1step" --lr 0.0001 --num_act ${ACT} --T 5 --K 4 --num_step 1 --batch_size 32 --test_batch_size 50 --gpu $GPU --num_iter 1500000 --mean=mode.binaryproto
#python2 ../train.py --model 1 --prefix $PREFIX"/3step" --lr 0.00001 --num_act ${ACT} --T 7 --K 4 --num_step 3 --batch_size 8 --test_batch_size 50 --gpu $GPU --weights $PREFIX"/1step_iter_1500000.caffemodel.h5" --num_iter 1000000
#python2 ../train.py --model 1 --prefix $PREFIX"/5step" --lr 0.00001 --num_act ${ACT} --T 9 --K 4 --num_step 5 --batch_size 8 --test_batch_size 50 --gpu $GPU --weights $PREFIX"/3step_iter_1000000.caffemodel.h5" --num_iter 1000000
