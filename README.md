# test.ai Classifier Builder

This is the classifier and training/running scripts for the test.ai elements classifier. This is used to build a model for the testai appium-classifier-plugin(https://github.com/testdotai/appium-classifier-plugin).

## System setup
You will need to install Tensorflow(www.tensorflow.org), Google's open source AI framework.

```pip install tensorflow```

## Build Classifier
To build the classifier from the provided images run the following. 

```python retrain.py --image_dir training_images/ --output_graph output/saved_model.pb --output_labels output/saved_model.pbtxt --how_many_training_steps 4000 --learning_rate 0.30 --testing_percentage 25 --validation_percentage 25 --eval_step_interval 50 --train_batch_size 2000 --test_batch_size -1 --validation_batch_size -1 --bottleneck_dir /tmp/bottleneck```

## Run Classifier
To use the classifier to classify images, run the script under `sample_run/` directory.

```python run_model.py --image cart.png```
