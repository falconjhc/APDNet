# APD-Net with Pathology

Implementation of the **APDNet** model to perform disentanglement of anatomical, modality, and pathology information in medical images. For further details please see our [paper], accepted in [MICCAI-2020 Workshop: DART].

Python dependencies to run the code is listed in the file ;requirements.txt'.

The structure of this project is the following:

* **configuration**: package containing configuration parameters for running an experiment.
* **layers**: package with custom Keras layers
* **loaders**: package with data loaders
* **models**: package with the SDNet model and other Keras models
* **model_executors**: package with scripts for running an experiment
* **callbacks**: package with Keras callbacks for printing images and losses during training
* **DataProcess**: package with some of the data preprocess codes for some public datasets


To define a new data loader, extend class `base_loader.Loader`, and register the loader in `loader_factory.py`. The datapath is specified in `parameters.py`.

To run an experiment, execute `experiment.py`, passing the configuration filename, the split number as runtime parameters, and the pathology annotation amount 'l-mix a-b', where 'a' denotes the amount of volumes among the full training data, while 'b' represents the amount of data in each volume:
```
python experiment.py --config myconfiguration --split 0 --l_mix 1-1
```

To run an test, execute `experiment.py` as follows:
```
python experiment.py --config myconfiguration --split 0 --l_mix 1-1 --test True
```

Citation

If you use this code for your research, please cite our paper:

```
@incollection{jiang2020pathology,
  title={Semi-supervised Pathology Segmentation with Disentangled Representations},
  author={Haochuan, Jiang and Chartsias, Agisilaos and Papanastasiou, Giorgos and Semple, Scott and Dweck, Mark and and Dharmakumar, Rohan and Tsaftaris, Sotirios A},
  booktitle={Domain Adaptation and Representation Transfer},
  year={2020},
  publisher={Springer}
}
```
 
[Keras]: https://keras.io/
[tensorflow]: https://www.tensorflow.org/
[MICCAI-2020]: https://miccai2020.org/en/
[DART-2020]: https://sites.google.com/view/dart2020/
